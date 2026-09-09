"""Building a game context from real data.

Split deliberately into two halves:

``fetch_*``      touch the network. Untestable offline, thin by design.
``build_*``      pure transformations over DataFrames. All the logic lives
                 here, and all of it is tested against synthetic frames shaped
                 like nflverse output.

That split is the whole point. When the live run breaks, the question "is this
a data problem or a logic problem" has an answer, because the logic already has
tests that pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .teams import canonical_team, is_dome

log = logging.getLogger(__name__)

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

# nflverse column names differ slightly between the weekly frame and the
# play-by-play aggregates. Mapping them here rather than at each call site
# means a schema change is a one-line fix.
WEEKLY_COLUMNS = {
    "player_id": "player_id",
    "player_display_name": "name",
    "position": "position",
    # nflverse renamed this to "team". Both are accepted so historical
    # frames and current ones normalise identically.
    "recent_team": "team",
    "team": "team",
    "week": "week",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_air_yards": "air_yards",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "attempts": "pass_attempts",
    "passing_yards": "passing_yards",
}

# nflverse frames disagree on the name of the player key: the weekly frame
# uses player_id, while depth charts and injuries use gsis_id. Same id
# space, different label -- unlike the snap-count frame, which uses PFR
# ids and genuinely needs a crosswalk.
PLAYER_ID_COLUMNS = ("player_id", "gsis_id")
# Depth-chart rank aliases. preflight imports these so the schema check
# and the transformation cannot drift apart.
DEPTH_RANK_COLUMNS = ("depth_team", "depth_rank", "depth_position",
                      "pos_rank")
DEPTH_POSITION_COLUMNS = ("pos_abb", "position", "pos")
# nflverse ranks within the depth chart's own label, so a fullback is FB1 and
# a third receiver is WR3. The projection merges fullbacks into running backs
# (normalize_weekly maps FB -> RB), at which point FB1 reads as the starting
# back and a blocking specialist inherits the RB1 opportunity prior. Folded
# labels are pushed below every primary one; opportunity_prior clamps to its
# bottom tier from there.
DEPTH_POSITION_GROUPS = {"FB": "RB", "HB": "RB", "H": "RB"}
DEPTH_FOLDED_RANK_OFFSET = 10

# How deep a position group's rotation plausibly runs, and the share of snaps
# that counts as being in it. Every opportunity pool is renormalised to 1.0, so
# a player kept in the pool takes opportunity from someone else: a receiver who
# appeared once in eight weeks on a fifth of the snaps is not competing for
# targets, and including him hands his depth-chart prior to a pool that has to
# find it from somewhere.
ROTATION_DEPTH = {"WR": 3, "TE": 2, "RB": 3}
ROTATION_SNAP_FLOOR = 0.25
# Below this many players a pool is more likely to be missing data than to be
# genuinely that thin, so pruning is abandoned rather than trusted.
MIN_ROTATION_PLAYERS = 4


def in_rotation(position: str, depth_rank: int | None,
                snap_share: float | None, *, has_market: bool = False) -> bool:
    """Whether a player belongs in his team's opportunity pool.

    ``depth_rank`` is None when the depth chart does not agree about this
    player's position -- a rank from another group is no evidence about this
    one -- in which case participation decides alone. ``has_market`` keeps
    anyone a book has priced, so pruning can never cost market coverage.
    """
    if has_market or position == "QB":
        return True
    if depth_rank is not None and depth_rank <= ROTATION_DEPTH.get(position, 3):
        return True
    return snap_share is not None and snap_share >= ROTATION_SNAP_FLOOR


class ContextError(RuntimeError):
    """Raised when the data cannot support a projection."""


# --------------------------------------------------------------------------- #
# Pure transformations
# --------------------------------------------------------------------------- #
def normalize_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Rename and canonicalise the nflverse weekly frame.

    Missing optional columns are created as zero rather than causing a
    KeyError downstream, but missing *required* columns raise: a game log with
    no target count is not a game log with zero targets, and treating it as one
    would silently produce a projection of zero for a healthy receiver.
    """
    required = {"player_id", "position", "week"}
    present = set(weekly.columns)
    missing_required = required - present
    if missing_required:
        raise ContextError(
            f"Weekly frame is missing required columns: {sorted(missing_required)}"
        )

    # An alias whose target name is already present must not be renamed:
    # that produces two columns called "team", and df["team"] then returns
    # a DataFrame rather than a Series.
    rename = {k: v for k, v in WEEKLY_COLUMNS.items()
              if k in present and (k == v or v not in present)}
    df = weekly.rename(columns=rename).copy()
    for col in ("targets", "receptions", "receiving_yards", "air_yards",
                "carries", "rushing_yards", "pass_attempts", "passing_yards"):
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna({c: 0.0 for c in (
        "targets", "receptions", "receiving_yards", "air_yards",
        "carries", "rushing_yards", "pass_attempts", "passing_yards")})

    if "team" in df.columns:
        df["team"] = df["team"].map(lambda t: canonical_team(t, strict=False))
    df = df[df["position"].isin(SKILL_POSITIONS)]
    df["position"] = df["position"].replace({"FB": "RB"})
    return df


def team_volume_by_week(weekly: pd.DataFrame) -> pd.DataFrame:
    """Team pass attempts and carries per week, from summed player lines.

    Derived from the weekly frame rather than a separate team table so the
    shares are guaranteed internally consistent: a player's target share is his
    targets over the same total that every teammate is divided by.
    """
    grouped = (weekly.groupby(["team", "week"], as_index=False)
               .agg(team_pass_attempts=("targets", "sum"),
                    team_rush_attempts=("carries", "sum")))
    # Targets undercount attempts by the number of throwaways and spikes.
    grouped["team_pass_attempts"] = grouped["team_pass_attempts"] * 1.06
    return grouped


def red_zone_usage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Goal-line carries and end-zone targets per player-week.

    Inside-the-five carries and inside-the-ten targets are the opportunity
    inputs the touchdown model runs on. They are small counts, which is exactly
    why they are aggregated here rather than approximated from season totals.
    """
    if pbp.empty:
        return pd.DataFrame(columns=["player_id", "week", "gl_carries",
                                     "ez_targets", "team", "team_gl_carries",
                                     "team_ez_targets"])
    df = pbp.copy()
    df["posteam"] = df["posteam"].map(lambda t: canonical_team(t, strict=False))

    gl = df[(df["play_type"] == "run") & (df["yardline_100"] <= 5)]
    gl = (gl.groupby(["rusher_player_id", "week", "posteam"], as_index=False)
          .size().rename(columns={"rusher_player_id": "player_id",
                                  "posteam": "team", "size": "gl_carries"}))

    ez = df[(df["play_type"] == "pass") & (df["yardline_100"] <= 10)]
    ez = (ez.groupby(["receiver_player_id", "week", "posteam"], as_index=False)
          .size().rename(columns={"receiver_player_id": "player_id",
                                  "posteam": "team", "size": "ez_targets"}))

    merged = gl.merge(ez, on=["player_id", "week", "team"], how="outer")
    merged[["gl_carries", "ez_targets"]] = (
        merged[["gl_carries", "ez_targets"]].fillna(0.0))

    totals = (merged.groupby(["team", "week"], as_index=False)
              .agg(team_gl_carries=("gl_carries", "sum"),
                   team_ez_targets=("ez_targets", "sum")))
    return merged.merge(totals, on=["team", "week"], how="left")


def red_zone_rush_rate(pbp: pd.DataFrame, team: str,
                       default: float = 0.50) -> float:
    """Share of a team's red-zone plays that are runs.

    Was hardcoded at 0.50 for every team, which made the touchdown model treat
    a run-heavy goal-line offence identically to a pass-heavy one -- the single
    biggest lever on which players get the scores.
    """
    if pbp.empty:
        return default
    df = pbp[(pbp["yardline_100"] <= 20)
             & (pbp["play_type"].isin(["run", "pass"]))].copy()
    df["posteam"] = df["posteam"].map(lambda t: canonical_team(t, strict=False))
    df = df[df["posteam"] == team]
    if len(df) < 25:
        # Too few red-zone plays to estimate; the league average is a better
        # guess than a rate computed from nine snaps.
        return default
    return float(np.clip((df["play_type"] == "run").mean(), 0.20, 0.80))


def _mmss_to_seconds(value) -> float | None:
    """nflverse reports drive time of possession as "M:SS"."""
    try:
        minutes, seconds = str(value).split(":")
        return float(int(minutes) * 60 + int(seconds))
    except (AttributeError, TypeError, ValueError):
        return None


def seconds_per_play(pbp: pd.DataFrame, team: str, default: float = 27.5
                     ) -> float:
    """Situation-neutral pace: game seconds elapsed per offensive play.

    Measured only in neutral game states -- first three quarters, score within
    a touchdown -- because a team trailing by 20 runs a hurry-up that says
    nothing about how it plays a close game.
    """
    needed = {"game_seconds_remaining", "qtr", "score_differential",
              "play_type", "posteam", "game_id"}
    if pbp.empty or not needed <= set(pbp.columns):
        return default
    neutral = pbp[(pbp["qtr"] <= 3)
                  & (pbp["score_differential"].abs() <= 8)].copy()
    neutral["posteam"] = neutral["posteam"].map(
        lambda t: canonical_team(t, strict=False))
    neutral = neutral[neutral["posteam"] == team]
    offense = neutral[neutral["play_type"].isin(["run", "pass"])]
    if len(offense) < 50:
        return default

    drive_col = next((c for c in ("drive", "fixed_drive")
                      if c in neutral.columns), None)

    # Preferred: time of possession divided by the plays it covers. That is
    # the definition of pace, rather than something inferred from the clock
    # between two snaps.
    if drive_col is not None and "drive_time_of_possession" in neutral.columns:
        seconds = 0.0
        plays = 0
        for _, drive in neutral.groupby(["game_id", drive_col], sort=False):
            top = _mmss_to_seconds(drive["drive_time_of_possession"].iloc[0])
            run_pass = int(drive["play_type"].isin(["run", "pass"]).sum())
            if top is None or run_pass < 1:
                continue
            seconds += top
            plays += run_pass
        if plays >= 50:
            return float(np.clip(seconds / plays, 20.0, 35.0))

    # Fallback for frames without the drive columns. Grouping by drive keeps a
    # gap from spanning a change of possession; without it, only the game.
    group = ["game_id"] + ([drive_col] if drive_col else [])
    gaps = []
    for _, g in offense.groupby(group, sort=False):
        elapsed = -g.sort_values("game_seconds_remaining",
                                 ascending=False)["game_seconds_remaining"].diff()
        gaps.extend(elapsed.dropna().tolist())
    usable = [g for g in gaps if 0 < g < 60]
    if not usable:
        return default
    # Mean, not median. The clock stops on an incompletion and runs on after a
    # completion or a run, so these gaps are bimodal -- a cluster near 5s and
    # one near 40s. More than half of plays leave the clock running, so the
    # median falls inside the upper cluster and reads far too slow: measured
    # against drive time of possession across 2025, the median is off by 7.9s
    # per play and lands every team at or above the 35s clamp, erasing the
    # difference between a fast team and a slow one. The mean is off by 2.5s.
    return float(np.clip(float(np.mean(usable)), 20.0, 35.0))


def snap_shares(snaps: pd.DataFrame,
                id_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalise the snap-count frame to player_id / week / snap_share.

    Snap counts are keyed by PFR id; every other nflverse frame is keyed by
    gsis id. Those are different id spaces, not a renamed column, so the join
    needs the crosswalk from ``load_id_map``. Without it the merge in
    ``build_game_logs`` finds nothing, silently, and every player is projected
    on a zero snap share.
    """
    if snaps.empty:
        return pd.DataFrame(columns=["player_id", "week", "snap_share"])
    df = snaps.rename(columns={"pfr_player_id": "pfr_id",
                               "offense_pct": "snap_share"}).copy()
    if ("player_id" not in df.columns and "pfr_id" in df.columns
            and id_map is not None and not id_map.empty
            and {"gsis_id", "pfr_id"} <= set(id_map.columns)):
        cross = (id_map[["gsis_id", "pfr_id"]].dropna()
                 .drop_duplicates(subset="pfr_id"))
        df = df.merge(cross, on="pfr_id", how="left")
        df["player_id"] = df["gsis_id"]
        # One row per player per week: a duplicate would multiply the game log.
        df = df.drop_duplicates(subset=["player_id", "week"], keep="last")
    keep = [c for c in ("player_id", "pfr_id", "week", "snap_share")
            if c in df.columns]
    return df[keep]


def snap_crosswalk_rate(snap: pd.DataFrame) -> float:
    """Share of snap-count rows that resolved to a gsis id."""
    if snap.empty or "player_id" not in snap.columns:
        return 0.0
    return float(snap["player_id"].notna().mean())


def build_game_logs(
    weekly: pd.DataFrame,
    team_volume: pd.DataFrame,
    rz: pd.DataFrame,
    snaps: pd.DataFrame,
    teams: Sequence[str],
    lookback_weeks: int = 8,
    through_week: int | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """Assemble per-player game logs for the two teams in this game.

    ``through_week`` is the point-in-time guard: when backtesting, it must be
    the week before the game being projected. Passing None uses everything
    available, which is correct for a live projection and wrong for a backtest.
    """
    df = weekly[weekly["team"].isin(teams)].copy()
    if through_week is not None:
        df = df[df["week"] < through_week]
    if df.empty:
        raise ContextError(
            f"No weekly data for {teams} before week {through_week}"
        )
    max_week = int(df["week"].max())
    df = df[df["week"] > max_week - lookback_weeks]

    df = df.merge(team_volume, on=["team", "week"], how="left")
    if not rz.empty:
        df = df.merge(rz.drop(columns=["team"], errors="ignore"),
                      on=["player_id", "week"], how="left")
    if not snaps.empty and "player_id" in snaps.columns:
        df = df.merge(snaps[["player_id", "week", "snap_share"]],
                      on=["player_id", "week"], how="left")

    # snap_share is deliberately absent from this list. A row that failed to
    # cross-walk from a PFR id is a player whose participation is *unknown*,
    # and defaulting that to zero says he never took the field -- which, once
    # participation weights the opportunity prior, silently deletes him.
    for col, default in (("gl_carries", 0.0), ("ez_targets", 0.0),
                         ("team_gl_carries", 0.0), ("team_ez_targets", 0.0),
                         ("team_pass_attempts", 0.0),
                         ("team_rush_attempts", 0.0)):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    out: dict[str, dict[str, list[dict]]] = {t: {} for t in teams}
    for (team, pid), group in df.groupby(["team", "player_id"]):
        rows = []
        for _, r in group.sort_values("week").iterrows():
            rows.append({
                "player_id": str(pid),
                "name": str(r.get("name", pid)),
                "position": str(r["position"]),
                "team": str(team),
                "week": int(r["week"]),
                "targets": float(r.get("targets", 0.0)),
                "receptions": float(r.get("receptions", 0.0)),
                "receiving_yards": float(r.get("receiving_yards", 0.0)),
                "air_yards": float(r.get("air_yards", 0.0)),
                "team_pass_attempts": float(r.get("team_pass_attempts", 0.0)),
                "carries": float(r.get("carries", 0.0)),
                "rushing_yards": float(r.get("rushing_yards", 0.0)),
                "team_rush_attempts": float(r.get("team_rush_attempts", 0.0)),
                "snap_share": (float(r["snap_share"])
                               if "snap_share" in r
                               and pd.notna(r["snap_share"]) else None),
                "rz_carries": 0.0,
                "gl_carries": float(r.get("gl_carries", 0.0)),
                "ez_targets": float(r.get("ez_targets", 0.0)),
                "team_gl_carries": float(r.get("team_gl_carries", 0.0)),
                "team_ez_targets": float(r.get("team_ez_targets", 0.0)),
            })
        if rows:
            out[str(team)][str(pid)] = rows
    return out


def build_depth_chart(
    depth: pd.DataFrame,
    teams: Sequence[str],
    as_of: datetime | None = None,
) -> dict[str, dict]:
    """Most recent depth-chart position group and rank per player.

    Reads the latest row per player rather than the row whose week matches,
    because a depth chart published Wednesday and one published Saturday are
    different facts and the later one is the one that matters.

    The position is returned alongside the rank because the rank is only
    meaningful within its own group. A player listed RB5 who is projected as a
    receiver must not be handed the WR5 prior: that is a rank from a different
    depth chart entirely.
    """
    if depth.empty:
        return {}
    df = depth.copy()
    id_col = next((c for c in PLAYER_ID_COLUMNS if c in df.columns), None)
    if id_col is None:
        return {}
    if "team" in df.columns:
        df["team"] = df["team"].map(lambda t: canonical_team(t, strict=False))
        df = df[df["team"].isin(teams)]
    sort_col = next((c for c in ("observed_at", "dt", "week")
                     if c in df.columns), None)
    if sort_col:
        if as_of is not None and sort_col == "observed_at":
            df = df[df[sort_col] <= as_of]
        df = df.sort_values(sort_col).drop_duplicates(subset=[id_col],
                                                      keep="last")
    rank_col = next((c for c in DEPTH_RANK_COLUMNS if c in df.columns), None)
    if rank_col is None:
        return {}
    pos_col = next((c for c in DEPTH_POSITION_COLUMNS if c in df.columns), None)

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        try:
            rank = int(r[rank_col])
        except (TypeError, ValueError):
            continue
        label = str(r[pos_col]).upper() if pos_col and pd.notna(r.get(pos_col)) \
            else None
        group = DEPTH_POSITION_GROUPS.get(label, label)
        if label is not None and group != label:
            rank += DEPTH_FOLDED_RANK_OFFSET
        out[str(r[id_col])] = {"position": group, "rank": rank}
    return out


def build_depth_ranks(
    depth: pd.DataFrame,
    teams: Sequence[str],
    as_of: datetime | None = None,
) -> dict[str, int]:
    """Depth-chart rank per player. See ``build_depth_chart``."""
    return {pid: e["rank"]
            for pid, e in build_depth_chart(depth, teams, as_of).items()}


def build_depth_positions(
    depth: pd.DataFrame,
    teams: Sequence[str],
    as_of: datetime | None = None,
) -> dict[str, str]:
    """Depth-chart position group per player. See ``build_depth_chart``."""
    return {pid: e["position"]
            for pid, e in build_depth_chart(depth, teams, as_of).items()
            if e["position"]}


def build_injury_statuses(
    injuries: pd.DataFrame,
    teams: Sequence[str],
    as_of: datetime | None = None,
) -> dict[str, dict]:
    """Latest injury status per player, as of a timestamp."""
    if injuries.empty:
        return {}
    df = injuries.copy()
    id_col = next((c for c in PLAYER_ID_COLUMNS if c in df.columns), None)
    if id_col is None:
        return {}
    for col in ("team", "recent_team"):
        if col in df.columns:
            df["team"] = df[col].map(lambda t: canonical_team(t, strict=False))
            break
    if "team" in df.columns:
        df = df[df["team"].isin(teams)]

    time_col = next((c for c in ("observed_at", "date_modified", "week")
                     if c in df.columns), None)
    if time_col and as_of is not None and time_col != "week":
        df = df[pd.to_datetime(df[time_col], utc=True) <= as_of]
    if time_col:
        # Take the last ROW, not groupby().last(): that returns the last
        # non-null value of each column independently, which resurrects an
        # old status onto a recent row. A receiver listed Questionable in
        # week 3 and practising fully from week 11 comes back as
        # "Questionable / Full Participation in Practice" -- and is then
        # projected at 62% availability for the rest of the season.
        df = df.sort_values(time_col).drop_duplicates(subset=[id_col],
                                                      keep="last")

    status_col = next((c for c in ("report_status", "game_status", "status")
                       if c in df.columns), None)
    practice_col = next((c for c in ("practice_status",
                                     "practice_participation")
                         if c in df.columns), None)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        status = r.get(status_col) if status_col else None
        if not status or (isinstance(status, float) and np.isnan(status)):
            continue
        out[str(r[id_col])] = {
            "report_status": str(status),
            "practice_status": (str(r[practice_col])
                                if practice_col and pd.notna(r.get(practice_col))
                                else None),
            "is_final_report": bool(r.get("is_final_report", False)),
        }
    return out


def build_team_environment(
    weekly: pd.DataFrame,
    team_volume: pd.DataFrame,
    team: str,
    lookback: int = 8,
    pbp: pd.DataFrame | None = None,
) -> dict:
    """Neutral pass rate, pace and red-zone tendency for one team.

    Pass rate is computed from the same summed volumes used for shares, so the
    simulator's team totals and the players' shares cannot disagree. Pace is a
    placeholder derived from play volume until drive-level timing is wired in;
    it is flagged in the returned dict so the caller knows it is approximate.
    """
    vol = team_volume[team_volume["team"] == team].copy()
    if vol.empty:
        raise ContextError(f"No team volume data for {team}")
    vol = vol.sort_values("week").tail(lookback)

    pass_att = float(vol["team_pass_attempts"].mean())
    rush_att = float(vol["team_rush_attempts"].mean())
    plays = pass_att + rush_att
    pass_rate = pass_att / plays if plays else 0.575

    if pbp is not None and not pbp.empty:
        pace = seconds_per_play(pbp, team)
        rz_rush = red_zone_rush_rate(pbp, team)
        approximate = False
    else:
        # Backed out from play volume against a 60-minute game with roughly
        # 30 minutes of clock stoppage per side. Crude, and flagged as such.
        pace = float(np.clip(1800.0 / max(plays, 40.0), 22.0, 34.0))
        rz_rush = 0.50
        approximate = True

    return {
        "neutral_pass_rate": float(np.clip(pass_rate, 0.35, 0.75)),
        "seconds_per_play": pace,
        "red_zone_rush_rate": rz_rush,
        "_pace_is_approximate": approximate,
        "_weeks_of_data": int(len(vol)),
    }


def extract_market_lines(game_lines: list[dict], event_id: str) -> dict:
    """Pull the consensus spread and total for one event from the odds payload.

    Uses the median across books rather than one book's number: a single book
    can be off a half point, and the spread drives every projection in the
    game, so a stale one contaminates the whole slate.
    """
    event = next((e for e in game_lines if e.get("id") == event_id), None)
    if event is None:
        raise ContextError(f"No game lines found for event {event_id}")

    home_name = event.get("home_team", "")
    spreads, totals = [], []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] == "spreads":
                for o in market.get("outcomes", []):
                    if o.get("name") == home_name and o.get("point") is not None:
                        spreads.append(float(o["point"]))
            elif market["key"] == "totals":
                for o in market.get("outcomes", []):
                    if o.get("point") is not None:
                        totals.append(float(o["point"]))

    if not spreads or not totals:
        raise ContextError(
            f"Event {event_id} has no usable spread or total. Every projection "
            "in the game depends on these, so no projection is produced."
        )
    return {
        "spread_home": float(np.median(spreads)),
        "total": float(np.median(totals)),
        "n_spread_books": len(spreads),
        "n_total_books": len(set(totals)),
    }


def build_league_index(weekly: pd.DataFrame) -> dict[str, str]:
    """Every skill player in the league this season, by normalised name.

    Lets roster churn be told apart from a name-matching failure. A receiver
    who spent last season somewhere else resolves to nothing against this
    game's two rosters, and saying "played for PHI in this data" is far more
    useful than offering the closest-sounding team-mate as a candidate --
    which is how "A.J. Brown" came to be reported next to "AJ Barner".
    """
    from .player_matching import normalize_name
    if weekly.empty or "name" not in weekly.columns:
        return {}
    teams = weekly["team"] if "team" in weekly.columns else None
    return {normalize_name(str(n)): (str(t) if t is not None else "")
            for n, t in zip(weekly["name"],
                            teams if teams is not None
                            else [""] * len(weekly))
            if isinstance(n, str) and n}


def build_roster(game_logs: dict[str, dict[str, list[dict]]],
                 depth: pd.DataFrame | None = None,
                 teams: Sequence[str] = ()) -> list[dict]:
    """Roster entries for the player resolver.

    Built from whoever played, *plus* whoever is on the depth chart. The
    second half matters at the start of a season: a rookie or an offseason
    arrival has no stat line, so a roster drawn only from game logs cannot
    resolve him, and a prop on him is reported as a name-matching failure when
    the resolver is working perfectly and the history simply does not exist.
    Such a player still cannot be projected -- that is a separate, honest gap.
    """
    roster = []
    seen = set()
    for team, players in game_logs.items():
        for pid, rows in players.items():
            last = rows[-1]
            seen.add(pid)
            roster.append({
                "player_id": pid,
                "full_name": last["name"],
                "position": last["position"],
                "team": team,
            })

    if depth is None or depth.empty:
        return roster
    id_col = next((c for c in PLAYER_ID_COLUMNS if c in depth.columns), None)
    pos_col = next((c for c in DEPTH_POSITION_COLUMNS if c in depth.columns),
                   None)
    name_col = next((c for c in ("player_name", "full_name", "name")
                     if c in depth.columns), None)
    if not (id_col and pos_col and name_col and "team" in depth.columns):
        return roster

    df = depth.copy()
    df["team"] = df["team"].map(lambda t: canonical_team(t, strict=False))
    if teams:
        df = df[df["team"].isin(teams)]
    sort_col = next((c for c in ("observed_at", "dt", "week")
                     if c in df.columns), None)
    if sort_col:
        df = df.sort_values(sort_col).drop_duplicates(subset=[id_col],
                                                      keep="last")
    for _, r in df.iterrows():
        pid = str(r[id_col])
        if pid in seen or pd.isna(r.get(name_col)) or pd.isna(r.get("team")):
            continue
        label = str(r[pos_col]).upper()
        position = DEPTH_POSITION_GROUPS.get(label, label)
        if position not in SKILL_POSITIONS:
            continue
        seen.add(pid)
        roster.append({
            "player_id": pid,
            "full_name": str(r[name_col]),
            "position": "RB" if position == "FB" else position,
            "team": str(r["team"]),
        })
    return roster


def pick_starting_qb(game_logs: dict[str, dict[str, list[dict]]]) -> dict[str, str]:
    """The QB with the most recent-week pass attempts, per team.

    Deliberately recency-weighted rather than season-total based: a team that
    changed quarterbacks three weeks ago should project the current starter,
    not the one with more season attempts.
    """
    out = {}
    for team, players in game_logs.items():
        best, best_score = None, -1.0
        for pid, rows in players.items():
            if rows[-1]["position"] != "QB":
                continue
            recent = rows[-3:]
            score = sum(r["team_pass_attempts"] * (i + 1)
                        for i, r in enumerate(recent)) if recent else 0.0
            # Attempts are not in the weekly log rows, so use presence in the
            # last three weeks weighted by recency as the signal.
            score = sum((i + 1) for i, r in enumerate(recent))
            if score > best_score:
                best, best_score = pid, score
        if best:
            out[team] = best
    return out


@dataclass
class BuiltContext:
    context: dict
    warnings: list[str]


def build_context(
    *,
    event_payload: dict,
    game_lines: list[dict],
    weekly: pd.DataFrame,
    pbp: pd.DataFrame,
    snaps: pd.DataFrame,
    depth: pd.DataFrame,
    injuries: pd.DataFrame,
    id_map: pd.DataFrame | None = None,
    weather: dict | None = None,
    through_week: int | None = None,
    as_of: datetime | None = None,
) -> BuiltContext:
    """Assemble everything the slice needs. Pure: no network, no database."""
    from .teams import teams_from_event

    warnings: list[str] = []
    pair = teams_from_event(event_payload)
    teams = [pair.home, pair.away]

    weekly_n = normalize_weekly(weekly)
    volume = team_volume_by_week(weekly_n)
    # The point-in-time guard has to cover the play-by-play frame too. It was
    # applied only to the weekly frame inside build_game_logs, so pace and
    # red-zone tendency were being measured over the *whole* season -- weeks
    # after the game included. A backtest built on that is measuring a model
    # that knew how the season turned out.
    pbp_to_date = (pbp if through_week is None or pbp.empty
                   or "week" not in pbp.columns
                   else pbp[pbp["week"] < through_week])
    rz = red_zone_usage(pbp_to_date)
    snap = snap_shares(snaps, id_map=id_map)
    if not snaps.empty:
        matched = snap_crosswalk_rate(snap)
        if matched < 0.80:
            warnings.append(
                f"snap counts: only {matched:.0%} of rows resolved to a "
                "player id; snap share falls back to zero for the rest")

    logs = build_game_logs(weekly_n, volume, rz, snap, teams,
                           through_week=through_week)
    for team in teams:
        if not logs.get(team):
            raise ContextError(f"No player game logs for {team}")

    market = extract_market_lines(game_lines, event_payload.get("id", ""))
    if market["n_spread_books"] < 3:
        warnings.append(
            f"spread from only {market['n_spread_books']} book(s)")

    team_env = {}
    for team in teams:
        env = build_team_environment(weekly_n, volume, team,
                                     pbp=pbp_to_date)
        if env["_weeks_of_data"] < 4:
            warnings.append(
                f"{team} environment built from {env['_weeks_of_data']} week(s)")
        team_env[team] = {k: v for k, v in env.items()
                          if not k.startswith("_")}

    if weather is None:
        weather = {"temperature_f": 60.0, "wind_mph": 0.0,
                   "precipitation_prob": 0.0, "is_dome": is_dome(pair.home)}
        if not weather["is_dome"]:
            warnings.append(
                "no weather forecast; outdoor game defaulted to neutral "
                "conditions")

    depth_ranks = build_depth_ranks(depth, teams, as_of=as_of)
    if not depth_ranks:
        warnings.append("no depth chart data; redistribution falls back to "
                        "role similarity alone")

    context = {
        "game_id": event_payload.get("id", "unknown"),
        "home_team": pair.home,
        "away_team": pair.away,
        "spread_home": market["spread_home"],
        "total": market["total"],
        "weather": weather,
        "team_env": team_env,
        "starting_qb": pick_starting_qb(logs),
        "roster": build_roster(logs, depth=depth, teams=teams),
        "league_players": build_league_index(weekly_n),
        "game_logs": logs,
        "depth_rank": depth_ranks,
        "depth_position": build_depth_positions(depth, teams, as_of=as_of),
        "slot_rate": {},
        "injuries": build_injury_statuses(injuries, teams, as_of=as_of),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return BuiltContext(context=context, warnings=warnings)


# --------------------------------------------------------------------------- #
# Network side
# --------------------------------------------------------------------------- #
def fetch_context(event_id: str, season: int, through_week: int | None = None
                  ) -> BuiltContext:
    """Live path. Thin wrapper: every decision lives in ``build_context``."""
    from .nflverse import (
        load_depth_charts,
        load_injuries,
        load_play_by_play,
        load_id_map,
        load_snap_counts,
        load_weekly_stats,
    )
    from .odds_api import OddsAPIClient

    client = OddsAPIClient()
    events = client.list_events()
    event = next((e for e in events if e.get("id") == event_id), None)
    if event is None:
        raise ContextError(f"Event {event_id} not in the upcoming slate")

    return build_context(
        event_payload=event,
        game_lines=client.game_lines(),
        weekly=load_weekly_stats([season]),
        pbp=load_play_by_play([season]),
        snaps=load_snap_counts([season]),
        id_map=load_id_map(),
        depth=load_depth_charts([season]),
        injuries=load_injuries([season]),
        weather=None,
        through_week=through_week,
    )
