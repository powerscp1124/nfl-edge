"""nflverse ingestion, via nflreadpy.

nfl_data_py was deprecated in favour of nflreadpy and its repository archived,
so this module targets nflreadpy. Two practical differences:

- nflreadpy returns **Polars** DataFrames. Everything downstream in this
  project is pandas, so the conversion happens here and nowhere else. Column
  selection is done in Polars *before* converting, because a season of
  play-by-play is hundreds of columns wide and converting all of it is the
  slowest step in the ingest.
- Function names changed from ``import_*`` to ``load_*``.

What the data actually covers, since the brief asks for more than exists:

Available here, free, no key
    play-by-play with EPA and success rate, weekly player stats, snap counts,
    depth charts, injury reports with practice participation, rosters,
    schedules, and the Next Gen Stats aggregates (separation, time to throw,
    expected rushing yards, completion probability over expectation).

Available but paid or scraped
    PFF grades (offensive line, coverage, pressure) and player-level man/zone
    splits. Every one is optional: ``FeatureAvailability`` records which are
    present, and a projection built without them is marked lower-confidence
    rather than silently treating the feature as zero.

Not available anywhere public
    Raw player tracking. Route participation is *approximated* from snap counts
    and target data rather than measured. That approximation is a real
    limitation and is labelled as one in the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RECENCY_WINDOWS = {"last3": 3, "last5": 5, "last8": 8, "season": None}

# Columns the projection pipeline actually uses. Selected in Polars before the
# pandas conversion. A column missing from a season's release is skipped with a
# warning rather than raising: nflverse adds and renames fields between
# seasons, and one missing field should not kill an ingest.
PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type",
    "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
    "score_differential", "epa", "success", "wp",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "passing_yards", "receiving_yards", "rushing_yards",
    "air_yards", "yards_after_catch", "complete_pass", "pass_touchdown",
    "rush_touchdown", "sack", "shotgun", "no_huddle", "pass_location",
]


@dataclass
class FeatureAvailability:
    """Which optional feature groups are actually present for this run.

    Consumed by the confidence score. A receiving projection built without
    route data is not wrong, but it is less certain than one built with it, and
    the user should be able to see which.
    """

    play_by_play: bool = False
    snap_counts: bool = False
    ngs_receiving: bool = False
    ngs_passing: bool = False
    ngs_rushing: bool = False
    depth_charts: bool = False
    injuries: bool = False
    pff_line_grades: bool = False
    coverage_splits: bool = False

    def projection_quality(self) -> float:
        """0-1 quality score for the confidence engine."""
        weights = {
            "play_by_play": 0.30, "snap_counts": 0.15, "ngs_receiving": 0.12,
            "ngs_passing": 0.08, "ngs_rushing": 0.05, "depth_charts": 0.12,
            "injuries": 0.12, "pff_line_grades": 0.03, "coverage_splits": 0.03,
        }
        return float(sum(w for k, w in weights.items() if getattr(self, k)))

    def missing(self) -> list[str]:
        return [k for k in vars(self) if not getattr(self, k)]


def _nfl():
    try:
        import nflreadpy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nflreadpy is required for statistical ingestion. Install it with:\n"
            "    pip install nflreadpy\n"
            "Note: nfl_data_py is deprecated and archived; do not use it."
        ) from exc
    return nflreadpy


def to_pandas(frame, source: str, columns: Sequence[str] | None = None
              ) -> pd.DataFrame:
    """Convert a Polars frame to pandas, selecting columns first.

    Tolerant of a pandas frame being passed in, so tests and any future
    provider that already returns pandas work without a separate path.
    """
    if columns and hasattr(frame, "columns"):
        available = [c for c in columns if c in frame.columns]
        missing = set(columns) - set(available)
        if missing:
            log.warning("%s: columns not in this release: %s",
                        source, sorted(missing))
        if available:
            frame = frame[available] if isinstance(frame, pd.DataFrame) \
                else frame.select(available)

    if hasattr(frame, "to_pandas"):
        df = frame.to_pandas()
    elif isinstance(frame, pd.DataFrame):
        df = frame.copy()
    else:
        df = pd.DataFrame(frame)
    df["_source"] = source
    df["_ingested_at"] = datetime.now(timezone.utc)
    return df


# --------------------------------------------------------------------------- #
# Raw pulls
# --------------------------------------------------------------------------- #
def load_play_by_play(seasons: Iterable[int],
                      columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Play-by-play with EPA, success rate, air yards, YAC and situation."""
    return to_pandas(_nfl().load_pbp(list(seasons)), "nflverse_pbp",
                     columns or PBP_COLUMNS)


def load_weekly_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Player game-level statistics. Was ``import_weekly_data``."""
    return to_pandas(_nfl().load_player_stats(list(seasons)), "nflverse_weekly")


def load_team_stats(seasons: Iterable[int]) -> pd.DataFrame:
    return to_pandas(_nfl().load_team_stats(list(seasons)), "nflverse_team")


def load_snap_counts(seasons: Iterable[int]) -> pd.DataFrame:
    return to_pandas(_nfl().load_snap_counts(list(seasons)), "nflverse_snaps")


def load_ngs(seasons: Iterable[int], stat_type: str) -> pd.DataFrame:
    """Next Gen Stats: 'passing', 'rushing' or 'receiving'.

    Carries the tracking-derived metrics the brief asks for -- separation,
    cushion, expected rushing yards, completion probability over expectation --
    at weekly granularity rather than per play.

    UNVERIFIED SIGNATURE: the keyword names here are inferred from the
    documented function list, not from a live call. If this raises a TypeError
    on your first run, check the nflreadpy docs and fix it here.
    """
    return to_pandas(
        _nfl().load_nextgen_stats(seasons=list(seasons), stat_type=stat_type),
        f"nflverse_ngs_{stat_type}",
    )


def load_depth_charts(seasons: Iterable[int]) -> pd.DataFrame:
    return to_pandas(_nfl().load_depth_charts(list(seasons)), "nflverse_depth")


def load_injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Injury statuses and practice participation."""
    return to_pandas(_nfl().load_injuries(list(seasons)), "nflverse_injuries")


def load_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    return to_pandas(_nfl().load_schedules(list(seasons)), "nflverse_schedules")


def load_rosters_weekly(seasons: Iterable[int]) -> pd.DataFrame:
    return to_pandas(_nfl().load_rosters_weekly(list(seasons)),
                     "nflverse_rosters")


def load_id_map() -> pd.DataFrame:
    """Cross-provider player ID map.

    This is why fuzzy name matching stays a fallback rather than the primary
    join. Two active receivers named Michael Thomas is not a hypothetical.
    """
    return to_pandas(_nfl().load_ff_playerids(), "nflverse_ids")


def current_season_week() -> tuple[int, int]:
    """Current season and week from nflreadpy rather than the wall clock."""
    nfl = _nfl()
    return int(nfl.get_current_season()), int(nfl.get_current_week())


def compute_usage(pbp: pd.DataFrame, snaps: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Per-player, per-game usage from play-by-play.

    Produces the opportunity half of every projection: target share, air-yards
    share, rush share, and the red-zone and goal-line splits the touchdown
    model runs on. Efficiency is computed separately so the two can be
    projected independently, which is what makes a projection decomposable
    back into "why".
    """
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    rush_plays = pbp[pbp["play_type"] == "run"].copy()

    team_pass = (pass_plays.groupby(["game_id", "posteam"])
                 .agg(team_attempts=("play_type", "size"),
                      team_air_yards=("air_yards", "sum"))
                 .reset_index())
    team_rush = (rush_plays.groupby(["game_id", "posteam"])
                 .agg(team_carries=("play_type", "size"))
                 .reset_index())

    rec = (pass_plays.groupby(["game_id", "posteam", "receiver_player_id"])
           .agg(targets=("play_type", "size"),
                receptions=("complete_pass", "sum"),
                rec_yards=("receiving_yards", "sum"),
                air_yards=("air_yards", "sum"),
                yac=("yards_after_catch", "sum"),
                rec_td=("pass_touchdown", "sum"),
                rz_targets=("yardline_100", lambda s: int((s <= 20).sum())),
                ez_targets=("yardline_100", lambda s: int((s <= 10).sum())))
           .reset_index()
           .rename(columns={"receiver_player_id": "player_id"}))
    rec = rec.merge(team_pass, on=["game_id", "posteam"], how="left")
    rec["target_share"] = rec["targets"] / rec["team_attempts"]
    rec["air_yards_share"] = rec["air_yards"] / rec["team_air_yards"].replace(0, np.nan)
    rec["adot"] = rec["air_yards"] / rec["targets"]
    rec["yards_per_target"] = rec["rec_yards"] / rec["targets"]
    rec["catch_rate"] = rec["receptions"] / rec["targets"]

    rush = (rush_plays.groupby(["game_id", "posteam", "rusher_player_id"])
            .agg(carries=("play_type", "size"),
                 rush_yards=("rushing_yards", "sum"),
                 rush_td=("rush_touchdown", "sum"),
                 rz_carries=("yardline_100", lambda s: int((s <= 20).sum())),
                 gl_carries=("yardline_100", lambda s: int((s <= 5).sum())))
            .reset_index()
            .rename(columns={"rusher_player_id": "player_id"}))
    rush = rush.merge(team_rush, on=["game_id", "posteam"], how="left")
    rush["rush_share"] = rush["carries"] / rush["team_carries"]
    rush["yards_per_carry"] = rush["rush_yards"] / rush["carries"]

    usage = rec.merge(rush, on=["game_id", "posteam", "player_id"], how="outer")
    if snaps is not None and not snaps.empty:
        cols = [c for c in ("game_id", "player_id", "offense_pct")
                if c in snaps.columns]
        if len(cols) == 3:
            usage = usage.merge(snaps[cols], on=["game_id", "player_id"],
                                how="left")
            usage = usage.rename(columns={"offense_pct": "snap_share"})
    return usage


def recency_weighted(
    frame: pd.DataFrame,
    value_col: str,
    group_cols: Iterable[str] = ("player_id",),
    half_life_games: float = 4.0,
    min_games: int = 3,
    prior_value: float | None = None,
    prior_weight: float = 2.5,
) -> pd.DataFrame:
    """Exponentially weighted average with a shrinkage prior.

    Two failure modes this exists to avoid. Straight season averages ignore the
    role change that is usually the whole reason an edge exists. Last-three-games
    averages overreact to a 2-catch game in a blowout. The half-life makes recent
    games count more without letting them dominate, and the prior pulls a
    three-game sample back toward the position baseline so a rookie's one big
    game does not become his projection.

    ``prior_value`` should be the positional or career baseline. With
    ``prior_weight=2.5``, a player needs roughly five games before his own data
    outweighs the prior.
    """
    group_cols = list(group_cols)
    decay = np.log(2.0) / half_life_games
    out = []
    for keys, g in frame.groupby(group_cols, sort=False):
        g = g.sort_values("game_order" if "game_order" in g else "week")
        values = g[value_col].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        values, n = values[mask], int(mask.sum())
        if n == 0:
            continue
        age = np.arange(n - 1, -1, -1, dtype=float)
        w = np.exp(-decay * age)
        num, den = float(values @ w), float(w.sum())
        if prior_value is not None:
            num += prior_value * prior_weight
            den += prior_weight
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row[f"{value_col}_weighted"] = num / den
        row[f"{value_col}_n_games"] = n
        row[f"{value_col}_sufficient"] = n >= min_games
        out.append(row)
    return pd.DataFrame(out)


def detect_role_change(
    usage: pd.DataFrame,
    metric: str = "target_share",
    recent_games: int = 3,
    baseline_games: int = 8,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Flag players whose role has moved, with a significance check.

    A role change is the single most valuable signal the model has, because it
    is the thing the market prices slowest. It is also the easiest thing to
    hallucinate from noise: three games is a small sample and target share on
    30 attempts has a standard error near 8 points on its own.

    So a change is only flagged when it clears both an absolute threshold and
    roughly two standard errors of the recent sample. The flag feeds the role
    certainty component of the confidence score in both directions — a detected
    change raises the projection but lowers confidence until it persists.
    """
    rows = []
    for player_id, g in usage.groupby("player_id", sort=False):
        g = g.sort_values("week")
        series = g[metric].dropna().to_numpy(dtype=float)
        if series.size < recent_games + 2:
            continue
        recent = series[-recent_games:]
        baseline = series[-(baseline_games + recent_games):-recent_games]
        if baseline.size < 2:
            continue
        delta = float(recent.mean() - baseline.mean())
        se = float(np.sqrt(recent.var(ddof=1) / recent.size
                           + baseline.var(ddof=1) / baseline.size)) or 1e-9
        rows.append({
            "player_id": player_id,
            "metric": metric,
            "baseline": float(baseline.mean()),
            "recent": float(recent.mean()),
            "delta": delta,
            "z_score": delta / se,
            "changed": bool(abs(delta) >= threshold and abs(delta / se) >= 2.0),
            "direction": "up" if delta > 0 else "down",
            "games_since_change": recent_games,
        })
    return pd.DataFrame(rows)
