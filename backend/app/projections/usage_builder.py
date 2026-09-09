"""Building a player's projected role for one specific game.

This is the step that turns history into "what is this player likely to do this
week", and it is where the brief's central instruction lives: the output is a
role in *this* game, not a season average.

Three mechanisms do that work:

- Recency-weighted usage with a shrinkage prior, so a role change registers
  without a single blowout game becoming the projection.
- Positional priors that a thin sample is pulled toward, so a rookie with two
  games does not get a projection built on two games.
- Explicit share renormalisation at the team level, so the sum of a team's
  target shares is 1.0 by construction rather than by luck.

Efficiency and opportunity are built separately and never blended, because the
explanation layer has to be able to say which one moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..sim.game_sim import PlayerUsage

# Positional priors, used as shrinkage targets for thin samples. These are
# league-average roles by depth position and should be refit each season from
# play-by-play rather than left as constants.
POSITION_PRIORS = {
    "WR": {"catch_rate": 0.63, "yards_per_reception": 12.2,
           "reception_yards_cv": 1.05, "adot": 10.5, "yards_per_carry": 6.0},
    "TE": {"catch_rate": 0.69, "yards_per_reception": 10.8,
           "reception_yards_cv": 0.95, "adot": 7.2, "yards_per_carry": 3.0},
    "RB": {"catch_rate": 0.75, "yards_per_reception": 7.6,
           "reception_yards_cv": 0.90, "adot": 0.8, "yards_per_carry": 4.3},
    "QB": {"catch_rate": 0.0, "yards_per_reception": 0.0,
           "reception_yards_cv": 1.0, "adot": 0.0, "yards_per_carry": 4.6},
}

# Dispersion of a single carry's yardage. High because rushing outcomes are
# dominated by a small number of explosive runs: the median carry is about 3.5
# yards and the mean is 4.3, and that gap is the whole distribution.
CARRY_CV = 1.65

# Games of own data needed before a player's own efficiency outweighs the
# positional prior.
EFFICIENCY_PRIOR_WEIGHT = 4.0
OPPORTUNITY_PRIOR_WEIGHT = 1.5
# A skill player at or above this share of snaps is treated as holding his
# listed role and carries the full depth-chart prior. Below it the prior is
# scaled down, because a receiver on 5% of snaps is not a WR3 who happened to
# be quiet -- he is not in the rotation. Giving him a WR3's prior share is not
# harmless: every share pool is renormalised to 1.0 across the team, so prior
# mass handed to players who do not play is taken directly from those who do.
FULL_PARTICIPATION = 0.50
# How many recent team games decide whether a player is currently playing.
# Short, because the question is "is he in the line-up now", not "how much of
# the season did he play" -- a rookie promoted three weeks ago is a starter,
# and a veteran who has not dressed since October is not.
AVAILABILITY_WINDOW = 4
# Never assert a player is certainly absent. A blanket zero turns his line
# into a guaranteed under, which is exactly the degenerate 100% projection the
# review guard has to catch.
MIN_AVAILABILITY = 0.05

# Opportunity shares are shrunk toward a depth-chart baseline, not toward zero.
# Shrinking toward zero applies the identical multiplicative factor
# den/(den + w) to every player on the team, which the team-level
# renormalisation then cancels exactly -- so the shrinkage does nothing at all
# and a two-game sample is trusted as much as a ten-game one. Shrinking toward
# a real baseline moves thin-sample players toward it and leaves established
# ones alone, which is what shrinkage is for.
OPPORTUNITY_PRIORS = {
    ("WR", 1): {"target_share": 0.24, "rush_share": 0.01},
    ("WR", 2): {"target_share": 0.18, "rush_share": 0.01},
    ("WR", 3): {"target_share": 0.11, "rush_share": 0.00},
    ("TE", 1): {"target_share": 0.17, "rush_share": 0.00},
    ("TE", 2): {"target_share": 0.06, "rush_share": 0.00},
    ("RB", 1): {"target_share": 0.12, "rush_share": 0.62},
    ("RB", 2): {"target_share": 0.06, "rush_share": 0.26},
    ("QB", 1): {"target_share": 0.00, "rush_share": 0.06},
}
DEFAULT_OPPORTUNITY_PRIOR = {"target_share": 0.08, "rush_share": 0.05}


def opportunity_prior(position: str, depth_rank: int) -> dict:
    """Baseline share for a player at this position and depth."""
    return OPPORTUNITY_PRIORS.get(
        (position, min(max(depth_rank, 1), 3)), DEFAULT_OPPORTUNITY_PRIOR
    )


@dataclass
class PlayerGameLog:
    """One player's line from one game, as stored in ``player_game_stats``."""

    player_id: str
    name: str
    position: str
    team: str
    week: int
    targets: float = 0.0
    receptions: float = 0.0
    receiving_yards: float = 0.0
    air_yards: float = 0.0
    team_pass_attempts: float = 0.0
    carries: float = 0.0
    rushing_yards: float = 0.0
    team_rush_attempts: float = 0.0
    # None means the snap count did not resolve, which is not the same
    # fact as a player who was on the field for none of them.
    snap_share: float | None = None
    rz_carries: float = 0.0
    gl_carries: float = 0.0
    ez_targets: float = 0.0
    team_gl_carries: float = 0.0
    team_ez_targets: float = 0.0


def _weights(n: int, half_life: float) -> np.ndarray:
    """Exponential recency weights, most recent game last."""
    decay = np.log(2.0) / half_life
    age = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-decay * age)


def _shrunk_mean(
    values: Sequence[float],
    prior: float,
    prior_weight: float,
    half_life: float = 4.0,
) -> tuple[float, int]:
    """Recency-weighted mean pulled toward a prior. Returns (value, n_games)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=float)
    if arr.size == 0:
        return float(prior), 0
    w = _weights(arr.size, half_life)
    num = float(arr @ w) + prior * prior_weight
    den = float(w.sum()) + prior_weight
    return num / den, int(arr.size)


def _prior_fraction(n_values: int, prior_weight: float,
                    half_life: float) -> float:
    """Share of a shrunk estimate contributed by the prior rather than data."""
    if prior_weight <= 0:
        return 0.0
    if n_values <= 0:
        return 1.0
    evidence = float(_weights(n_values, half_life).sum())
    return prior_weight / (evidence + prior_weight)


def _safe_ratio(num: Sequence[float], den: Sequence[float]) -> list[float]:
    out = []
    for n, d in zip(num, den):
        out.append(float(n) / float(d) if d and d > 0 else np.nan)
    return out


@dataclass
class UsageBuildResult:
    usage: PlayerUsage
    n_games: int
    sufficient: bool
    notes: list[str]
    # How much of each opportunity share came from the depth-chart prior
    # rather than from anything this player was observed doing. Consumed by
    # ``normalize_team_shares`` so that mass invented by priors is what gets
    # removed when the pool overshoots 1.0.
    prior_mass: dict[str, float] = field(default_factory=dict)


def roster_prior_scale(entries: Sequence[tuple[str, int]]) -> float:
    """How much to shrink every depth-chart prior so the roster's sum to 1.0.

    ``OPPORTUNITY_PRIORS`` says what a WR1 or a WR3 typically commands. Those
    numbers are per-player and were never meant to be summed, but a projection
    does sum them: sixteen players each pulled toward a plausible individual
    share produce a team that throws 150% of its passes. Renormalising the
    blended estimates afterwards is too late, because by then the invented
    mass is indistinguishable from the real usage it is mixed with, and the
    correction lands on everyone alike.

    So scale the priors *before* they are blended, by the amount the roster
    overshoots. Returns 1.0 when the priors already fit inside one team.
    """
    total = sum(opportunity_prior(pos, rank)["target_share"]
                for pos, rank in entries)
    return 1.0 if total <= 1.0 else 1.0 / total


def build_player_usage(
    logs: Sequence[PlayerGameLog],
    *,
    depth_rank: int = 1,
    min_games: int = 3,
    half_life: float = 4.0,
    prior_scale: float = 1.0,
    team_weeks: Sequence[int] | None = None,
) -> UsageBuildResult:
    """Turn a player's game logs into a projected role.

    Every rate is computed as a share of the team's opportunity in that same
    game, not as a raw per-game count, so a player's role is not distorted by
    the pace of the games he happened to play in. A receiver on a 78-play
    offence and one on a 55-play offence with the same target share have the
    same role; their raw target counts do not say so.
    """
    if not logs:
        raise ValueError("Cannot build usage from an empty game log")

    logs = sorted(logs, key=lambda x: x.week)
    latest = logs[-1]

    # Availability: in how many of the team's recent games did this player
    # actually appear? Opportunity is renormalised across whoever is on the
    # field, so a player carried at full availability who does not dress takes
    # his whole share out of the game. Measured over the 2025 season that was
    # 20% of a team's targets landing on players who never took the field,
    # which is most of why projections came in low.
    availability = 1.0
    if team_weeks:
        recent = sorted({int(w) for w in team_weeks})[-AVAILABILITY_WINDOW:]
        if recent:
            # Weighted toward the most recent game. Whether a player dressed
            # last week says far more about Sunday than whether he dressed a
            # month ago, and an unweighted rate leaves a returning starter and
            # a fading one looking identical.
            appeared = {g.week for g in logs}
            weights = np.arange(1, len(recent) + 1, dtype=float)
            hit = np.array([1.0 if w in appeared else 0.0 for w in recent])
            availability = max(float((hit * weights).sum() / weights.sum()),
                               MIN_AVAILABILITY)
    position = latest.position
    prior = POSITION_PRIORS.get(position, POSITION_PRIORS["WR"])
    opp_prior = opportunity_prior(position, depth_rank)
    if prior_scale != 1.0:
        opp_prior = {k: v * prior_scale for k, v in opp_prior.items()}
    notes: list[str] = []

    # Participation is computed first, because it decides how much
    # depth-chart prior the opportunity shares below are allowed to carry.
    snap_share, _ = _shrunk_mean(
        [g.snap_share for g in logs], prior=0.55,
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)
    # Observed snaps only, with no prior: the question here is "was this
    # player in the rotation", and shrinking that toward an average player's
    # 0.55 would answer "probably" for someone who was never on the field.
    # Rows whose snap count did not resolve are None and drop out, so an
    # unresolved player keeps the full prior rather than being deleted.
    observed_snaps, n_snap = _shrunk_mean(
        [g.snap_share for g in logs], prior=0.0, prior_weight=0.0,
        half_life=half_life)
    participation = (float(np.clip(observed_snaps / FULL_PARTICIPATION, 0.0, 1.0))
                     if n_snap else 1.0)
    opp_weight = OPPORTUNITY_PRIOR_WEIGHT * participation

    target_share, n_tgt = _shrunk_mean(
        _safe_ratio([g.targets for g in logs],
                    [g.team_pass_attempts for g in logs]),
        prior=opp_prior["target_share"],
        prior_weight=opp_weight, half_life=half_life)
    rush_share, _ = _shrunk_mean(
        _safe_ratio([g.carries for g in logs],
                    [g.team_rush_attempts for g in logs]),
        prior=opp_prior["rush_share"],
        prior_weight=opp_weight, half_life=half_life)

    catch_rate, n_catch = _shrunk_mean(
        _safe_ratio([g.receptions for g in logs], [g.targets for g in logs]),
        prior=prior["catch_rate"], prior_weight=EFFICIENCY_PRIOR_WEIGHT,
        half_life=half_life)
    ypr, _ = _shrunk_mean(
        _safe_ratio([g.receiving_yards for g in logs],
                    [g.receptions for g in logs]),
        prior=prior["yards_per_reception"],
        prior_weight=EFFICIENCY_PRIOR_WEIGHT, half_life=half_life)
    adot, _ = _shrunk_mean(
        _safe_ratio([g.air_yards for g in logs], [g.targets for g in logs]),
        prior=prior["adot"], prior_weight=EFFICIENCY_PRIOR_WEIGHT,
        half_life=half_life)
    ypc, n_carry = _shrunk_mean(
        _safe_ratio([g.rushing_yards for g in logs], [g.carries for g in logs]),
        prior=prior["yards_per_carry"], prior_weight=EFFICIENCY_PRIOR_WEIGHT,
        half_life=half_life)

    gl_share, _ = _shrunk_mean(
        _safe_ratio([g.gl_carries for g in logs],
                    [g.team_gl_carries for g in logs]),
        prior=opp_prior["rush_share"] * 0.8,
        prior_weight=opp_weight, half_life=half_life)
    ez_share, _ = _shrunk_mean(
        _safe_ratio([g.ez_targets for g in logs],
                    [g.team_ez_targets for g in logs]),
        prior=opp_prior["target_share"],
        prior_weight=opp_weight, half_life=half_life)

    n_games = len(logs)
    if n_games < min_games:
        notes.append(
            f"only {n_games} game(s) of data; efficiency is mostly the "
            f"{position} positional prior"
        )
    if n_catch == 0 and position in ("WR", "TE"):
        notes.append("no receiving history; catch rate is the positional prior")

    usage = PlayerUsage(
        player_id=latest.player_id,
        name=latest.name,
        position=position,
        team=latest.team,
        target_share=float(np.clip(target_share, 0.0, 0.55)),
        rush_share=float(np.clip(rush_share, 0.0, 0.95)),
        snap_share=float(np.clip(snap_share, 0.0, 1.0)),
        route_participation=float(np.clip(snap_share, 0.0, 1.0)),
        catch_rate=float(np.clip(catch_rate, 0.05, 0.95)),
        yards_per_reception=float(np.clip(ypr, 2.0, 25.0)),
        reception_yards_cv=float(prior["reception_yards_cv"]),
        adot=float(np.clip(adot, -3.0, 22.0)),
        yards_per_carry=float(np.clip(ypc, 1.5, 8.0)),
        carry_yards_cv=CARRY_CV,
        goal_line_share=float(np.clip(gl_share, 0.0, 1.0)),
        end_zone_target_share=float(np.clip(ez_share, 0.0, 1.0)),
        active_probability=float(np.clip(availability, 0.0, 1.0)),
        is_starting_qb=False,
    )
    prior_fraction = _prior_fraction(n_tgt, opp_weight, half_life)
    rush_fraction = _prior_fraction(n_carry, opp_weight, half_life)
    return UsageBuildResult(
        usage=usage,
        n_games=n_games,
        sufficient=n_games >= min_games,
        notes=notes,
        prior_mass={
            "target_share": usage.target_share * prior_fraction,
            "rush_share": usage.rush_share * rush_fraction,
            "goal_line_share": usage.goal_line_share * rush_fraction,
            "end_zone_target_share": (usage.end_zone_target_share
                                      * prior_fraction),
        },
    )


def normalize_team_shares(
    usages: Sequence[PlayerUsage],
    prior_mass: Sequence[dict[str, float]] | None = None,
) -> list[PlayerUsage]:
    """Force each share pool to sum to 1.0 across the team.

    Individually estimated shares do not sum to one. Which direction they miss
    in depends on the roster: with every player shrunk toward a depth-chart
    prior, a roster carrying a long tail of fringe players *overshoots*,
    because each of them is pulled up toward a prior share he has not earned.

    Scaling everyone down by a common factor to correct that is the wrong
    correction. It taxes a starter with seven games of consistent usage
    exactly as hard as a receiver with one appearance, even though the excess
    was created entirely by the second. On a live slate that cost the WR1
    about nine points of target share.

    So when ``prior_mass`` is supplied — how much of each player's share came
    from the prior rather than from observed usage — the overshoot is taken
    out of that prior mass first, in proportion to it, and only spills over
    into observed usage if the priors cannot absorb it. Without it the old
    proportional behaviour is kept, so callers that have no decomposition
    still work.
    """
    from dataclasses import replace

    out = list(usages)
    for attr, positions in (
        ("target_share", {"WR", "TE", "RB"}),
        ("rush_share", {"RB", "QB", "WR"}),
        ("goal_line_share", {"QB", "RB", "WR", "TE"}),
        ("end_zone_target_share", {"QB", "RB", "WR", "TE"}),
    ):
        idxs = [i for i, u in enumerate(out) if u.position in positions]
        total = sum(getattr(out[i], attr) for i in idxs)
        if total <= 0:
            continue

        excess = total - 1.0
        absorbed = False
        if prior_mass is not None and excess > 1e-12:
            # Never claim to remove more prior mass than the player's share.
            available = {i: min(float(prior_mass[i].get(attr, 0.0)),
                                getattr(out[i], attr)) for i in idxs}
            pool = sum(available.values())
            if pool >= excess:
                for i in idxs:
                    share = getattr(out[i], attr)
                    cut = excess * (available[i] / pool) if pool else 0.0
                    out[i] = replace(out[i], **{attr: max(share - cut, 0.0)})
                absorbed = True
            elif pool > 0:
                # Strip the priors entirely, then scale what observation left.
                for i in idxs:
                    share = getattr(out[i], attr)
                    out[i] = replace(out[i],
                                     **{attr: max(share - available[i], 0.0)})
                total = sum(getattr(out[i], attr) for i in idxs)
                if total <= 0:
                    continue

        if not absorbed:
            for i in idxs:
                out[i] = replace(out[i],
                                 **{attr: getattr(out[i], attr) / total})
    return out


def flag_role_changes(
    logs: Sequence[PlayerGameLog],
    recent: int = 3,
    baseline: int = 6,
    threshold: float = 0.05,
) -> dict | None:
    """Detect a target-share shift with a significance check.

    Returns ``None`` when there is no evidence of a change. A detected change
    raises the projection but should *lower* role certainty until it persists,
    because three games is a small sample and target share on 30 attempts
    carries roughly eight points of standard error on its own.
    """
    if len(logs) < recent + 2:
        return None
    logs = sorted(logs, key=lambda x: x.week)
    shares = np.asarray(
        [t / a if a else np.nan
         for t, a in ((g.targets, g.team_pass_attempts) for g in logs)],
        dtype=float,
    )
    shares = shares[~np.isnan(shares)]
    if shares.size < recent + 2:
        return None

    recent_vals = shares[-recent:]
    base_vals = shares[-(baseline + recent):-recent]
    if base_vals.size < 2:
        return None

    delta = float(recent_vals.mean() - base_vals.mean())
    se = float(np.sqrt(recent_vals.var(ddof=1) / recent_vals.size
                       + base_vals.var(ddof=1) / base_vals.size)) or 1e-9
    z = delta / se
    if abs(delta) < threshold or abs(z) < 2.0:
        return None
    return {
        "metric": "target_share",
        "baseline": float(base_vals.mean()),
        "recent": float(recent_vals.mean()),
        "delta": delta,
        "z_score": z,
        "changed": True,
        "games_since_change": recent,
    }
