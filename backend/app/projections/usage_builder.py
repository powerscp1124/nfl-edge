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

from dataclasses import dataclass
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
    snap_share: float = 0.0
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


def build_player_usage(
    logs: Sequence[PlayerGameLog],
    *,
    depth_rank: int = 1,
    min_games: int = 3,
    half_life: float = 4.0,
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
    position = latest.position
    prior = POSITION_PRIORS.get(position, POSITION_PRIORS["WR"])
    opp_prior = opportunity_prior(position, depth_rank)
    notes: list[str] = []

    target_share, n_tgt = _shrunk_mean(
        _safe_ratio([g.targets for g in logs],
                    [g.team_pass_attempts for g in logs]),
        prior=opp_prior["target_share"],
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)
    rush_share, _ = _shrunk_mean(
        _safe_ratio([g.carries for g in logs],
                    [g.team_rush_attempts for g in logs]),
        prior=opp_prior["rush_share"],
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)
    snap_share, _ = _shrunk_mean(
        [g.snap_share for g in logs], prior=0.55,
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)

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
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)
    ez_share, _ = _shrunk_mean(
        _safe_ratio([g.ez_targets for g in logs],
                    [g.team_ez_targets for g in logs]),
        prior=opp_prior["target_share"],
        prior_weight=OPPORTUNITY_PRIOR_WEIGHT, half_life=half_life)

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
        is_starting_qb=False,
    )
    return UsageBuildResult(
        usage=usage,
        n_games=n_games,
        sufficient=n_games >= min_games,
        notes=notes,
    )


def normalize_team_shares(usages: Sequence[PlayerUsage]) -> list[PlayerUsage]:
    """Force each share pool to sum to 1.0 across the team.

    Individually estimated shares will not sum to one — shrinkage alone
    guarantees they undershoot. Renormalising here means the simulator's
    completeness check is meaningful: it will still catch a roster that is
    genuinely missing players, because the shares of who *is* present get
    inflated in a way the per-player estimates would not explain.
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
