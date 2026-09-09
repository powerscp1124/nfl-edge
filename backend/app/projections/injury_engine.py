"""Injury and depth-chart propagation.

When a receiver is ruled out his targets do not vanish, and they do not
redistribute evenly either. They go disproportionately to players who occupy a
similar role: the slot receiver's targets flow to the other slot player and the
tight end far more than to the outside X. Splitting a vacated share evenly
across the depth chart is the single most common way a prop model gets an
injury week wrong, because it systematically underrates the direct backup and
overrates everyone else.

The redistribution weights below are priors that should be refit from
historical injury weeks (``fit_redistribution_weights``). They are deliberately
conservative: when the model is uncertain how a room will reallocate, it is
better to spread the share and let the confidence score fall than to commit
hard to a backup who may not see the role.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

# Probability a player at each report status actually plays. Calibrated from
# historical final-status-to-active rates; questionable is close to a coin flip
# early in the week and resolves as inactives are published 90 minutes out.
STATUS_ACTIVE_PROBABILITY = {
    "out": 0.00,
    "doubtful": 0.08,
    "questionable": 0.62,
    "probable": 0.95,
    "active": 1.00,
    None: 1.00,
}

# Once inactives are published there is no uncertainty left, so the
# questionable prior is replaced by the observed truth.
FINAL_STATUS_ACTIVE_PROBABILITY = {"out": 0.0, "active": 1.0, "inactive": 0.0}


@dataclass
class RoleProfile:
    """How a player is used, for similarity-weighted redistribution."""

    player_id: str
    position: str
    depth_rank: int = 1
    slot_rate: float = 0.0
    adot: float = 9.0
    routes_share: float = 0.0


def _similarity(a: RoleProfile, b: RoleProfile) -> float:
    """How interchangeable two pass catchers are, on 0-1.

    Same position counts for a lot, alignment (slot versus outside) counts for
    more than raw depth rank, and target depth matters because a possession
    slot receiver does not absorb a vertical X's air yards.
    """
    score = 0.35 if a.position == b.position else 0.12
    score += 0.30 * (1.0 - abs(a.slot_rate - b.slot_rate))
    score += 0.20 * float(np.exp(-abs(a.adot - b.adot) / 6.0))
    score += 0.15 * float(np.exp(-abs(a.depth_rank - b.depth_rank) / 2.0))
    return float(np.clip(score, 0.05, 1.0))


def redistribute_share(
    vacated_share: float,
    absent: RoleProfile,
    remaining: Sequence[RoleProfile],
    direct_backup_bonus: float = 0.35,
) -> dict[str, float]:
    """Split a vacated opportunity share among the players who remain.

    Returns ``{player_id: additional_share}``, summing to ``vacated_share``.
    The direct backup — the next man at the same position on the depth chart —
    gets a bonus on top of his role similarity, because depth charts encode
    coaching intent that raw usage similarity does not.
    """
    if not remaining or vacated_share <= 0:
        return {}

    weights = np.array([_similarity(absent, r) for r in remaining], dtype=float)

    backup_idx = None
    candidates = [
        i for i, r in enumerate(remaining)
        if r.position == absent.position and r.depth_rank > absent.depth_rank
    ]
    if candidates:
        backup_idx = min(candidates, key=lambda i: remaining[i].depth_rank)
        weights[backup_idx] *= (1.0 + direct_backup_bonus)

    total = weights.sum()
    if total <= 0:
        even = vacated_share / len(remaining)
        return {r.player_id: even for r in remaining}

    weights = weights / total
    return {r.player_id: float(vacated_share * w)
            for r, w in zip(remaining, weights)}


def active_probability(
    report_status: str | None,
    practice_status: str | None = None,
    is_final_report: bool = False,
) -> float:
    """P(player takes the field), from the injury report.

    Practice participation moves the questionable tag meaningfully: a
    questionable player who did not practise Friday is much closer to out than
    the raw 62% base rate suggests.
    """
    key = (report_status or "").strip().lower() or None
    base = STATUS_ACTIVE_PROBABILITY.get(key, 1.0)
    if is_final_report and key in ("out", "active", "inactive"):
        return FINAL_STATUS_ACTIVE_PROBABILITY.get(key, base)
    if key == "questionable" and practice_status:
        practice = practice_status.strip().lower()
        if practice in ("dnp", "did not participate"):
            base = 0.28
        elif practice in ("limited", "lp"):
            base = 0.60
        elif practice in ("full", "fp"):
            base = 0.85
    return float(np.clip(base, 0.0, 1.0))


def injury_uncertainty(active_probs: Iterable[float]) -> float:
    """How much of this projection is riding on unresolved injury news.

    Peaks when a key player is a true coin flip and falls to zero once the
    room is settled. Feeds both the confidence score and the bet filter, which
    is why a Sunday-morning edge on a team with three questionable receivers
    should not clear the strong-bet threshold.
    """
    probs = np.array(list(active_probs), dtype=float)
    if probs.size == 0:
        return 0.0
    entropy_like = 4.0 * probs * (1.0 - probs)  # 1.0 at p=0.5, 0 at 0 or 1
    return float(np.clip(entropy_like.max(), 0.0, 1.0))


@dataclass
class PropagationResult:
    usages: list
    changes: list[dict]
    injury_uncertainty: float


def propagate_injuries(
    usages: Sequence,
    roles: dict[str, RoleProfile],
    statuses: dict[str, dict],
) -> PropagationResult:
    """Apply injury statuses to a team's ``PlayerUsage`` list.

    ``statuses`` maps player_id to ``{"report_status", "practice_status",
    "is_final_report"}``. Players who are out have their shares zeroed and
    redistributed; players who are questionable keep their share but carry an
    ``active_probability`` below 1, which the simulator turns into a genuine
    two-humped distribution rather than a scaled-down mean.

    That distinction matters for pricing. A questionable WR1 is not "80% of a
    WR1" — he is 62% of a full WR1 game and 38% of nothing, and those two
    descriptions imply very different probabilities of clearing a 70-yard line.
    """
    from ..sim.game_sim import PlayerUsage  # local import avoids a cycle

    adjusted: list[PlayerUsage] = []
    changes: list[dict] = []
    probs: list[float] = []

    out_players, active_players = [], []
    for u in usages:
        info = statuses.get(u.player_id, {})
        p_active = active_probability(
            info.get("report_status"),
            info.get("practice_status"),
            info.get("is_final_report", False),
        )
        probs.append(p_active)
        if p_active <= 0.02:
            out_players.append(u)
        else:
            active_players.append(replace(u, active_probability=p_active))

    if not active_players:
        return PropagationResult(list(usages), [], 0.0)

    bumps: dict[str, dict[str, float]] = {}
    for absent in out_players:
        absent_role = roles.get(
            absent.player_id,
            RoleProfile(absent.player_id, absent.position),
        )
        remaining_roles = [
            roles.get(a.player_id, RoleProfile(a.player_id, a.position))
            for a in active_players
        ]
        for attr in ("target_share", "rush_share",
                     "goal_line_share", "end_zone_target_share"):
            vacated = getattr(absent, attr, 0.0)
            if vacated <= 0:
                continue
            split = redistribute_share(vacated, absent_role, remaining_roles)
            for pid, extra in split.items():
                bumps.setdefault(pid, {}).setdefault(attr, 0.0)
                bumps[pid][attr] += extra

    for u in active_players:
        delta = bumps.get(u.player_id, {})
        if delta:
            new = replace(u, **{
                attr: getattr(u, attr) + amount for attr, amount in delta.items()
            })
            for attr, amount in delta.items():
                changes.append({
                    "player_id": u.player_id,
                    "attribute": attr,
                    "before": round(getattr(u, attr), 4),
                    "after": round(getattr(new, attr), 4),
                    "delta": round(amount, 4),
                    "trigger": "injury",
                })
            adjusted.append(new)
        else:
            adjusted.append(u)

    # Players who are out stay in the roster at zero share so the simulator's
    # completeness check still passes and the UI can show them as ruled out.
    for absent in out_players:
        adjusted.append(replace(
            absent, active_probability=0.0, target_share=0.0, rush_share=0.0,
            goal_line_share=0.0, end_zone_target_share=0.0,
        ))

    return PropagationResult(
        usages=adjusted,
        changes=changes,
        injury_uncertainty=injury_uncertainty(probs),
    )
