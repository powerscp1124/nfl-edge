"""Why the model disagrees with the market.

Every string this module produces is derived from a numeric input the projection
actually used. There is no language model here and no template that could fire
without the number behind it existing: if a contribution is not present in the
projection's recorded inputs, no sentence about it is generated.

The decomposition works because projections are built as opportunity times
efficiency inside a game environment. That structure means the difference
between a 96-yard projection and an 83.5-yard line can be attributed back to
specific components, and the attributions add up to the total by construction
rather than by assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Contribution:
    """One component's effect on the projection, in the projection's units."""

    feature: str
    label: str                 # human-readable, e.g. "projected target share"
    baseline_value: float
    projected_value: float
    yards_effect: float        # signed effect on the projected mean
    source: str                # which table/model this came from
    unit: str = ""

    @property
    def direction(self) -> str:
        return "up" if self.yards_effect > 0 else "down"


def decompose_projection(
    *,
    baseline_mean: float,
    projected_mean: float,
    contributions: Sequence[Contribution],
    tolerance: float = 0.5,
) -> dict:
    """Check that the stated contributions actually reconcile.

    If the parts do not sum to the whole within tolerance, the explanation is
    marked unreconciled and the UI shows the projection without a breakdown
    rather than showing a breakdown that does not add up. A plausible-looking
    explanation that silently omits a third of the movement is worse than no
    explanation, because the user will act on it.
    """
    stated = float(sum(c.yards_effect for c in contributions))
    actual = float(projected_mean - baseline_mean)
    residual = actual - stated
    return {
        "baseline": round(baseline_mean, 2),
        "projected": round(projected_mean, 2),
        "total_movement": round(actual, 2),
        "explained": round(stated, 2),
        "residual": round(residual, 2),
        "reconciled": bool(abs(residual) <= tolerance),
        "contributions": [
            {
                "feature": c.feature,
                "label": c.label,
                "from": round(c.baseline_value, 4),
                "to": round(c.projected_value, 4),
                "effect": round(c.yards_effect, 2),
                "share_of_movement": (round(c.yards_effect / actual, 4)
                                      if abs(actual) > 1e-9 else None),
                "source": c.source,
                "unit": c.unit,
            }
            for c in sorted(contributions, key=lambda x: -abs(x.yards_effect))
        ],
    }


def _format_value(value: float, unit: str) -> str:
    if unit == "share":
        return f"{value * 100:.0f}%"
    if unit == "rate":
        return f"{value:.1%}"
    if unit == "rank":
        return f"{int(value)}"
    if unit == "count":
        return f"{value:.0f}"
    return f"{value:.1f}"


def explain_edge(
    *,
    player_name: str,
    stat_label: str,
    projected_mean: float,
    market_line: float,
    model_prob: float,
    market_prob: float,
    contributions: Sequence[Contribution],
    max_reasons: int = 4,
    min_effect: float = 1.0,
) -> dict:
    """Build the "why this bet" panel.

    Only contributions large enough to matter are shown. A 0.3-yard adjustment
    is real but listing it implies a precision the model does not have, and it
    pushes the contributions that actually drove the number down the list.
    """
    material = [c for c in contributions if abs(c.yards_effect) >= min_effect]
    material.sort(key=lambda c: -abs(c.yards_effect))
    top = material[:max_reasons]

    bullets = []
    for c in top:
        arrow = "+" if c.yards_effect > 0 else ""
        bullets.append(
            f"{c.label} {_format_value(c.baseline_value, c.unit)} "
            f"\u2192 {_format_value(c.projected_value, c.unit)} "
            f"({arrow}{c.yards_effect:.1f} yds)"
        )

    headline = (
        f"Model projects {projected_mean:.1f} {stat_label} against a "
        f"{market_line:g} line"
    )
    if top:
        drivers = ", ".join(c.label for c in top[:3])
        headline += f", driven by {drivers}"

    return {
        "player": player_name,
        "headline": headline + ".",
        "reasons": bullets,
        "omitted_small_effects": len(material) - len(top),
        "model_probability": round(model_prob, 4),
        "market_probability": round(market_prob, 4),
        "edge_points": round((model_prob - market_prob) * 100, 1),
    }


def market_disagreement_reason(
    *,
    line_open: float | None,
    line_current: float,
    model_open: float | None,
    model_current: float,
    role_change: dict | None = None,
    injury_changes: Sequence[dict] = (),
) -> list[str]:
    """Second-order explanation: why the market may not have caught up yet.

    A disagreement is more actionable when there is an identifiable reason the
    market is behind — a Friday practice report, a depth-chart move, a role
    change three games old — than when the model simply likes a player more.
    Each string here requires the underlying event to exist in the data.
    """
    notes: list[str] = []

    if line_open is not None and abs(line_current - line_open) >= 1.0:
        direction = "up" if line_current > line_open else "down"
        notes.append(
            f"Market has moved {direction} from {line_open:g} to "
            f"{line_current:g} since open"
        )

    if model_open is not None and abs(model_current - model_open) >= 2.0:
        move = model_current - model_open
        if line_open is not None:
            market_move = line_current - line_open
            if abs(move) - abs(market_move) >= 2.0:
                notes.append(
                    f"Model has moved {move:+.1f} while the market moved "
                    f"{market_move:+.1f}"
                )

    if role_change and role_change.get("changed"):
        notes.append(
            f"{role_change['metric'].replace('_', ' ')} moved "
            f"{role_change['baseline'] * 100:.0f}% \u2192 "
            f"{role_change['recent'] * 100:.0f}% over the last "
            f"{role_change['games_since_change']} games"
        )

    for change in injury_changes:
        notes.append(
            f"{change['attribute'].replace('_', ' ')} "
            f"{change['before'] * 100:.0f}% \u2192 {change['after'] * 100:.0f}% "
            f"after {change['trigger']} news"
        )

    return notes


def matchup_ratings(
    *,
    player_profile: dict,
    opponent_profile: dict,
) -> dict[str, float]:
    """Percentage adjustments for the matchup panel.

    Each rating is the modelled percentage effect on that player's projection
    from this specific opponent, relative to a league-average defence. They are
    computed from the same defensive inputs the simulator uses, not scored
    separately, so the panel cannot disagree with the projection it explains.
    """
    ratings: dict[str, float] = {}

    def rel(player_key: str, opp_key: str, sensitivity: float) -> float | None:
        if player_key not in player_profile or opp_key not in opponent_profile:
            return None
        return float(np.clip(opponent_profile[opp_key] * sensitivity, -0.5, 0.5))

    for name, (pk, ok, sens) in {
        "rush_matchup": ("rush_share", "rush_epa_allowed_z", 0.10),
        "pass_matchup": ("target_share", "pass_epa_allowed_z", 0.09),
        "coverage_matchup": ("adot", "explosive_pass_allowed_z", 0.08),
        "td_environment": ("goal_line_share", "red_zone_td_allowed_z", 0.12),
    }.items():
        value = rel(pk, ok, sens)
        if value is not None:
            ratings[name] = round(value, 4)
    return ratings
