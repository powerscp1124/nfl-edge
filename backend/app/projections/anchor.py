"""Anchor a projection on the market line, and model deviations from it.

Measured on the 2025 season, this model's projection explained none of the
outcome that the book's line did not already explain: regressing actual yards
on both, the coefficient on the projection was -0.012 for receiving and -0.003
for rushing, and R^2 with both inputs equalled R^2 with the line alone. Fitting
the blend ``w * line + (1 - w) * projection`` out of sample put ``w`` between
0.92 and 1.00 for every market.

That is not a reason to stop projecting. It is a reason to change what the
projection is *for*. A line already contains the market's view of usage,
health, matchup and game script; rebuilding that from scratch and hoping to
beat it is how a model spends its variance budget reproducing what it could
have started from. Anchoring keeps the line as the centre and asks a narrower,
answerable question: is there a specific reason to sit away from it?

So an edge here has to be *argued* rather than assumed. A deviation must name
its reason -- a role change the market has not repriced, a team-mate ruled out
after the line was set -- and each reason has to earn its magnitude against
outcomes before it moves anything. With no deviation applied the model agrees
with the market and declines to bet, which is the correct behaviour for a
model with nothing to add.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.distributions import EmpiricalDistribution

# Fitted on 2025 by minimising held-out squared error of
# ``w * line + (1 - w) * projection`` against the actual outcome. Per-market
# fits landed at 0.97 / 0.92 / 1.00 (passing / receiving / rushing) and the
# error curve is flat near the optimum, so one shared value is used rather
# than three that differ only by noise.
DEFAULT_ANCHOR_WEIGHT = 0.95


@dataclass(frozen=True)
class Deviation:
    """A named, signed reason to sit away from the line.

    ``magnitude`` is a multiplicative nudge: 0.06 means "six per cent above
    the line". The reason travels with the number so a projection can always
    say why it disagrees, and so an unexplained deviation is impossible to
    express.
    """

    reason: str
    magnitude: float

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("A deviation must name its reason")


@dataclass(frozen=True)
class LineAnchor:
    """Blends the market line with the model, then applies deviations."""

    weight: float = DEFAULT_ANCHOR_WEIGHT
    # Deviations are capped: a signal that wants to move a projection by more
    # than this is making a claim the evidence behind it has never supported,
    # and is far more likely to be a mapping error than an insight.
    max_total_deviation: float = 0.25

    def centre(self, line: float, projection: float,
               deviations: list[Deviation] | None = None) -> float:
        """Where the projection should sit."""
        if not np.isfinite(line):
            return float(projection)
        base = self.weight * float(line) + (1.0 - self.weight) * float(projection)
        total = sum(d.magnitude for d in (deviations or []))
        total = float(np.clip(total, -self.max_total_deviation,
                              self.max_total_deviation))
        return float(max(base * (1.0 + total), 0.0))

    def apply(self, distribution: EmpiricalDistribution, line: float,
              deviations: list[Deviation] | None = None
              ) -> EmpiricalDistribution:
        """Recentre a simulated distribution on the anchored value.

        Anchors the **median**, not the mean. A book sets a line where roughly
        half the outcomes fall either side of it, so the line is an estimate
        of the median. Yardage distributions are right-skewed -- a gamma sum
        with a long upside tail -- so their mean sits well above their median:
        forcing the mean onto the line drags the median far below it and the
        model reads unders as 60% shots on skew alone. Measured on week 5 of
        2025 that mistake cost 15 points of ROI.

        Rescales rather than shifts, so yards cannot go negative and the
        simulated shape survives instead of sliding somewhere it cannot go.
        """
        median = distribution.quantile(0.5)
        if median <= 0 or not np.isfinite(line):
            return distribution
        target = self.centre(line, median, deviations)
        if target <= 0:
            return distribution
        return EmpiricalDistribution(distribution.samples * (target / median),
                                     stat=distribution.stat)


def fit_anchor_weight(lines, projections, actuals,
                      grid: int = 101) -> float:
    """The blend weight that minimises squared error against outcomes.

    Returns 1.0 when the model adds nothing, which is a finding rather than a
    failure -- and the number to watch when judging whether new features have
    started to earn their place.
    """
    line = np.asarray(lines, dtype=float)
    proj = np.asarray(projections, dtype=float)
    act = np.asarray(actuals, dtype=float)
    keep = np.isfinite(line) & np.isfinite(proj) & np.isfinite(act)
    if keep.sum() < 30:
        return DEFAULT_ANCHOR_WEIGHT
    line, proj, act = line[keep], proj[keep], act[keep]
    weights = np.linspace(0.0, 1.0, grid)
    errors = [float(np.mean((w * line + (1.0 - w) * proj - act) ** 2))
              for w in weights]
    return float(weights[int(np.argmin(errors))])
