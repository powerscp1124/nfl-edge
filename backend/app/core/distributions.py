"""Outcome distributions for player markets.

The application never compares a point estimate to a line. Every market is
answered from a distribution, and the distribution family is chosen per stat
rather than assumed normal everywhere:

- receiving / rushing yards: strongly right-skewed with a real mass at zero, so
  a zero-inflated gamma fits far better than a normal. A normal fit to a WR
  averaging 60 yards puts meaningful probability below zero, which is not a
  thing that happens.
- passing yards: roughly symmetric with a mild right tail once a QB is a
  confirmed starter; a shifted gamma handles it and degrades to near-normal as
  the shape parameter grows.
- touchdowns: counts, and correlated with teammates. Handled by Poisson-binomial
  over per-opportunity conversion probabilities, or read straight off the
  simulator.

The preferred path in production is ``EmpiricalDistribution`` built from the
Monte Carlo game simulator, because it carries the correlation between a QB's
yards and his receivers' yards. The parametric classes exist for players with
too little simulator support, and as a sanity check on the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy import stats

_EPS = 1e-9


class OutcomeDistribution(Protocol):
    """Everything the edge engine needs from a projection."""

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def prob_over(self, line: float) -> float: ...


def _summary(dist) -> dict[str, float]:
    return {
        "mean": dist.mean(),
        "median": dist.quantile(0.50),
        "std": dist.std(),
        "p10": dist.quantile(0.10),
        "p25": dist.quantile(0.25),
        "p50": dist.quantile(0.50),
        "p75": dist.quantile(0.75),
        "p90": dist.quantile(0.90),
    }


# --------------------------------------------------------------------------- #
# Empirical (simulator output)
# --------------------------------------------------------------------------- #
@dataclass
class EmpiricalDistribution:
    """Distribution defined by Monte Carlo samples.

    ``samples`` is one draw per simulated game. Sportsbook prop lines are
    half-points (83.5), so ties are not possible and ``prob_over`` is an
    unambiguous strict comparison. Integer lines are handled by splitting the
    push mass out, which is what a book does when it voids a push.
    """

    samples: np.ndarray
    stat: str = "yards"

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.size == 0:
            raise ValueError("EmpiricalDistribution requires at least one sample")
        self._sorted = np.sort(self.samples)

    @property
    def n(self) -> int:
        return int(self.samples.size)

    def mean(self) -> float:
        return float(np.mean(self.samples))

    def std(self) -> float:
        return float(np.std(self.samples, ddof=1)) if self.n > 1 else 0.0

    def quantile(self, q: float) -> float:
        return float(np.quantile(self._sorted, q))

    def cdf(self, x: float) -> float:
        return float(np.searchsorted(self._sorted, x, side="right") / self.n)

    def prob_over(self, line: float) -> float:
        """P(stat > line), excluding pushes from the sample."""
        over = float(np.count_nonzero(self.samples > line))
        push = float(np.count_nonzero(self.samples == line))
        live = self.n - push
        if live <= 0:
            return 0.5
        return over / live

    def prob_under(self, line: float) -> float:
        return 1.0 - self.prob_over(line)

    def prob_at_least(self, k: float) -> float:
        """P(stat >= k). For count markets such as touchdowns."""
        return float(np.count_nonzero(self.samples >= k) / self.n)

    def monte_carlo_error(self, line: float) -> float:
        """Standard error on ``prob_over`` from finite sample size.

        Used by the confidence score: a 2-point edge inside 1 point of
        simulation noise is not an edge. At 10k sims this is about 0.5pp.
        """
        p = self.prob_over(line)
        return float(np.sqrt(max(p * (1.0 - p), _EPS) / self.n))

    def summary(self) -> dict[str, float]:
        return _summary(self)

    def threshold_curve(self, thresholds: Sequence[float]) -> list[dict[str, float]]:
        """P(stat > t) at each threshold, for the alt-line probability curve."""
        return [{"threshold": float(t), "prob_over": self.prob_over(t)}
                for t in thresholds]


# --------------------------------------------------------------------------- #
# Parametric fallbacks
# --------------------------------------------------------------------------- #
@dataclass
class ZeroInflatedGamma:
    """Gamma body with an explicit point mass at zero.

    Fits receiving and rushing yards. ``p_zero`` is the probability the player
    finishes with no yards (inactive-in-game, zero targets, or a lone stuffed
    carry); ``mean_positive`` and ``cv`` describe the conditional distribution
    given non-zero production.
    """

    p_zero: float
    mean_positive: float
    cv: float  # coefficient of variation of the positive part

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_zero < 1.0:
            raise ValueError("p_zero must be in [0, 1)")
        if self.mean_positive <= 0:
            raise ValueError("mean_positive must be positive")
        if self.cv <= 0:
            raise ValueError("cv must be positive")
        self.shape = 1.0 / (self.cv ** 2)
        self.scale = self.mean_positive / self.shape
        self._g = stats.gamma(a=self.shape, scale=self.scale)

    @classmethod
    def from_moments(cls, mean: float, std: float, p_zero: float = 0.0):
        """Build from an unconditional mean and standard deviation."""
        if mean <= 0:
            raise ValueError("mean must be positive")
        mean_pos = mean / (1.0 - p_zero)
        var_total = std ** 2
        # Var = (1-p)(var_pos + mean_pos^2) - ((1-p) mean_pos)^2
        var_pos = var_total / (1.0 - p_zero) - (p_zero * mean_pos ** 2)
        var_pos = max(var_pos, (0.35 * mean_pos) ** 2)  # floor keeps cv sane
        return cls(p_zero=p_zero, mean_positive=mean_pos,
                   cv=float(np.sqrt(var_pos) / mean_pos))

    def mean(self) -> float:
        return float((1.0 - self.p_zero) * self.mean_positive)

    def std(self) -> float:
        m2 = (1.0 - self.p_zero) * (self._g.var() + self.mean_positive ** 2)
        return float(np.sqrt(max(m2 - self.mean() ** 2, 0.0)))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return float(self.p_zero + (1.0 - self.p_zero) * self._g.cdf(x))

    def prob_over(self, line: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - self.cdf(line))))

    def quantile(self, q: float) -> float:
        if q <= self.p_zero:
            return 0.0
        return float(self._g.ppf((q - self.p_zero) / (1.0 - self.p_zero)))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        draws = rng.gamma(self.shape, self.scale, size=n)
        return np.where(rng.random(n) < self.p_zero, 0.0, draws)

    def summary(self) -> dict[str, float]:
        return _summary(self)


@dataclass
class ShiftedGamma:
    """Gamma shifted left, for passing yards.

    Passing yards for a healthy starter are near-symmetric but bounded below and
    with a longer right tail than left. A gamma with a large shape parameter and
    a location shift captures that and converges to a normal as shape grows.
    """

    mean_value: float
    std_value: float
    skew: float = 0.35

    def __post_init__(self) -> None:
        if self.std_value <= 0:
            raise ValueError("std must be positive")
        self.shape = max(4.0 / (self.skew ** 2), 1.0)
        self.scale = self.std_value / np.sqrt(self.shape)
        self.loc = self.mean_value - self.shape * self.scale
        self._g = stats.gamma(a=self.shape, loc=self.loc, scale=self.scale)

    def mean(self) -> float:
        return float(self.mean_value)

    def std(self) -> float:
        return float(self.std_value)

    def cdf(self, x: float) -> float:
        return float(self._g.cdf(x))

    def prob_over(self, line: float) -> float:
        return float(max(0.0, min(1.0, self._g.sf(line))))

    def quantile(self, q: float) -> float:
        return float(max(self._g.ppf(q), 0.0))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.maximum(self._g.ppf(rng.random(n)), 0.0)

    def summary(self) -> dict[str, float]:
        return _summary(self)


# --------------------------------------------------------------------------- #
# Touchdown counts
# --------------------------------------------------------------------------- #
def poisson_binomial_pmf(probs: Sequence[float]) -> np.ndarray:
    """Exact PMF of the number of successes across independent trials.

    ``probs`` is one conversion probability per scoring opportunity: a carry
    inside the 5, a red-zone target, and so on. Returns an array where index k
    is P(exactly k touchdowns).

    O(n^2) convolution, exact to floating point. This is the right primitive for
    anytime-TD because a player's opportunities differ in quality: three carries
    from the 1 are not the same as three from the 18.
    """
    probs = np.asarray(probs, dtype=float)
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("All probabilities must be in [0, 1]")
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def td_probabilities(pmf: np.ndarray) -> dict[str, float]:
    """Turn a touchdown PMF into the quantities the touchdown page needs."""
    pmf = np.asarray(pmf, dtype=float)
    padded = np.zeros(max(5, pmf.size))
    padded[: pmf.size] = pmf
    return {
        "p0": float(padded[0]),
        "p1": float(padded[1]),
        "p2": float(padded[2]),
        "p3": float(padded[3]),
        "p4_plus": float(padded[4:].sum()),
        "anytime": float(1.0 - padded[0]),
        "two_plus": float(1.0 - padded[0] - padded[1]),
        "expected": float(np.arange(padded.size) @ padded),
    }


def negative_binomial_td(mean: float, dispersion: float = 1.6) -> np.ndarray:
    """Overdispersed count PMF for touchdowns, as a parametric fallback.

    ``dispersion`` is variance / mean. NFL player touchdown counts are
    overdispersed relative to Poisson because opportunity itself varies from
    week to week (blowouts, injuries, game script), so a Poisson understates
    P(2+) — which is exactly the market this matters most for.
    """
    if mean <= 0:
        raise ValueError("mean must be positive")
    if dispersion <= 1.0:
        return stats.poisson(mean).pmf(np.arange(0, 8))
    r = mean / (dispersion - 1.0)
    p = r / (r + mean)
    return stats.nbinom(n=r, p=p).pmf(np.arange(0, 8))


def implied_team_totals(spread: float, total: float) -> tuple[float, float]:
    """Split a game total into implied team totals.

    ``spread`` is the home spread (-3.5 means home favoured by 3.5). Returns
    ``(home_implied, away_implied)``.
    """
    home = total / 2.0 - spread / 2.0
    away = total / 2.0 + spread / 2.0
    return float(home), float(away)
