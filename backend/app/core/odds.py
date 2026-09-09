"""Odds conversion, vig removal, and expected-value math.

Every function here is pure and deterministic. This module is the single source
of truth for price math in the application: nothing else may convert odds or
compute EV inline. See tests/test_odds.py for the worked examples that pin the
behaviour down.

Conventions
-----------
- ``american`` odds are ints or floats, negative for favourites (-110), positive
  for underdogs (+145). ``-100`` and ``+100`` are both treated as even money.
- ``decimal`` odds are total return per 1 unit staked, so even money is 2.0.
- probabilities are floats in (0, 1).
- "raw" probability includes the bookmaker's margin. "fair"/"no-vig" probability
  has the margin removed and sums to 1 across the market's outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import brentq

DevigMethod = Literal["multiplicative", "additive", "power", "shin"]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (total return per unit staked)."""
    a = float(american)
    if a == 0:
        raise ValueError("American odds of 0 are undefined")
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError(f"Decimal odds must exceed 1.0, got {d}")
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def american_to_prob(american: float) -> float:
    """Raw (vig-inclusive) implied probability of American odds.

    -110 -> 0.5238..., +145 -> 0.4082...
    """
    a = float(american)
    if a == 0:
        raise ValueError("American odds of 0 are undefined")
    if a > 0:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)


def prob_to_american(prob: float) -> float:
    """Convert a probability to fair American odds (zero margin)."""
    p = float(prob)
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}")
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def prob_to_decimal(prob: float) -> float:
    """Convert a probability to fair decimal odds."""
    p = float(prob)
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}")
    return 1.0 / p


def decimal_to_prob(decimal_odds: float) -> float:
    """Raw implied probability of decimal odds."""
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError(f"Decimal odds must exceed 1.0, got {d}")
    return 1.0 / d


def format_american(american: float) -> str:
    """Render American odds the way a sportsbook would: -170, +145, EVEN."""
    a = round(float(american))
    if a == 100 or a == -100:
        return "EVEN"
    return f"+{a}" if a > 0 else str(a)


# --------------------------------------------------------------------------- #
# Market margin
# --------------------------------------------------------------------------- #
def overround(american_odds: Sequence[float]) -> float:
    """Sum of raw implied probabilities across a market's outcomes.

    A two-way -110/-110 market returns ~1.0476, i.e. a 4.76% overround.
    """
    return float(sum(american_to_prob(o) for o in american_odds))


def hold_percentage(american_odds: Sequence[float]) -> float:
    """Bookmaker hold: the fraction of handle retained at balanced action."""
    ov = overround(american_odds)
    return (ov - 1.0) / ov


# --------------------------------------------------------------------------- #
# Vig removal
# --------------------------------------------------------------------------- #
def _devig_multiplicative(raw: np.ndarray) -> np.ndarray:
    """Proportional normalisation. Fast, and the industry default."""
    return raw / raw.sum()


def _devig_additive(raw: np.ndarray) -> np.ndarray:
    """Subtract the margin equally from each outcome.

    Removes less vig from longshots than the multiplicative method, which
    partially counteracts favourite-longshot bias. Can go non-positive on very
    lopsided markets, so we fall back to multiplicative if that happens.
    """
    adjusted = raw - (raw.sum() - 1.0) / len(raw)
    if np.any(adjusted <= _EPS):
        return _devig_multiplicative(raw)
    return adjusted


def _devig_power(raw: np.ndarray) -> np.ndarray:
    """Find k such that sum(p_i ** k) == 1.

    Applies a heavier discount to longshots than to favourites, which matches
    observed sportsbook pricing better than proportional de-vigging on markets
    with a wide price spread (anytime TD, 2+ TD).
    """
    if raw.sum() <= 1.0 + _EPS:
        return _devig_multiplicative(raw)

    def objective(k: float) -> float:
        return float(np.sum(np.power(raw, k)) - 1.0)

    try:
        k = brentq(objective, 1.0, 20.0, xtol=1e-12, maxiter=200)
    except ValueError:
        return _devig_multiplicative(raw)
    out = np.power(raw, k)
    return out / out.sum()


def _devig_shin(raw: np.ndarray) -> np.ndarray:
    """Shin (1993): back out the implied share of insider money, z.

    Treats the margin as compensation for informed traders, which produces
    fair probabilities between the multiplicative and power results.
    """
    total = raw.sum()
    if total <= 1.0 + _EPS:
        return _devig_multiplicative(raw)

    def implied(z: float) -> np.ndarray:
        disc = np.sqrt(z * z + 4.0 * (1.0 - z) * (raw * raw) / total)
        return (disc - z) / (2.0 * (1.0 - z))

    def objective(z: float) -> float:
        return float(implied(z).sum() - 1.0)

    try:
        z = brentq(objective, 1e-9, 0.4, xtol=1e-12, maxiter=200)
    except ValueError:
        return _devig_multiplicative(raw)
    out = implied(z)
    return out / out.sum()


_DEVIG_FUNCS = {
    "multiplicative": _devig_multiplicative,
    "additive": _devig_additive,
    "power": _devig_power,
    "shin": _devig_shin,
}


def devig(
    american_odds: Sequence[float],
    method: DevigMethod = "multiplicative",
) -> list[float]:
    """Remove the bookmaker margin from a complete market.

    Pass every outcome of one market at one book: both sides of an over/under,
    or every player in an anytime-TD market if you have the full board.

    >>> [round(p, 4) for p in devig([-110, -110])]
    [0.5, 0.5]
    """
    if len(american_odds) < 2:
        raise ValueError("De-vigging requires at least two outcomes")
    if method not in _DEVIG_FUNCS:
        raise ValueError(f"Unknown devig method: {method}")
    raw = np.array([american_to_prob(o) for o in american_odds], dtype=float)
    fair = _DEVIG_FUNCS[method](raw)
    return [float(p) for p in fair]


def devig_two_way(
    over_odds: float,
    under_odds: float,
    method: DevigMethod = "multiplicative",
) -> tuple[float, float]:
    """De-vig an over/under pair. Returns ``(p_over, p_under)``."""
    p_over, p_under = devig([over_odds, under_odds], method=method)
    return p_over, p_under


def no_vig_probability(
    side_odds: float,
    opposite_odds: float,
    method: DevigMethod = "multiplicative",
) -> float:
    """Fair probability of the side priced at ``side_odds``."""
    return devig_two_way(side_odds, opposite_odds, method=method)[0]


# --------------------------------------------------------------------------- #
# Expected value
# --------------------------------------------------------------------------- #
def expected_value(model_prob: float, american: float, stake: float = 1.0) -> float:
    """Expected profit on ``stake`` at ``american`` given ``model_prob``.

    EV = p * profit_if_win - (1 - p) * stake
    """
    p = float(model_prob)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    profit = (american_to_decimal(american) - 1.0) * stake
    return p * profit - (1.0 - p) * stake


def expected_roi(model_prob: float, american: float) -> float:
    """Expected return per unit staked. +0.229 means +22.9% EV."""
    return expected_value(model_prob, american, stake=1.0)


def breakeven_probability(american: float) -> float:
    """Win probability at which a price is exactly break-even (= raw implied)."""
    return american_to_prob(american)


def probability_edge(model_prob: float, market_prob: float) -> float:
    """Model probability minus no-vig market probability, in probability points.

    0.638 vs 0.512 -> 0.126, displayed as +12.6%.
    """
    return float(model_prob) - float(market_prob)


def kelly_fraction(
    model_prob: float,
    american: float,
    fraction: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Fractional-Kelly stake as a share of bankroll.

    Defaults to quarter Kelly, hard-capped at 2% of bankroll. Full Kelly is
    never the default: it assumes the model's probability is exactly right, and
    a player-prop model's probabilities carry real estimation error.

    Returns 0.0 when the bet has no edge.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Kelly fraction must be in (0, 1]")
    p = float(model_prob)
    b = american_to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    full = (p * b - (1.0 - p)) / b
    if full <= 0:
        return 0.0
    return float(min(full * fraction, cap))


# --------------------------------------------------------------------------- #
# Line shopping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BookQuote:
    """One book's price on one side of one prop."""

    bookmaker: str
    line: float
    american: float
    last_update: str | None = None

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.american)

    @property
    def raw_prob(self) -> float:
        return american_to_prob(self.american)


def best_quote(
    quotes: Iterable[BookQuote],
    side: Literal["over", "under"],
    model_cdf,
) -> BookQuote | None:
    """Pick the quote with the highest expected value, not the best price.

    A better number at a worse price frequently beats a worse number at a better
    price, so line shopping has to be scored on EV against the model's own
    distribution rather than on the American odds alone. ``model_cdf(x)`` must
    return P(stat <= x).
    """
    quotes = list(quotes)
    if not quotes:
        return None
    best, best_ev = None, -np.inf
    for q in quotes:
        p_over = 1.0 - model_cdf(q.line)
        p = p_over if side == "over" else 1.0 - p_over
        ev = expected_value(p, q.american)
        if ev > best_ev:
            best, best_ev = q, ev
    return best


def consensus_line(quotes: Iterable[BookQuote]) -> dict[str, float]:
    """Median, mean and modal line across books, plus book count."""
    lines = np.array([q.line for q in quotes], dtype=float)
    if lines.size == 0:
        return {"median": float("nan"), "mean": float("nan"),
                "mode": float("nan"), "n_books": 0}
    values, counts = np.unique(lines, return_counts=True)
    return {
        "median": float(np.median(lines)),
        "mean": float(np.mean(lines)),
        "mode": float(values[int(np.argmax(counts))]),
        "n_books": int(lines.size),
    }
