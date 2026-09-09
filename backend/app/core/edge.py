"""Model-versus-market edge engine.

Turns a model distribution plus a board of sportsbook prices into a ranked,
graded, explainable recommendation. This is the step the whole application
exists to produce:

    "63% chance of going over 81.5 while the market implies 51%"

not:

    "projected for 94 yards".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Sequence

import numpy as np

from .odds import (
    BookQuote,
    DevigMethod,
    american_to_prob,
    consensus_line,
    devig_two_way,
    expected_value,
    format_american,
    kelly_fraction,
    prob_to_american,
    probability_edge,
)

Side = Literal["over", "under"]
Recommendation = Literal[
    "strong", "moderate", "lean", "pass", "needs_review", "insufficient_data"
]


@dataclass
class BetThresholds:
    """Every threshold is configurable; these are the defaults from the spec."""

    strong_edge: float = 0.05
    strong_confidence: float = 70.0
    moderate_edge: float = 0.03
    moderate_confidence: float = 60.0
    lean_edge: float = 0.02
    lean_confidence: float = 50.0
    min_books: int = 3
    min_ev: float = 0.0
    max_injury_uncertainty: float = 0.35
    # An edge this large on a liquid market is far more often a bad player
    # mapping, a stale quote, or an alternate line misread as the main line
    # than it is a real opportunity. It is routed to review instead of being
    # recommended or ranked, which is what keeps a data bug from becoming the
    # top pick on the dashboard.
    review_edge: float = 0.18


@dataclass
class ConfidenceInputs:
    """Independent signals feeding the 0-100 confidence score.

    Each component is a 0-1 quality score. They are deliberately kept separate
    so the UI can show *why* confidence is low, and so correlated signals are
    not counted twice: market agreement and book count, for example, are folded
    into one market component rather than added as two.
    """

    projection_quality: float = 0.5   # sample size, feature completeness
    market_quality: float = 0.5       # book count, price dispersion, liquidity
    injury_certainty: float = 0.5     # inactives known? questionable tags?
    role_certainty: float = 0.5       # snap/route/target share stability
    simulation_precision: float = 1.0 # 1 - normalised Monte Carlo error
    market_agreement: float = 0.5     # do books agree with each other?


@dataclass
class PropEdge:
    """One side of one prop at one price, fully scored."""

    player_id: str
    player_name: str
    team: str
    opponent: str
    position: str
    market: str
    side: Side
    line: float
    bookmaker: str
    american: float
    model_prob: float
    market_prob: float
    edge: float
    ev: float
    fair_american: float
    projection: float
    median: float
    p25: float
    p75: float
    confidence: float
    confidence_parts: dict[str, float]
    grade: str
    recommendation: Recommendation
    kelly: float
    consensus: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    model_version: str = "unversioned"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def display_odds(self) -> str:
        return format_american(self.american)

    @property
    def display_fair(self) -> str:
        return format_american(self.fair_american)


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def confidence_score(inputs: ConfidenceInputs, edge: float) -> tuple[float, dict]:
    """Blend independent signals into a 0-100 score.

    Edge magnitude contributes, but with a deliberately concave transform and a
    modest weight. A very large edge is more often a data error than a genuine
    opportunity, so a 30-point "edge" does not earn a 99 confidence — it earns
    roughly what a 10-point edge earns, and the data-quality components decide
    the rest.
    """
    parts = {
        "projection": float(np.clip(inputs.projection_quality, 0, 1)),
        "market": float(np.clip(inputs.market_quality, 0, 1)),
        "injury": float(np.clip(inputs.injury_certainty, 0, 1)),
        "role": float(np.clip(inputs.role_certainty, 0, 1)),
        "precision": float(np.clip(inputs.simulation_precision, 0, 1)),
        "agreement": float(np.clip(inputs.market_agreement, 0, 1)),
    }
    weights = {
        "projection": 0.26,
        "market": 0.16,
        "injury": 0.18,
        "role": 0.18,
        "precision": 0.12,
        "agreement": 0.10,
    }
    base = sum(parts[k] * w for k, w in weights.items())

    # Concave in |edge|, saturating around 10 points of edge.
    edge_bonus = float(np.tanh(abs(edge) / 0.10)) * 0.12
    score = float(np.clip((base * 0.88 + edge_bonus) * 100.0, 0.0, 100.0))
    return score, {k: round(v * 100, 1) for k, v in parts.items()}


def grade_from(edge: float, confidence: float) -> str:
    """Letter grade combining edge size and confidence."""
    if edge <= 0:
        return "F"
    composite = (min(edge, 0.15) / 0.15) * 0.6 + (confidence / 100.0) * 0.4
    cuts = [
        (0.90, "A+"), (0.82, "A"), (0.74, "A-"),
        (0.66, "B+"), (0.58, "B"), (0.50, "B-"),
        (0.42, "C+"), (0.34, "C"), (0.26, "C-"),
        (0.18, "D"),
    ]
    for cut, grade in cuts:
        if composite >= cut:
            return grade
    return "F"


def classify(
    edge: float,
    ev: float,
    confidence: float,
    n_books: int,
    injury_uncertainty: float,
    thresholds: BetThresholds,
    sufficient_data: bool = True,
) -> Recommendation:
    """Apply the bet filters. Being above the line is not enough on its own."""
    if not sufficient_data:
        return "insufficient_data"
    if n_books < thresholds.min_books:
        return "insufficient_data"
    if abs(edge) >= thresholds.review_edge:
        return "needs_review"
    if ev <= thresholds.min_ev or edge <= 0:
        return "pass"
    if injury_uncertainty > thresholds.max_injury_uncertainty:
        return "pass"
    if edge >= thresholds.strong_edge and confidence >= thresholds.strong_confidence:
        return "strong"
    if edge >= thresholds.moderate_edge and confidence >= thresholds.moderate_confidence:
        return "moderate"
    if edge >= thresholds.lean_edge and confidence >= thresholds.lean_confidence:
        return "lean"
    return "pass"


# --------------------------------------------------------------------------- #
# Market quality
# --------------------------------------------------------------------------- #
def market_agreement_score(quotes: Sequence[BookQuote]) -> float:
    """How tightly books agree, on 0-1.

    Wide disagreement between books means one of them is stale or wrong, which
    is often where the real edge lives — but it also means the consensus number
    is a weaker benchmark, so confidence should fall even as edge rises.
    """
    if len(quotes) < 2:
        return 0.3
    lines = np.array([q.line for q in quotes], dtype=float)
    spread = float(np.max(lines) - np.min(lines))
    scale = max(np.median(np.abs(lines)), 1.0)
    dispersion = spread / scale
    return float(np.clip(1.0 - dispersion * 4.0, 0.0, 1.0))


def market_quality_score(quotes: Sequence[BookQuote]) -> float:
    """Book count and price sanity, on 0-1."""
    n = len(quotes)
    if n == 0:
        return 0.0
    count_score = float(np.clip(np.log1p(n) / np.log1p(8), 0.0, 1.0))
    prices = np.array([abs(q.american) for q in quotes], dtype=float)
    # Extreme prices are thinner markets with wider effective vig.
    price_score = float(np.clip(1.0 - (np.median(prices) - 110.0) / 400.0, 0.3, 1.0))
    return float(0.7 * count_score + 0.3 * price_score)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def evaluate_prop(
    *,
    player_id: str,
    player_name: str,
    team: str,
    opponent: str,
    position: str,
    market: str,
    distribution,
    over_quotes: Sequence[BookQuote],
    under_quotes: Sequence[BookQuote],
    confidence_inputs: ConfidenceInputs,
    reasons: Sequence[str] = (),
    thresholds: BetThresholds | None = None,
    devig_method: DevigMethod = "multiplicative",
    injury_uncertainty: float = 0.0,
    sufficient_data: bool = True,
    model_version: str = "unversioned",
    bankroll_fraction: float = 0.25,
) -> list[PropEdge]:
    """Score every book's price on both sides of one prop.

    De-vigging is done per book against that same book's opposite side, because
    the margin differs by book. Comparing a DraftKings over price to a
    FanDuel-derived fair probability would smuggle the difference in their holds
    into the edge.
    """
    thresholds = thresholds or BetThresholds()
    all_quotes = list(over_quotes) + list(under_quotes)
    if not all_quotes:
        return []

    cons = consensus_line(all_quotes)
    ci = ConfidenceInputs(**vars(confidence_inputs))
    ci.market_quality = market_quality_score(all_quotes)
    ci.market_agreement = market_agreement_score(all_quotes)

    under_by_book: dict[tuple[str, float], BookQuote] = {
        (q.bookmaker, q.line): q for q in under_quotes
    }
    over_by_book: dict[tuple[str, float], BookQuote] = {
        (q.bookmaker, q.line): q for q in over_quotes
    }

    results: list[PropEdge] = []
    for side, quotes, opposite in (
        ("over", over_quotes, under_by_book),
        ("under", under_quotes, over_by_book),
    ):
        for q in quotes:
            model_p_over = distribution.prob_over(q.line)
            model_p = model_p_over if side == "over" else 1.0 - model_p_over

            counter = opposite.get((q.bookmaker, q.line))
            if counter is not None:
                fair_side, fair_other = devig_two_way(
                    q.american, counter.american, method=devig_method
                )
                market_p = fair_side
            else:
                # No counterpart at this book and number. Fall back to the raw
                # implied probability with a half-hold haircut so a one-sided
                # quote can never manufacture an edge out of pure vig.
                raw = american_to_prob(q.american)
                market_p = float(np.clip(raw - 0.022, 1e-4, 0.9999))

            edge = probability_edge(model_p, market_p)
            ev = expected_value(model_p, q.american)

            mc_err = getattr(distribution, "monte_carlo_error", None)
            if callable(mc_err):
                err = mc_err(q.line)
                ci.simulation_precision = float(np.clip(1.0 - err / 0.02, 0.0, 1.0))

            conf, parts = confidence_score(ci, edge)
            summary = distribution.summary()
            clipped = float(np.clip(model_p, 1e-4, 0.9999))

            results.append(
                PropEdge(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    position=position,
                    market=market,
                    side=side,  # type: ignore[arg-type]
                    line=q.line,
                    bookmaker=q.bookmaker,
                    american=q.american,
                    model_prob=model_p,
                    market_prob=market_p,
                    edge=edge,
                    ev=ev,
                    fair_american=prob_to_american(clipped),
                    projection=summary["mean"],
                    median=summary["median"],
                    p25=summary["p25"],
                    p75=summary["p75"],
                    confidence=conf,
                    confidence_parts=parts,
                    grade=grade_from(edge, conf),
                    recommendation=classify(
                        edge, ev, conf, cons["n_books"], injury_uncertainty,
                        thresholds, sufficient_data,
                    ),
                    kelly=kelly_fraction(model_p, q.american, fraction=bankroll_fraction),
                    consensus=cons,
                    reasons=list(reasons),
                    model_version=model_version,
                )
            )
    return results


def rank_edges(edges: Sequence[PropEdge], top_n: int | None = None) -> list[PropEdge]:
    """Rank by a blended score rather than by raw edge.

    Ranking on edge alone promotes exactly the bets most likely to be data
    errors: thin markets, stale numbers, and players whose role the model has
    misread. The blend pulls EV, edge, confidence and market depth together, and
    penalises a bet whose edge is small relative to simulation noise.
    """
    def score(e: PropEdge) -> float:
        depth = float(np.clip(e.consensus.get("n_books", 0) / 6.0, 0.0, 1.0))

        # A measured edge is an estimate, and its reliability depends on how
        # much market evidence stands behind it. One book quoting a number
        # nobody else quotes is weak evidence that the number is wrong, so the
        # edge is shrunk toward zero before it is scored. Without this, ranking
        # systematically promotes the thinnest markets, which is where stale
        # lines and bad player mappings live.
        reliability = 0.30 + 0.45 * depth + 0.25 * (e.confidence / 100.0)
        shrunk_edge = e.edge * reliability
        shrunk_ev = e.ev * reliability

        base = (
            0.35 * float(np.clip(shrunk_ev / 0.25, -1.0, 1.0))
            + 0.30 * float(np.clip(shrunk_edge / 0.12, -1.0, 1.0))
            + 0.25 * (e.confidence / 100.0)
            + 0.10 * depth
        )
        # Anything the filters sent to review or blocked for thin data ranks
        # below every genuine candidate rather than topping the board.
        if e.recommendation in ("needs_review", "insufficient_data"):
            base -= 1.0
        return base

    ordered = sorted(edges, key=score, reverse=True)
    return ordered[:top_n] if top_n else ordered


def best_by_market(edges: Sequence[PropEdge]) -> dict[str, PropEdge]:
    """Best available price per (player, market, side), for line shopping."""
    best: dict[tuple[str, str, str], PropEdge] = {}
    for e in edges:
        key = (e.player_id, e.market, e.side)
        if key not in best or e.ev > best[key].ev:
            best[key] = e
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in best.items()}
