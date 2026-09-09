"""Backtest runner: replay past games through the live projection path.

The engine module holds the metrics and the look-ahead guards. This holds the
loop that produces bets to measure. Two rules shape it:

* It runs the *same* pipeline the live slice runs, imported rather than
  reimplemented. A backtest with its own copy of the projection code measures
  a model nobody ships, and every divergence is a way for the result to be
  optimistic about the thing actually being bet.
* It only counts wagers the model would actually place. ``needs_review`` and
  ``pass`` are not bets, and a backtest that grades them anyway reports the
  performance of a strategy the shipped thresholds forbid.

Data access is injected rather than imported so the loop can be exercised
offline. The live wiring lives in ``scripts/backtest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..core.odds import expected_value, kelly_fraction
from .engine import (
    BacktestConfig,
    BacktestResult,
    brier_score,
    calibration_curve,
    closing_line_value,
    decision_timestamp,
    edge_bucket,
    log_loss,
    max_drawdown,
    sharpe_ratio,
    summarize_by_bucket,
)

# Recommendations that represent an actual wager. "pass" declines the bet and
# "needs_review" routes it away from the dashboard; neither is a position.
BETTABLE = ("strong", "moderate", "lean")

# Market -> the column in the weekly frame that settles it.
MARKET_STAT_COLUMNS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


@dataclass
class SettledBet:
    game_id: str
    kickoff: datetime
    player_id: str
    player_name: str
    market: str
    side: str
    line: float
    bookmaker: str
    american: float
    model_prob: float
    market_prob: float
    edge: float
    confidence: float
    recommendation: str
    actual: float
    result: str
    stake: float
    pnl: float
    clv: float = 0.0
    # Whether a closing price was actually found for this exact bet. Averaging
    # a zero in for every bet that had none drags the mean toward zero and
    # makes a real edge look like noise, so the two cases are kept apart.
    clv_available: bool = False
    # Every candidate the model evaluated is graded, not only the ones the
    # thresholds backed. Calibration has to be measured on the full population
    # of probabilities the model emits; measuring it only on the subset that
    # cleared a threshold conditions on the model's own confidence and hides
    # exactly the overconfidence being looked for.
    selected: bool = True
    projection: float = 0.0

    @property
    def won(self) -> int:
        return 1 if self.result == "win" else 0


def grade(side: str, line: float, actual: float) -> str:
    """Settle one prop. A number landing exactly on the line is a push."""
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return "void"
    if float(actual) == float(line):
        return "push"
    went_over = float(actual) > float(line)
    return "win" if (side == "over") == went_over else "loss"


def actual_values(weekly: pd.DataFrame, season: int,
                  week: int) -> dict[tuple[str, str], float]:
    """Settled stat per (player, market) for one week.

    Read from the *post-game* weekly frame, which is exactly the data a
    projection is forbidden to see. It is only ever used to score a bet after
    the decision has been made and recorded.
    """
    if weekly.empty:
        return {}
    df = weekly
    if "season" in df.columns:
        df = df[df["season"] == season]
    df = df[df["week"] == week]
    id_col = "player_id" if "player_id" in df.columns else "gsis_id"
    out: dict[tuple[str, str], float] = {}
    for market, column in MARKET_STAT_COLUMNS.items():
        if column not in df.columns:
            continue
        for pid, value in zip(df[id_col], df[column]):
            if pd.isna(pid):
                continue
            out[(str(pid), market)] = (0.0 if pd.isna(value)
                                       else float(value))
    return out


def stake_for(edge, config: BacktestConfig, bankroll: float) -> float:
    """Units risked on one bet.

    Flat staking is the honest default for a model whose calibration is not
    yet established: Kelly sizing on a miscalibrated probability amplifies the
    calibration error rather than the edge.
    """
    if config.staking != "kelly":
        return 1.0
    fraction = kelly_fraction(edge.model_prob, edge.american,
                              fraction=config.kelly_fraction)
    return max(0.0, float(fraction) * bankroll)


def select(edges: Iterable, config: BacktestConfig) -> list:
    """The subset the shipped thresholds would actually have backed."""
    chosen = []
    for e in edges:
        if e.recommendation not in BETTABLE:
            continue
        if e.edge < config.min_edge or e.confidence < config.min_confidence:
            continue
        if e.market not in MARKET_STAT_COLUMNS:
            continue
        chosen.append(e)
    return chosen


def settle(
    edges: Sequence,
    actuals: dict[tuple[str, str], float],
    config: BacktestConfig,
    *,
    game_id: str,
    kickoff: datetime,
    bankroll: float = 100.0,
    closing: dict[tuple[str, str, str], float] | None = None,
    grade_all: bool = False,
) -> list[SettledBet]:
    """Grade one game against what actually happened.

    ``grade_all`` keeps every evaluated candidate, flagged with whether the
    thresholds would have backed it. Staking and P&L stay meaningful only for
    selected rows; the rest exist so calibration can be measured on the whole
    population of probabilities the model emits.
    """
    chosen = {id(e) for e in select(edges, config)}
    population = list(edges) if grade_all else select(edges, config)
    out = []
    for e in population:
        picked = id(e) in chosen
        actual = actuals.get((e.player_id, e.market))
        if actual is None:
            # No stat line: the player did not take the field. Void rather
            # than score it, or a scratch reads as a winning under.
            continue
        result = grade(e.side, e.line, actual)
        if result == "void":
            continue
        stake = stake_for(e, config, bankroll) if picked else 0.0
        if result == "push" or not picked:
            pnl = 0.0
        else:
            pnl = (expected_value(1.0, e.american, stake) if result == "win"
                   else -stake)
        clv, clv_available = 0.0, False
        if closing:
            # Matched on the line as well as the side: a price at 82.5 is not
            # a close for a bet struck at 79.5, and comparing them would score
            # a line move as if it were a better price for the same wager.
            close = closing.get((e.player_id, e.market, e.side, float(e.line)))
            if close is not None:
                clv = closing_line_value(e.american, close,
                                         config.devig_method)
                clv_available = True
        out.append(SettledBet(
            game_id=game_id, kickoff=kickoff, player_id=e.player_id,
            player_name=e.player_name, market=e.market, side=e.side,
            line=e.line, bookmaker=e.bookmaker, american=e.american,
            model_prob=e.model_prob, market_prob=e.market_prob, edge=e.edge,
            confidence=e.confidence, recommendation=e.recommendation,
            actual=float(actual), result=result, stake=stake, pnl=pnl,
            clv=clv, clv_available=clv_available, selected=picked,
            projection=float(getattr(e, "projection", 0.0) or 0.0),
        ))
    return out


def equity_curve(bets: Sequence[SettledBet],
                 starting_bankroll: float) -> list[float]:
    equity = [float(starting_bankroll)]
    for bet in bets:
        equity.append(equity[-1] + bet.pnl)
    return equity


def aggregate(bets: Sequence[SettledBet],
              config: BacktestConfig) -> BacktestResult:
    """Roll settled bets into the engine's result type."""
    decided = [b for b in bets
               if b.selected and b.result in ("win", "loss")]
    if not decided:
        return BacktestResult(
            n_bets=0, roi=0.0, win_rate=0.0, units_won=0.0, avg_edge=0.0,
            avg_clv=0.0, avg_odds=0.0, max_drawdown=0.0, sharpe=0.0,
            brier=0.0, log_loss=0.0,
        )
    probs = np.array([b.model_prob for b in decided], dtype=float)
    outcomes = np.array([b.won for b in decided], dtype=float)
    staked = sum(b.stake for b in decided)
    pnl = sum(b.pnl for b in decided)
    equity = equity_curve([b for b in bets if b.selected],
                          config.starting_bankroll)
    returns = [b.pnl / b.stake if b.stake else 0.0 for b in decided]
    curve = calibration_curve(probs, outcomes)
    return BacktestResult(
        n_bets=len(decided),
        roi=pnl / staked if staked else 0.0,
        win_rate=float(outcomes.mean()),
        units_won=pnl,
        avg_edge=float(np.mean([b.edge for b in decided])),
        avg_clv=(float(np.mean([b.clv for b in decided if b.clv_available]))
                 if any(b.clv_available for b in decided) else 0.0),
        avg_odds=float(np.mean([b.american for b in decided])),
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe_ratio(returns),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        calibration=curve,
        by_bucket=summarize_by_bucket(
            # summarize_by_bucket settles in units and reads "result"
            # directly, so hand it that vocabulary rather than this module's.
            [{"bucket": edge_bucket(b.edge), "result": b.result,
              "stake_units": b.stake, "profit_units": b.pnl,
              "edge": b.edge} for b in decided],
            key="bucket",
        ),
    )


def run_backtest(
    games: Iterable[dict],
    config: BacktestConfig,
    *,
    propose: Callable[[dict, datetime], Sequence],
    actuals_for: Callable[[dict], dict],
    closing_for: Callable[[dict], dict] | None = None,
    on_game: Callable[[dict, list[SettledBet], str | None], None] | None = None,
    grade_all: bool = False,
) -> tuple[BacktestResult, list[SettledBet]]:
    """Replay games in chronological order and grade what the model backed.

    ``propose`` is handed the game and the *decision* timestamp, never the
    kickoff, so nothing downstream can reach for a price or a stat line that
    did not exist when the wager would have been placed.

    A game that raises is skipped and reported rather than aborting the run:
    one unresolvable roster should not cost a season of evidence.
    """
    ordered = sorted(games, key=lambda g: g["kickoff"])
    all_bets: list[SettledBet] = []
    bankroll = float(config.starting_bankroll)
    for game in ordered:
        decided_at = decision_timestamp(game["kickoff"],
                                        config.decision_offset_minutes)
        error = None
        bets: list[SettledBet] = []
        try:
            edges = propose(game, decided_at)
            bets = settle(
                edges, actuals_for(game), config,
                game_id=str(game.get("game_id", "")),
                kickoff=game["kickoff"], bankroll=bankroll,
                closing=closing_for(game) if closing_for else None,
                grade_all=grade_all,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        all_bets.extend(bets)
        bankroll += sum(b.pnl for b in bets if b.selected)
        if on_game is not None:
            on_game(game, [b for b in bets if b.selected], error)
    return aggregate(all_bets, config), all_bets
