"""Backtesting with enforced point-in-time correctness.

Look-ahead bias is the failure mode that makes a bad model look profitable, and
it is almost never introduced deliberately. It arrives through a join: a usage
table keyed only by game_id that happens to contain post-game statistics, a
depth chart row whose `week` matches but whose `observed_at` is Sunday evening,
a closing line used as "the market price".

So the guard is structural rather than procedural. Every query issued during a
backtest goes through ``PointInTimeReader``, which requires an ``as_of``
timestamp and filters every table on its observation column. A table without an
observation column cannot be read by the backtester at all — it raises rather
than silently returning rows from the future.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Decision points, in minutes before kickoff. Running all of them shows where
# the model's edge actually lives: an edge that only exists at closing is not
# an edge you could have bet.
DECISION_OFFSETS_MINUTES = [10080, 2880, 1440, 360, 60, 0]

# Every table the backtester may read, and the column that timestamps the
# observation. A table not in this map is unreadable during a backtest.
OBSERVATION_COLUMNS = {
    "odds_snapshots": "snapshot_at",
    "injuries": "observed_at",
    "depth_charts": "observed_at",
    "weather": "observed_at",
    "player_projections": "generated_at",
    "line_movements": "observed_at",
    "player_game_stats": None,      # post-game by definition: never readable
    "team_game_stats": None,
    "player_props": None,           # materialised current state: never readable
}


class LookAheadError(RuntimeError):
    """Raised when a backtest tries to read data that did not exist yet."""


@dataclass
class PointInTimeReader:
    """Wraps data access so nothing from after ``as_of`` can be returned."""

    as_of: datetime
    connection: Any = None

    def read(
        self,
        table: str,
        where: str = "TRUE",
        params: Sequence[Any] = (),
        columns: str = "*",
    ):
        if table not in OBSERVATION_COLUMNS:
            raise LookAheadError(
                f"Table '{table}' is not registered for point-in-time reads. "
                "Add it to OBSERVATION_COLUMNS with its observation column, or "
                "do not read it during a backtest."
            )
        column = OBSERVATION_COLUMNS[table]
        if column is None:
            raise LookAheadError(
                f"Table '{table}' contains post-hoc data and can never be read "
                f"at decision time. Reading it would leak the outcome into the "
                f"prediction."
            )
        sql = (f"SELECT {columns} FROM {table} "
               f"WHERE ({where}) AND {column} <= %s")
        return self._execute(sql, (*params, self.as_of))

    def latest_per(
        self,
        table: str,
        partition_by: Sequence[str],
        where: str = "TRUE",
        params: Sequence[Any] = (),
    ):
        """Most recent row per entity as of the decision time.

        This is the correct way to read an injury report or a depth chart: not
        "the row for week 5" but "the most recent row observed before the
        moment we would have bet".
        """
        column = OBSERVATION_COLUMNS.get(table)
        if column is None:
            raise LookAheadError(f"Table '{table}' is not point-in-time readable")
        partition = ", ".join(partition_by)
        sql = (
            f"SELECT DISTINCT ON ({partition}) * FROM {table} "
            f"WHERE ({where}) AND {column} <= %s "
            f"ORDER BY {partition}, {column} DESC"
        )
        return self._execute(sql, (*params, self.as_of))

    def _execute(self, sql: str, params: tuple):
        if self.connection is None:
            raise RuntimeError("PointInTimeReader has no database connection")
        with self.connection.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@dataclass
class BacktestConfig:
    seasons: Sequence[int]
    decision_offset_minutes: int = 1440
    markets: Sequence[str] = ("passing_yards", "rushing_yards",
                              "receiving_yards", "anytime_td", "multi_td")
    min_edge: float = 0.03
    min_confidence: float = 60.0
    staking: str = "flat"           # flat / kelly
    kelly_fraction: float = 0.25
    starting_bankroll: float = 100.0
    devig_method: str = "multiplicative"


@dataclass
class BacktestResult:
    n_bets: int
    roi: float
    win_rate: float
    units_won: float
    avg_edge: float
    avg_clv: float
    avg_odds: float
    max_drawdown: float
    sharpe: float
    brier: float
    log_loss: float
    calibration: list[dict] = field(default_factory=list)
    by_bucket: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_bets": self.n_bets,
            "roi": round(self.roi, 4),
            "win_rate": round(self.win_rate, 4),
            "units_won": round(self.units_won, 2),
            "avg_edge": round(self.avg_edge, 4),
            "avg_clv": round(self.avg_clv, 4),
            "max_drawdown": round(self.max_drawdown, 3),
            "sharpe": round(self.sharpe, 3),
            "brier": round(self.brier, 5),
            "log_loss": round(self.log_loss, 5),
        }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions. Lower is better."""
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_curve(
    probs: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> list[dict]:
    """Reliability diagram data.

    If the model says 60%, the event should happen about 60% of the time. This
    matters more than ROI over a short sample: a well-calibrated model with a
    losing month is far more trustworthy than a poorly calibrated one with a
    winning month, because the winning month is mostly variance.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_low": float(lo), "bin_high": float(hi),
                         "n": 0, "predicted": None, "observed": None})
            continue
        rows.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": n,
            "predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
            # Binomial standard error, so the UI can show whether a bin's
            # deviation is meaningful or just a thin sample.
            "std_error": float(np.sqrt(max(y[mask].mean() *
                                           (1 - y[mask].mean()), 1e-9) / n)),
        })
    return rows


def expected_calibration_error(curve: Sequence[dict]) -> float:
    total = sum(b["n"] for b in curve if b["n"])
    if total == 0:
        return 0.0
    return float(sum(
        b["n"] / total * abs(b["predicted"] - b["observed"])
        for b in curve if b["n"] and b["predicted"] is not None
    ))


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough decline in the bankroll curve."""
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max((peak - arr) / np.where(peak == 0, 1, peak)))


def sharpe_ratio(returns: Sequence[float], periods_per_season: int = 18) -> float:
    """Return per unit of volatility, annualised to a season.

    Reported alongside ROI because two models with the same ROI and very
    different variance are not equally useful, and the higher-variance one needs
    a much longer sample before its ROI means anything.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        # Constant returns have no volatility, so the ratio is unbounded.
        # Collapsing this to zero would rank a riskless winning strategy below
        # a volatile one, which is backwards. Return a large sentinel with the
        # sign of the mean instead.
        if r.mean() == 0.0:
            return 0.0
        return float(np.sign(r.mean()) * 1e3)
    return float(r.mean() / sd * np.sqrt(periods_per_season))


def closing_line_value(
    bet_odds: float,
    closing_odds: float,
    devig_method: str = "multiplicative",
) -> float:
    """CLV in probability points, using raw implied probabilities.

    Beating the close is the best short-run evidence that a model finds real
    inefficiency, because it does not depend on whether the bets happened to
    win. A model with positive CLV and a losing month is probably fine. A model
    with negative CLV and a winning month is probably lucky.
    """
    from ..core.odds import american_to_prob
    return float(american_to_prob(closing_odds) - american_to_prob(bet_odds))


def walk_forward_splits(
    seasons: Sequence[int],
    min_train_seasons: int = 3,
) -> list[dict]:
    """Expanding-window splits that never train on the future.

    Random k-fold on a shuffled NFL dataset is the most common way a prop model
    reports a validation score it cannot reproduce live: it trains on week 12
    and tests on week 4 of the same season, so team and player form leak across
    the split. Every split here trains strictly before it tests.
    """
    seasons = sorted(seasons)
    splits = []
    for i in range(min_train_seasons, len(seasons)):
        splits.append({
            "train": seasons[:i],
            "test": [seasons[i]],
            "train_end": seasons[i - 1],
        })
    return splits


def decision_timestamp(kickoff: datetime, offset_minutes: int) -> datetime:
    return kickoff - timedelta(minutes=offset_minutes)


def summarize_by_bucket(
    bets: Iterable[dict],
    key: str,
    bucket_fn=None,
) -> dict:
    """Slice results by market, position, edge bucket, book, and so on."""
    from collections import defaultdict

    groups: dict[Any, list[dict]] = defaultdict(list)
    for b in bets:
        value = bucket_fn(b[key]) if bucket_fn else b.get(key)
        groups[value].append(b)

    out = {}
    for value, rows in groups.items():
        staked = sum(r["stake_units"] for r in rows)
        profit = sum(r["profit_units"] for r in rows)
        wins = sum(1 for r in rows if r["result"] == "win")
        settled = sum(1 for r in rows if r["result"] in ("win", "loss"))
        out[str(value)] = {
            "n": len(rows),
            "roi": round(profit / staked, 4) if staked else 0.0,
            "win_rate": round(wins / settled, 4) if settled else 0.0,
            "units": round(profit, 2),
            "avg_edge": round(float(np.mean([r["edge"] for r in rows])), 4),
        }
    return out


def edge_bucket(edge: float) -> str:
    for lo, hi in ((0.00, 0.02), (0.02, 0.04), (0.04, 0.06),
                   (0.06, 0.09), (0.09, 0.15)):
        if lo <= edge < hi:
            return f"{lo:.0%}-{hi:.0%}"
    return "15%+" if edge >= 0.15 else "negative"
