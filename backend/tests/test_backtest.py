"""Backtest correctness tests. The look-ahead guards matter most."""

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from app.backtest.engine import (
    LookAheadError,
    PointInTimeReader,
    brier_score,
    calibration_curve,
    closing_line_value,
    decision_timestamp,
    edge_bucket,
    expected_calibration_error,
    log_loss,
    max_drawdown,
    sharpe_ratio,
    summarize_by_bucket,
    walk_forward_splits,
)


class TestLookAheadGuards(unittest.TestCase):
    def setUp(self):
        self.reader = PointInTimeReader(
            as_of=datetime(2026, 9, 6, 13, 0, tzinfo=timezone.utc))

    def test_post_game_tables_are_never_readable(self):
        for table in ("player_game_stats", "team_game_stats", "player_props"):
            with self.subTest(table=table):
                with self.assertRaises(LookAheadError):
                    self.reader.read(table)

    def test_unregistered_tables_raise_rather_than_leak(self):
        with self.assertRaises(LookAheadError):
            self.reader.read("some_new_table_nobody_registered")

    def test_registered_tables_filter_on_the_observation_column(self):
        # No connection, so it fails at execution -- but only after passing the
        # guard, which is what we are checking.
        with self.assertRaises(RuntimeError):
            self.reader.read("odds_snapshots")

    def test_decision_timestamps_precede_kickoff(self):
        kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        for offset in (10080, 1440, 60):
            self.assertLess(decision_timestamp(kickoff, offset), kickoff)
        self.assertEqual(
            decision_timestamp(kickoff, 1440), kickoff - timedelta(days=1))


class TestWalkForward(unittest.TestCase):
    def test_training_always_precedes_testing(self):
        for split in walk_forward_splits([2021, 2022, 2023, 2024, 2025]):
            self.assertTrue(all(t < split["test"][0] for t in split["train"]))

    def test_window_expands(self):
        splits = walk_forward_splits([2020, 2021, 2022, 2023, 2024, 2025])
        sizes = [len(s["train"]) for s in splits]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(splits[0]["test"], [2023])

    def test_too_few_seasons_yields_no_splits(self):
        self.assertEqual(walk_forward_splits([2024, 2025]), [])


class TestMetrics(unittest.TestCase):
    def test_brier_is_zero_for_perfect_predictions(self):
        self.assertAlmostEqual(
            brier_score(np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1])), 0.0)

    def test_brier_of_a_coin_flip(self):
        self.assertAlmostEqual(
            brier_score(np.array([0.5, 0.5]), np.array([1, 0])), 0.25)

    def test_log_loss_hand_computed(self):
        # Single prediction of 0.8 on a win: -ln(0.8) = 0.22314
        self.assertAlmostEqual(
            log_loss(np.array([0.8]), np.array([1])), 0.2231435513, places=8)

    def test_log_loss_handles_certainty_without_exploding(self):
        self.assertTrue(np.isfinite(log_loss(np.array([1.0]), np.array([0]))))

    def test_calibration_is_perfect_when_the_model_is_honest(self):
        rng = np.random.default_rng(3)
        probs = rng.uniform(0.05, 0.95, 60_000)
        outcomes = (rng.random(60_000) < probs).astype(float)
        curve = calibration_curve(probs, outcomes)
        self.assertLess(expected_calibration_error(curve), 0.01)
        for b in curve:
            if b["n"] > 100:
                self.assertAlmostEqual(b["predicted"], b["observed"], delta=0.02)

    def test_calibration_detects_an_overconfident_model(self):
        rng = np.random.default_rng(4)
        stated = rng.uniform(0.55, 0.85, 40_000)
        true = stated - 0.10
        outcomes = (rng.random(40_000) < true).astype(float)
        curve = calibration_curve(stated, outcomes)
        self.assertGreater(expected_calibration_error(curve), 0.05)

    def test_calibration_reports_standard_errors(self):
        rng = np.random.default_rng(9)
        probs = rng.uniform(0.1, 0.9, 5_000)
        outcomes = (rng.random(5_000) < probs).astype(float)
        for b in calibration_curve(probs, outcomes):
            if b["n"]:
                self.assertGreater(b["std_error"], 0.0)

    def test_max_drawdown(self):
        # 100 -> 120 -> 90: trough is 25% below the 120 peak.
        self.assertAlmostEqual(max_drawdown([100, 120, 90, 110]), 0.25, places=9)
        self.assertAlmostEqual(max_drawdown([100, 110, 120]), 0.0, places=9)

    def test_sharpe_rewards_consistency(self):
        steady = [0.02] * 40
        choppy = [0.20, -0.16] * 20
        self.assertGreater(sharpe_ratio(steady), sharpe_ratio(choppy))

    def test_clv_is_positive_when_the_line_moves_your_way(self):
        # Bet +120, closes -110: the market moved toward the bet.
        self.assertGreater(closing_line_value(120, -110), 0)
        self.assertLess(closing_line_value(-110, 120), 0)

    def test_edge_buckets_partition_cleanly(self):
        self.assertEqual(edge_bucket(0.01), "0%-2%")
        self.assertEqual(edge_bucket(0.05), "4%-6%")
        self.assertEqual(edge_bucket(0.22), "15%+")
        self.assertEqual(edge_bucket(-0.01), "negative")

    def test_bucket_summary(self):
        bets = [
            {"market": "receiving_yards", "stake_units": 1.0,
             "profit_units": 0.91, "result": "win", "edge": 0.06},
            {"market": "receiving_yards", "stake_units": 1.0,
             "profit_units": -1.0, "result": "loss", "edge": 0.04},
            {"market": "rushing_yards", "stake_units": 1.0,
             "profit_units": 0.91, "result": "win", "edge": 0.08},
        ]
        out = summarize_by_bucket(bets, "market")
        self.assertEqual(out["receiving_yards"]["n"], 2)
        self.assertAlmostEqual(out["receiving_yards"]["win_rate"], 0.5)
        self.assertAlmostEqual(out["rushing_yards"]["roi"], 0.91, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
