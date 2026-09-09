"""Runner tests.

The backtest is the instrument the model is judged by, so these check the ways
an instrument can flatter the thing it measures: grading bets the thresholds
forbid, scoring players who never took the field, or letting the projection see
a price from after the decision.
"""

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.backtest.engine import BacktestConfig
from app.backtest.runner import (
    actual_values,
    aggregate,
    grade,
    run_backtest,
    select,
    settle,
)

KICKOFF = datetime(2025, 10, 5, 17, 0, tzinfo=timezone.utc)


@dataclass
class FakeEdge:
    player_id: str = "p1"
    player_name: str = "Test Player"
    market: str = "receiving_yards"
    side: str = "over"
    line: float = 50.0
    bookmaker: str = "book"
    american: float = -110
    model_prob: float = 0.60
    market_prob: float = 0.50
    edge: float = 0.10
    confidence: float = 75.0
    recommendation: str = "strong"


def config(**kw):
    return BacktestConfig(seasons=[2025], **kw)


class TestGrading(unittest.TestCase):
    def test_over_and_under(self):
        self.assertEqual(grade("over", 50.0, 61.0), "win")
        self.assertEqual(grade("over", 50.0, 40.0), "loss")
        self.assertEqual(grade("under", 50.0, 40.0), "win")
        self.assertEqual(grade("under", 50.0, 61.0), "loss")

    def test_landing_on_the_line_is_a_push(self):
        self.assertEqual(grade("over", 50.0, 50.0), "push")
        self.assertEqual(grade("under", 50.0, 50.0), "push")


class TestSelection(unittest.TestCase):
    """A backtest that grades bets the model declined measures a strategy the
    shipped thresholds forbid."""

    def test_needs_review_is_not_a_bet(self):
        edges = [FakeEdge(recommendation="needs_review")]
        self.assertEqual(select(edges, config()), [])

    def test_pass_is_not_a_bet(self):
        edges = [FakeEdge(recommendation="pass")]
        self.assertEqual(select(edges, config()), [])

    def test_thin_edges_and_low_confidence_are_excluded(self):
        self.assertEqual(select([FakeEdge(edge=0.01)], config(min_edge=0.03)),
                         [])
        self.assertEqual(
            select([FakeEdge(confidence=40.0)], config(min_confidence=60.0)),
            [])

    def test_a_qualifying_bet_survives(self):
        self.assertEqual(len(select([FakeEdge()], config())), 1)


class TestSettlement(unittest.TestCase):
    def settle_one(self, edge, actuals, **kw):
        return settle([edge], actuals, config(**kw), game_id="g1",
                      kickoff=KICKOFF)

    def test_a_player_who_did_not_play_is_voided_not_won(self):
        """The trap: no stat line scored as zero makes every under a winner."""
        bets = self.settle_one(FakeEdge(side="under"), {})
        self.assertEqual(bets, [])

    def test_a_win_pays_the_price(self):
        bets = self.settle_one(FakeEdge(american=-110),
                               {("p1", "receiving_yards"): 80.0})
        self.assertEqual(bets[0].result, "win")
        self.assertAlmostEqual(bets[0].pnl, 0.9091, places=3)

    def test_a_loss_costs_the_stake(self):
        bets = self.settle_one(FakeEdge(),
                               {("p1", "receiving_yards"): 10.0})
        self.assertEqual(bets[0].result, "loss")
        self.assertAlmostEqual(bets[0].pnl, -1.0)

    def test_a_push_returns_the_stake(self):
        bets = self.settle_one(FakeEdge(),
                               {("p1", "receiving_yards"): 50.0})
        self.assertEqual(bets[0].result, "push")
        self.assertEqual(bets[0].pnl, 0.0)

    def test_clv_is_recorded_against_the_close(self):
        """Struck at +100, closed at -120: the price moved our way."""
        bets = settle([FakeEdge(american=100)],
                      {("p1", "receiving_yards"): 80.0}, config(),
                      game_id="g1", kickoff=KICKOFF,
                      closing={("p1", "receiving_yards", "over", 50.0): -120})
        self.assertGreater(bets[0].clv, 0.0)
        self.assertTrue(bets[0].clv_available)

    def test_a_close_at_a_different_line_is_not_a_close(self):
        """A price at 60.5 is not the close for a bet struck at 50.5; scoring
        it as one would read a line move as a better price."""
        bets = settle([FakeEdge(american=100)],
                      {("p1", "receiving_yards"): 80.0}, config(),
                      game_id="g1", kickoff=KICKOFF,
                      closing={("p1", "receiving_yards", "over", 60.5): -120})
        self.assertEqual(bets[0].clv, 0.0)
        self.assertFalse(bets[0].clv_available)

    def test_negative_clv_when_the_price_moved_against_us(self):
        bets = settle([FakeEdge(american=-120)],
                      {("p1", "receiving_yards"): 80.0}, config(),
                      game_id="g1", kickoff=KICKOFF,
                      closing={("p1", "receiving_yards", "over", 50.0): 100})
        self.assertLess(bets[0].clv, 0.0)

    def test_bets_without_a_close_do_not_drag_the_average(self):
        """Averaging a zero in for every missing close makes a real edge look
        like noise."""
        with_close = settle([FakeEdge(american=100)],
                            {("p1", "receiving_yards"): 80.0}, config(),
                            game_id="g1", kickoff=KICKOFF,
                            closing={("p1", "receiving_yards", "over", 50.0): -140})
        without = settle([FakeEdge(player_id="p2", american=100)],
                         {("p2", "receiving_yards"): 80.0}, config(),
                         game_id="g2", kickoff=KICKOFF, closing={})
        result = aggregate(with_close + without, config())
        self.assertAlmostEqual(result.avg_clv, with_close[0].clv, places=6)


class TestActualValues(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            {"player_id": "p1", "season": 2025, "week": 5,
             "receiving_yards": 74.0, "rushing_yards": 0.0,
             "passing_yards": 0.0},
            {"player_id": "p1", "season": 2025, "week": 6,
             "receiving_yards": 12.0, "rushing_yards": 0.0,
             "passing_yards": 0.0},
        ])

    def test_reads_only_the_requested_week(self):
        got = actual_values(self.frame(), 2025, 5)
        self.assertEqual(got[("p1", "receiving_yards")], 74.0)

    def test_a_different_week_is_a_different_number(self):
        self.assertEqual(
            actual_values(self.frame(), 2025, 6)[("p1", "receiving_yards")],
            12.0)


class TestRunnerLoop(unittest.TestCase):
    def games(self):
        return [
            {"game_id": "late", "kickoff": KICKOFF + timedelta(days=7),
             "season": 2025, "week": 6},
            {"game_id": "early", "kickoff": KICKOFF, "season": 2025,
             "week": 5},
        ]

    def test_games_are_replayed_in_order(self):
        seen = []
        run_backtest(self.games(), config(),
                     propose=lambda g, t: (seen.append(g["game_id"]) or []),
                     actuals_for=lambda g: {})
        self.assertEqual(seen, ["early", "late"])

    def test_the_projection_never_sees_kickoff(self):
        """The whole point of the exercise: prices from after the decision
        would make any result meaningless."""
        stamps = []

        def propose(game, decided_at):
            stamps.append((game["kickoff"], decided_at))
            return []

        run_backtest(self.games(), config(decision_offset_minutes=1440),
                     propose=propose, actuals_for=lambda g: {})
        for kickoff, decided_at in stamps:
            self.assertLess(decided_at, kickoff)
            self.assertEqual(kickoff - decided_at, timedelta(minutes=1440))

    def test_one_broken_game_does_not_abort_the_run(self):
        errors = []

        def propose(game, decided_at):
            if game["game_id"] == "early":
                raise RuntimeError("no roster")
            return [FakeEdge()]

        result, bets = run_backtest(
            self.games(), config(),
            propose=propose,
            actuals_for=lambda g: {("p1", "receiving_yards"): 80.0},
            on_game=lambda g, b, e: errors.append((g["game_id"], e)))
        self.assertEqual(result.n_bets, 1)
        self.assertIn("no roster", dict(errors)["early"])


class TestAggregation(unittest.TestCase):
    def test_roi_and_win_rate(self):
        actuals = {("p1", "receiving_yards"): 80.0}
        losses = {("p1", "receiving_yards"): 10.0}
        bets = (settle([FakeEdge(american=100)], actuals, config(),
                       game_id="g1", kickoff=KICKOFF)
                + settle([FakeEdge(american=100)], losses, config(),
                         game_id="g2", kickoff=KICKOFF))
        result = aggregate(bets, config())
        self.assertEqual(result.n_bets, 2)
        self.assertAlmostEqual(result.win_rate, 0.5)
        self.assertAlmostEqual(result.units_won, 0.0)
        self.assertAlmostEqual(result.roi, 0.0)

    def test_no_bets_is_not_a_crash(self):
        result = aggregate([], config())
        self.assertEqual(result.n_bets, 0)
        self.assertEqual(result.roi, 0.0)


if __name__ == "__main__":
    unittest.main()
