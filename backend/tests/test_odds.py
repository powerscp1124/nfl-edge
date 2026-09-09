"""Odds math tests. Every expected value here is computed by hand."""

import unittest

from app.core.odds import (
    BookQuote,
    american_to_decimal,
    american_to_prob,
    breakeven_probability,
    consensus_line,
    decimal_to_american,
    devig,
    devig_two_way,
    expected_value,
    expected_roi,
    format_american,
    hold_percentage,
    kelly_fraction,
    overround,
    prob_to_american,
    probability_edge,
)


class TestConversions(unittest.TestCase):
    def test_american_to_decimal(self):
        self.assertAlmostEqual(american_to_decimal(100), 2.0)
        self.assertAlmostEqual(american_to_decimal(-100), 2.0)
        self.assertAlmostEqual(american_to_decimal(-110), 1.909090909, places=8)
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertAlmostEqual(american_to_decimal(-200), 1.5)

    def test_american_to_prob(self):
        # -110 -> 110/210
        self.assertAlmostEqual(american_to_prob(-110), 0.5238095238, places=8)
        # +145 -> 100/245
        self.assertAlmostEqual(american_to_prob(145), 0.4081632653, places=8)
        self.assertAlmostEqual(american_to_prob(-105), 0.5121951220, places=8)
        self.assertAlmostEqual(american_to_prob(100), 0.5)

    def test_prob_to_american(self):
        self.assertAlmostEqual(prob_to_american(0.5), -100.0)
        # 63% -> -100*0.63/0.37 = -170.27
        self.assertAlmostEqual(prob_to_american(0.63), -170.2702702, places=5)
        # 40% -> 100*0.6/0.4 = +150
        self.assertAlmostEqual(prob_to_american(0.40), 150.0)

    def test_roundtrip(self):
        # +100 and -100 are the same price, so the roundtrip is only unique
        # away from even money.
        for odds in (-450, -200, -110, -105, 120, 250, 900):
            p = american_to_prob(odds)
            self.assertAlmostEqual(prob_to_american(p), float(odds), places=6)
            d = american_to_decimal(odds)
            self.assertAlmostEqual(decimal_to_american(d), float(odds), places=6)
        self.assertAlmostEqual(american_to_decimal(prob_to_american(0.5)),
                               american_to_decimal(100), places=9)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            american_to_prob(0)
        with self.assertRaises(ValueError):
            prob_to_american(0.0)
        with self.assertRaises(ValueError):
            prob_to_american(1.0)

    def test_format(self):
        self.assertEqual(format_american(-170.27), "-170")
        self.assertEqual(format_american(145.4), "+145")
        self.assertEqual(format_american(100), "EVEN")


class TestVig(unittest.TestCase):
    def test_overround_and_hold(self):
        self.assertAlmostEqual(overround([-110, -110]), 1.0476190476, places=8)
        self.assertAlmostEqual(hold_percentage([-110, -110]), 0.0454545455, places=8)

    def test_symmetric_market_devigs_to_even(self):
        p_over, p_under = devig_two_way(-110, -110)
        self.assertAlmostEqual(p_over, 0.5, places=10)
        self.assertAlmostEqual(p_under, 0.5, places=10)

    def test_devig_sums_to_one(self):
        for method in ("multiplicative", "additive", "power", "shin"):
            with self.subTest(method=method):
                probs = devig([-130, +105], method=method)
                self.assertAlmostEqual(sum(probs), 1.0, places=9)
                self.assertTrue(all(0 < p < 1 for p in probs))

    def test_devig_preserves_ordering(self):
        # The favourite must stay the favourite under every method.
        for method in ("multiplicative", "additive", "power", "shin"):
            with self.subTest(method=method):
                p_fav, p_dog = devig([-250, +200], method=method)
                self.assertGreater(p_fav, p_dog)

    def test_power_discounts_longshots_more(self):
        # On a lopsided market the power method should assign the longshot a
        # lower fair probability than proportional de-vigging does.
        mult = devig([-400, +300], method="multiplicative")
        power = devig([-400, +300], method="power")
        self.assertLess(power[1], mult[1])

    def test_devig_requires_two_outcomes(self):
        with self.assertRaises(ValueError):
            devig([-110])

    def test_multiway_market(self):
        probs = devig([+300, +400, +500, +700], method="multiplicative")
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertEqual(len(probs), 4)


class TestExpectedValue(unittest.TestCase):
    def test_ev_hand_computed(self):
        # 63% at -105: decimal 1.952381, profit 0.952381
        # EV = 0.63 * 0.952381 - 0.37 = 0.6 - 0.37 = 0.230
        ev = expected_value(0.63, -105)
        self.assertAlmostEqual(ev, 0.2300, places=4)

    def test_ev_zero_at_breakeven(self):
        for odds in (-250, -110, 100, 175, 400):
            p = breakeven_probability(odds)
            self.assertAlmostEqual(expected_value(p, odds), 0.0, places=12)

    def test_ev_scales_with_stake(self):
        self.assertAlmostEqual(
            expected_value(0.60, -110, stake=100),
            expected_value(0.60, -110, stake=1) * 100,
            places=9,
        )

    def test_negative_ev(self):
        self.assertLess(expected_value(0.45, -110), 0)

    def test_roi_matches_unit_ev(self):
        self.assertAlmostEqual(expected_roi(0.58, 120),
                               expected_value(0.58, 120, 1.0), places=12)

    def test_probability_edge(self):
        self.assertAlmostEqual(probability_edge(0.638, 0.512), 0.126, places=9)


class TestKelly(unittest.TestCase):
    def test_no_stake_without_edge(self):
        self.assertEqual(kelly_fraction(0.45, -110), 0.0)
        self.assertEqual(kelly_fraction(0.50, -110), 0.0)

    def test_quarter_kelly_hand_computed(self):
        # p=0.60, b=1.0 (even money): full Kelly = (0.6*1 - 0.4)/1 = 0.20
        # quarter Kelly = 0.05, but the 2% cap binds.
        self.assertAlmostEqual(kelly_fraction(0.60, 100, fraction=0.25, cap=1.0),
                               0.05, places=10)
        self.assertAlmostEqual(kelly_fraction(0.60, 100), 0.02, places=10)

    def test_cap_is_enforced(self):
        self.assertLessEqual(kelly_fraction(0.95, 500), 0.02)

    def test_full_kelly_is_never_the_default(self):
        p, odds = 0.60, 100
        self.assertLess(kelly_fraction(p, odds, cap=1.0),
                        (p * 1.0 - (1 - p)) / 1.0)


class TestLineShopping(unittest.TestCase):
    def test_consensus(self):
        quotes = [
            BookQuote("DraftKings", 84.5, -105),
            BookQuote("FanDuel", 85.5, -110),
            BookQuote("BetMGM", 84.5, -115),
        ]
        c = consensus_line(quotes)
        self.assertAlmostEqual(c["median"], 84.5)
        self.assertAlmostEqual(c["mode"], 84.5)
        self.assertEqual(c["n_books"], 3)

    def test_better_number_can_beat_better_price(self):
        """The core line-shopping claim from the spec.

        Over 84.5 at -105 should beat Over 85.5 at -125 for a model that
        projects the middle of that range, even though neither dominates on
        price and number simultaneously.
        """
        from app.core.distributions import ShiftedGamma
        from app.core.odds import best_quote

        dist = ShiftedGamma(mean_value=92.0, std_value=34.0)
        quotes = [
            BookQuote("DraftKings", 84.5, -105),
            BookQuote("Caesars", 85.5, -125),
        ]
        best = best_quote(quotes, "over", dist.cdf)
        self.assertEqual(best.bookmaker, "DraftKings")


if __name__ == "__main__":
    unittest.main(verbosity=2)
