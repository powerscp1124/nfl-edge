"""Calibration tests.

A calibration layer is only worth having if it is honest about thin data and
never reorders the bets it is handed.
"""

import unittest

import numpy as np

from app.core.calibration import (
    ProbabilityCalibrator,
    brier,
    logloss,
)


def overconfident_sample(n=4000, retained=0.3, seed=7):
    """Probabilities stretched away from 50% relative to what happens.

    The true probability is ``0.5 + retained * (stated - 0.5)``, which is the
    shape the 2025 backtest showed: a stated 63.5% arriving 54% of the time.
    """
    rng = np.random.default_rng(seed)
    stated = rng.uniform(0.52, 0.80, size=n)
    true_p = 0.5 + retained * (stated - 0.5)
    outcomes = (rng.random(n) < true_p).astype(float)
    return stated, outcomes


class TestFitting(unittest.TestCase):
    def test_a_calibrated_model_fits_near_the_identity(self):
        rng = np.random.default_rng(3)
        stated = rng.uniform(0.3, 0.7, size=6000)
        outcomes = (rng.random(6000) < stated).astype(float)
        cal = ProbabilityCalibrator.fit(stated, outcomes)
        self.assertAlmostEqual(cal.a, 1.0, delta=0.25)
        self.assertAlmostEqual(cal.b, 0.0, delta=0.15)

    def test_an_overconfident_model_fits_a_slope_below_one(self):
        stated, outcomes = overconfident_sample()
        cal = ProbabilityCalibrator.fit(stated, outcomes)
        self.assertLess(cal.a, 0.8)

    def test_a_thin_sample_returns_the_identity(self):
        """Fitting on forty bets would replace one unmeasured error with
        another."""
        stated, outcomes = overconfident_sample(n=40)
        cal = ProbabilityCalibrator.fit(stated, outcomes)
        self.assertTrue(cal.is_identity)

    def test_a_single_outcome_class_returns_the_identity(self):
        cal = ProbabilityCalibrator.fit([0.6] * 100, [1.0] * 100)
        self.assertTrue(cal.is_identity)


class TestApplication(unittest.TestCase):
    def test_the_identity_changes_nothing(self):
        cal = ProbabilityCalibrator.identity()
        for p in (0.1, 0.5, 0.73, 0.99):
            self.assertAlmostEqual(cal.apply(p), p, places=6)

    def test_a_pure_shrink_leaves_a_coin_flip_alone(self):
        cal = ProbabilityCalibrator(a=0.3, b=0.0)
        self.assertAlmostEqual(cal.apply(0.5), 0.5, places=6)

    def test_shrinking_pulls_probabilities_toward_a_half(self):
        cal = ProbabilityCalibrator(a=0.3, b=0.0)
        self.assertLess(cal.apply(0.8), 0.8)
        self.assertGreater(cal.apply(0.2), 0.2)

    def test_it_never_reorders_two_bets(self):
        """Monotone, so calibration can change how much you bet but never
        which of two bets you prefer."""
        cal = ProbabilityCalibrator(a=0.42, b=-0.05)
        probs = np.linspace(0.02, 0.98, 200)
        out = cal.apply(probs)
        self.assertTrue(np.all(np.diff(out) > 0))

    def test_confidence_retained_is_readable(self):
        cal = ProbabilityCalibrator(a=1.0, b=0.0)
        self.assertAlmostEqual(cal.confidence_retained, 1.0, places=6)


class TestItActuallyImprovesScores(unittest.TestCase):
    """The point of the exercise, measured out of sample."""

    def test_held_out_brier_and_log_loss_improve(self):
        stated, outcomes = overconfident_sample(n=8000)
        train = slice(0, 4000)
        test = slice(4000, 8000)
        cal = ProbabilityCalibrator.fit(stated[train], outcomes[train])
        raw_b = brier(stated[test], outcomes[test])
        raw_l = logloss(stated[test], outcomes[test])
        cal_b = brier(cal.apply(stated[test]), outcomes[test])
        cal_l = logloss(cal.apply(stated[test]), outcomes[test])
        self.assertLess(cal_b, raw_b)
        self.assertLess(cal_l, raw_l)

    def test_it_beats_the_coin_flip_it_was_losing_to(self):
        """The 2025 backtest scored worse than predicting 0.5 on everything."""
        stated, outcomes = overconfident_sample(n=8000)
        cal = ProbabilityCalibrator.fit(stated[:4000], outcomes[:4000])
        coin = np.full(4000, 0.5)
        self.assertLess(logloss(cal.apply(stated[4000:]), outcomes[4000:]),
                        logloss(coin, outcomes[4000:]))


if __name__ == "__main__":
    unittest.main()
