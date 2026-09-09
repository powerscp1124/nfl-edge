"""Line-anchoring tests.

The point of anchoring is that disagreement with the market has to be argued.
These check that an unargued projection agrees with the line, that a deviation
cannot move it without naming a reason, and that recentring does not quietly
destroy the simulated shape.
"""

import unittest

import numpy as np

from app.core.distributions import EmpiricalDistribution
from app.projections.anchor import (
    DEFAULT_ANCHOR_WEIGHT,
    Deviation,
    LineAnchor,
    fit_anchor_weight,
)


class TestCentre(unittest.TestCase):
    def test_with_nothing_to_add_the_projection_is_the_line(self):
        """A model that has earned no disagreement should not manufacture one."""
        anchor = LineAnchor(weight=1.0)
        self.assertAlmostEqual(anchor.centre(80.0, 40.0), 80.0)

    def test_the_default_sits_close_to_the_line(self):
        anchor = LineAnchor()
        centre = anchor.centre(80.0, 40.0)
        self.assertGreater(centre, 70.0)
        self.assertLess(centre, 80.0)

    def test_a_deviation_moves_it_off_the_line(self):
        anchor = LineAnchor(weight=1.0)
        up = anchor.centre(80.0, 80.0, [Deviation("role change", 0.10)])
        self.assertAlmostEqual(up, 88.0, places=6)

    def test_deviations_accumulate_but_are_capped(self):
        """A signal wanting to move a line 60% is a mapping error, not insight."""
        anchor = LineAnchor(weight=1.0, max_total_deviation=0.25)
        wild = anchor.centre(100.0, 100.0, [Deviation("a", 0.4),
                                            Deviation("b", 0.4)])
        self.assertAlmostEqual(wild, 125.0, places=6)

    def test_a_deviation_must_name_a_reason(self):
        with self.assertRaises(ValueError):
            Deviation("", 0.05)

    def test_a_missing_line_falls_back_to_the_model(self):
        anchor = LineAnchor()
        self.assertAlmostEqual(anchor.centre(float("nan"), 42.0), 42.0)

    def test_it_never_returns_a_negative_projection(self):
        anchor = LineAnchor(weight=1.0, max_total_deviation=2.0)
        self.assertGreaterEqual(
            anchor.centre(10.0, 10.0, [Deviation("x", -3.0)]), 0.0)


class TestApplyToDistribution(unittest.TestCase):
    def dist(self, mean=40.0, n=20000, seed=3):
        rng = np.random.default_rng(seed)
        return EmpiricalDistribution(rng.gamma(2.0, mean / 2.0, size=n),
                                     stat="receiving_yards")

    def test_recentring_puts_the_line_at_the_median(self):
        """A book sets its line where about half the outcomes fall either
        side, so the line is a median. Anchoring the mean instead drags the
        median below the line and invents a lean toward the under."""
        anchor = LineAnchor(weight=1.0)
        moved = anchor.apply(self.dist(mean=40.0), line=60.0)
        self.assertAlmostEqual(moved.quantile(0.5), 60.0, delta=0.5)
        self.assertAlmostEqual(moved.prob_over(60.0), 0.5, delta=0.02)

    def test_it_keeps_the_simulated_shape(self):
        """Rescaling preserves the coefficient of variation; a shift would
        not, and could push yardage below zero."""
        original = self.dist(mean=40.0)
        moved = LineAnchor(weight=1.0).apply(original, line=60.0)
        self.assertAlmostEqual(moved.std() / moved.mean(),
                               original.std() / original.mean(), places=6)
        self.assertGreater(moved.mean(), moved.quantile(0.5))   # still skewed
        self.assertGreaterEqual(float(moved.samples.min()), 0.0)

    def test_an_empty_or_zero_projection_is_left_alone(self):
        zero = EmpiricalDistribution(np.zeros(100), stat="receiving_yards")
        self.assertIs(LineAnchor().apply(zero, line=50.0), zero)

    def test_probabilities_move_toward_the_market(self):
        """The whole point: an anchored model stops claiming a large edge."""
        anchor = LineAnchor(weight=1.0)
        d = self.dist(mean=20.0)
        raw = d.prob_over(60.0)
        moved = anchor.apply(d, line=60.0).prob_over(60.0)
        self.assertLess(raw, 0.15)
        self.assertAlmostEqual(moved, 0.5, delta=0.03)


class TestFitting(unittest.TestCase):
    def test_a_useless_model_fits_full_weight_on_the_line(self):
        rng = np.random.default_rng(5)
        actual = rng.normal(50, 20, 2000)
        line = actual + rng.normal(0, 8, 2000)
        projection = rng.normal(50, 20, 2000)          # pure noise
        self.assertGreater(fit_anchor_weight(line, projection, actual), 0.85)

    def test_a_perfect_model_fits_no_weight_on_the_line(self):
        rng = np.random.default_rng(5)
        actual = rng.normal(50, 20, 2000)
        line = actual + rng.normal(0, 25, 2000)
        projection = actual + rng.normal(0, 1, 2000)   # near perfect
        self.assertLess(fit_anchor_weight(line, projection, actual), 0.2)

    def test_a_thin_sample_returns_the_default(self):
        self.assertEqual(fit_anchor_weight([1, 2], [1, 2], [1, 2]),
                         DEFAULT_ANCHOR_WEIGHT)


if __name__ == "__main__":
    unittest.main()
