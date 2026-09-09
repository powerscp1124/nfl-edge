"""Distribution and simulator tests."""

import unittest

import numpy as np

from app.core.distributions import (
    EmpiricalDistribution,
    ShiftedGamma,
    ZeroInflatedGamma,
    implied_team_totals,
    negative_binomial_td,
    poisson_binomial_pmf,
    td_probabilities,
)


class TestEmpirical(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.dist = EmpiricalDistribution(rng.gamma(3.0, 25.0, 40_000))

    def test_prob_over_matches_counts(self):
        p = self.dist.prob_over(75.0)
        manual = np.count_nonzero(self.dist.samples > 75.0) / self.dist.n
        self.assertAlmostEqual(p, manual, places=12)

    def test_over_and_under_sum_to_one(self):
        self.assertAlmostEqual(
            self.dist.prob_over(83.5) + self.dist.prob_under(83.5), 1.0, places=12
        )

    def test_quantiles_are_ordered(self):
        s = self.dist.summary()
        self.assertLess(s["p10"], s["p25"])
        self.assertLess(s["p25"], s["p50"])
        self.assertLess(s["p50"], s["p75"])
        self.assertLess(s["p75"], s["p90"])

    def test_right_skew_puts_mean_above_median(self):
        s = self.dist.summary()
        self.assertGreater(s["mean"], s["median"])

    def test_prob_over_is_monotone_decreasing(self):
        probs = [self.dist.prob_over(t) for t in range(20, 200, 10)]
        self.assertTrue(all(a >= b for a, b in zip(probs, probs[1:])))

    def test_pushes_are_excluded(self):
        d = EmpiricalDistribution(np.array([1.0, 2.0, 2.0, 3.0]))
        # Two samples push on an integer line of 2; of the live two, one is over.
        self.assertAlmostEqual(d.prob_over(2.0), 0.5)

    def test_monte_carlo_error_shrinks_with_sims(self):
        rng = np.random.default_rng(1)
        small = EmpiricalDistribution(rng.gamma(3.0, 25.0, 1_000))
        big = EmpiricalDistribution(rng.gamma(3.0, 25.0, 100_000))
        self.assertGreater(small.monte_carlo_error(75.0),
                           big.monte_carlo_error(75.0))

    def test_threshold_curve(self):
        curve = self.dist.threshold_curve([49.5, 59.5, 69.5, 79.5])
        self.assertEqual(len(curve), 4)
        values = [c["prob_over"] for c in curve]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


class TestParametric(unittest.TestCase):
    def test_zero_inflated_gamma_never_goes_negative(self):
        d = ZeroInflatedGamma(p_zero=0.08, mean_positive=65.0, cv=0.8)
        self.assertEqual(d.cdf(-1.0), 0.0)
        self.assertGreaterEqual(d.quantile(0.01), 0.0)

    def test_zero_inflation_shows_up_in_the_cdf(self):
        d = ZeroInflatedGamma(p_zero=0.15, mean_positive=50.0, cv=0.9)
        self.assertGreaterEqual(d.cdf(0.0), 0.15)

    def test_from_moments_recovers_mean(self):
        d = ZeroInflatedGamma.from_moments(mean=62.0, std=38.0, p_zero=0.06)
        self.assertAlmostEqual(d.mean(), 62.0, places=6)

    def test_sampling_matches_analytic_prob(self):
        d = ZeroInflatedGamma(p_zero=0.05, mean_positive=70.0, cv=0.85)
        samples = d.sample(200_000, np.random.default_rng(3))
        self.assertAlmostEqual(np.mean(samples > 79.5), d.prob_over(79.5), places=2)

    def test_shifted_gamma_moments(self):
        d = ShiftedGamma(mean_value=268.0, std_value=62.0)
        self.assertAlmostEqual(d.mean(), 268.0, places=6)
        self.assertAlmostEqual(d.std(), 62.0, places=6)
        # Mildly right-skewed, so P(over the mean) sits just under a half.
        self.assertLess(d.prob_over(268.0), 0.50)
        self.assertGreater(d.prob_over(268.0), 0.44)

    def test_normal_would_misprice_a_low_line(self):
        """Why the family choice matters, not just the mean and sd.

        A WR projected for 45 yards with a 38-yard sd gets ~9.4% of its mass
        below zero under a normal. The zero-inflated gamma does not, and the
        two disagree materially on a live line.
        """
        from scipy import stats

        mean, sd, line = 45.0, 38.0, 29.5
        normal_p = float(stats.norm(mean, sd).sf(line))
        gamma_p = ZeroInflatedGamma.from_moments(mean, sd, p_zero=0.04).prob_over(line)
        self.assertGreater(abs(normal_p - gamma_p), 0.02)
        self.assertGreater(stats.norm(mean, sd).cdf(0.0), 0.05)


class TestTouchdownCounts(unittest.TestCase):
    def test_poisson_binomial_sums_to_one(self):
        pmf = poisson_binomial_pmf([0.12, 0.09, 0.30, 0.05])
        self.assertAlmostEqual(pmf.sum(), 1.0, places=12)

    def test_poisson_binomial_matches_hand_calc(self):
        # Two opportunities at 0.2 and 0.5:
        # P(0) = 0.8*0.5 = 0.40, P(2) = 0.2*0.5 = 0.10, P(1) = 0.50
        pmf = poisson_binomial_pmf([0.2, 0.5])
        self.assertAlmostEqual(pmf[0], 0.40, places=12)
        self.assertAlmostEqual(pmf[1], 0.50, places=12)
        self.assertAlmostEqual(pmf[2], 0.10, places=12)

    def test_matches_binomial_when_probs_are_equal(self):
        from scipy import stats

        pmf = poisson_binomial_pmf([0.25] * 6)
        binom = stats.binom(6, 0.25).pmf(np.arange(7))
        np.testing.assert_allclose(pmf, binom, atol=1e-12)

    def test_derived_probabilities_are_consistent(self):
        out = td_probabilities(poisson_binomial_pmf([0.18, 0.22, 0.10, 0.06]))
        self.assertAlmostEqual(out["anytime"], 1.0 - out["p0"], places=12)
        self.assertAlmostEqual(out["two_plus"],
                               1.0 - out["p0"] - out["p1"], places=12)
        self.assertLess(out["two_plus"], out["anytime"])

    def test_overdispersion_raises_two_plus(self):
        """Poisson understates P(2+), which is the market it matters most for."""
        mean = 0.7
        poisson = td_probabilities(negative_binomial_td(mean, dispersion=1.0))
        nb = td_probabilities(negative_binomial_td(mean, dispersion=1.8))
        self.assertGreater(nb["two_plus"], poisson["two_plus"])

    def test_implied_team_totals(self):
        home, away = implied_team_totals(spread=-3.5, total=47.5)
        self.assertAlmostEqual(home, 25.5)
        self.assertAlmostEqual(away, 22.0)
        self.assertAlmostEqual(home + away, 47.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
