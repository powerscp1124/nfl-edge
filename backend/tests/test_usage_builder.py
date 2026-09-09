"""Usage builder tests. The shrinkage behaviour is the point."""

import unittest

from app.projections.usage_builder import (
    PlayerGameLog,
    build_player_usage,
    flag_role_changes,
    normalize_team_shares,
    opportunity_prior,
)


def logs(pid, pos, target_share, n=8, attempts=34, name=None):
    return [
        PlayerGameLog(
            player_id=pid, name=name or pid, position=pos, team="MIN", week=w,
            targets=round(target_share * attempts), receptions=round(
                target_share * attempts * 0.65),
            receiving_yards=round(target_share * attempts * 0.65 * 12.0, 1),
            air_yards=round(target_share * attempts * 10.0, 1),
            team_pass_attempts=attempts, team_rush_attempts=26,
            snap_share=0.75, team_gl_carries=3, team_ez_targets=4,
        )
        for w in range(1, n + 1)
    ]


class TestShrinkage(unittest.TestCase):
    def test_shrinkage_toward_zero_would_be_cancelled_by_renormalisation(self):
        """The bug this guards against.

        Shrinking every player's share toward zero with the same weight applies
        one common multiplicative factor, which team-level renormalisation
        cancels exactly -- leaving a two-game sample trusted as much as a
        ten-game one. Shrinking toward a depth baseline must survive
        renormalisation.
        """
        thin = build_player_usage(logs("thin", "WR", 0.32, n=2), depth_rank=3)
        thick = build_player_usage(logs("thick", "WR", 0.32, n=10), depth_rank=3)
        # Same observed share, different sample size -> different projection.
        self.assertLess(thin.usage.target_share, thick.usage.target_share)

    def test_thin_sample_is_pulled_toward_its_depth_prior(self):
        prior = opportunity_prior("WR", 3)["target_share"]
        thin = build_player_usage(logs("t", "WR", 0.40, n=2), depth_rank=3)
        thick = build_player_usage(logs("t", "WR", 0.40, n=12), depth_rank=3)
        self.assertLess(abs(thin.usage.target_share - prior),
                        abs(thick.usage.target_share - prior))

    def test_established_player_is_barely_moved(self):
        built = build_player_usage(logs("wr1", "WR", 0.28, n=14), depth_rank=1)
        self.assertAlmostEqual(built.usage.target_share, 0.28, delta=0.02)

    def test_depth_priors_are_ordered(self):
        self.assertGreater(opportunity_prior("WR", 1)["target_share"],
                           opportunity_prior("WR", 3)["target_share"])
        self.assertGreater(opportunity_prior("RB", 1)["rush_share"],
                           opportunity_prior("RB", 2)["rush_share"])

    def test_thin_sample_is_flagged_as_insufficient(self):
        built = build_player_usage(logs("x", "WR", 0.20, n=2), depth_rank=2)
        self.assertFalse(built.sufficient)
        self.assertTrue(built.notes)


class TestNormalisation(unittest.TestCase):
    def test_shares_sum_to_one(self):
        built = [
            build_player_usage(logs("wr1", "WR", 0.28), depth_rank=1).usage,
            build_player_usage(logs("wr2", "WR", 0.20), depth_rank=2).usage,
            build_player_usage(logs("te1", "TE", 0.18), depth_rank=1).usage,
            build_player_usage(logs("rb1", "RB", 0.12), depth_rank=1).usage,
        ]
        norm = normalize_team_shares(built)
        self.assertAlmostEqual(sum(u.target_share for u in norm), 1.0, places=6)

    def test_ordering_is_preserved(self):
        built = [
            build_player_usage(logs("wr1", "WR", 0.30), depth_rank=1).usage,
            build_player_usage(logs("wr2", "WR", 0.15), depth_rank=2).usage,
        ]
        norm = normalize_team_shares(built)
        self.assertGreater(norm[0].target_share, norm[1].target_share)


class TestRoleChange(unittest.TestCase):
    def test_stable_usage_is_not_flagged(self):
        self.assertIsNone(flag_role_changes(logs("wr1", "WR", 0.22, n=10)))

    def test_step_change_is_flagged(self):
        rows = logs("wr2", "WR", 0.15, n=6) + logs("wr2", "WR", 0.30, n=3)
        for i, r in enumerate(rows):
            r.week = i + 1
        change = flag_role_changes(rows)
        self.assertIsNotNone(change)
        self.assertGreater(change["recent"], change["baseline"])

    def test_short_history_returns_none(self):
        self.assertIsNone(flag_role_changes(logs("x", "WR", 0.2, n=3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
