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


class TestParticipationWeightsThePrior(unittest.TestCase):
    """Prior mass handed to players who do not play is taken from those who do."""

    def log(self, pid, week, targets, snap):
        from app.projections.usage_builder import PlayerGameLog
        return PlayerGameLog(player_id=pid, name=pid, position="WR", team="SEA",
                             week=week, targets=targets, team_pass_attempts=35,
                             receptions=targets * 0.6,
                             receiving_yards=targets * 7.0, snap_share=snap)

    def test_a_fringe_player_carries_less_prior_than_a_starter(self):
        from app.projections.usage_builder import build_player_usage
        fringe = build_player_usage([self.log("f", 1, 0, 0.05)], depth_rank=3)
        regular = build_player_usage([self.log("r", 1, 0, 0.85)], depth_rank=3)
        self.assertLess(fringe.usage.target_share, regular.usage.target_share)

    def test_unknown_participation_keeps_the_full_prior(self):
        """Missing snap data must not be read as a player who never played."""
        from app.projections.usage_builder import build_player_usage
        unknown = build_player_usage([self.log("u", 1, 0, None)], depth_rank=3)
        played = build_player_usage([self.log("p", 1, 0, 0.85)], depth_rank=3)
        self.assertAlmostEqual(unknown.usage.target_share,
                               played.usage.target_share, places=6)

    def test_bench_players_no_longer_tax_the_starter(self):
        """The bug: 16 players each carrying a prior floor sum well above 1.0."""
        from app.projections.usage_builder import (build_player_usage,
                                                   normalize_team_shares)
        starter = build_player_usage([self.log("s", w, 11, 0.85)
                                      for w in range(1, 8)], depth_rank=1).usage
        bench = [build_player_usage([self.log(f"b{i}", 7, 0, 0.05)],
                                    depth_rank=3).usage for i in range(8)]
        normalised = normalize_team_shares([starter] + bench)
        kept = next(u for u in normalised if u.player_id == "s")
        self.assertGreater(kept.target_share, 0.75 * starter.target_share)


class TestRosterPriorsFitInsideOneTeam(unittest.TestCase):
    """Per-player priors are not a budget until someone makes them one."""

    def test_a_normal_roster_is_left_alone(self):
        from app.projections.usage_builder import roster_prior_scale
        lean = [("WR", 1), ("WR", 2), ("RB", 1), ("TE", 1)]
        self.assertLessEqual(roster_prior_scale(lean), 1.0)
        self.assertGreater(roster_prior_scale(lean), 0.5)

    def test_a_deep_roster_is_scaled_down(self):
        from app.projections.usage_builder import roster_prior_scale
        deep = [("WR", 1), ("WR", 2), ("RB", 1), ("TE", 1)] + [
            ("WR", 3)] * 12
        scale = roster_prior_scale(deep)
        self.assertLess(scale, 0.7)
        self.assertGreater(scale, 0.0)

    def test_scaling_makes_the_blended_pool_land_near_one(self):
        """The bug: a dozen one-game fringe players, each pulled toward a
        plausible individual share, produce a team throwing more passes than
        it has."""
        from app.projections.usage_builder import (PlayerGameLog,
                                                   build_player_usage,
                                                   roster_prior_scale)

        def log(pid, week, targets, snap):
            return PlayerGameLog(player_id=pid, name=pid, position="WR",
                                 team="SEA", week=week, targets=targets,
                                 team_pass_attempts=35,
                                 receptions=targets * 0.6,
                                 receiving_yards=targets * 7.0,
                                 snap_share=snap)

        roster = [("WR", 1), ("WR", 2)] + [("WR", 3)] * 12
        scale = roster_prior_scale(roster)

        def pool(sc):
            starters = sum(
                build_player_usage([log(pid, w, tg, 0.85)
                                    for w in range(1, 8)],
                                   depth_rank=rank, prior_scale=sc
                                   ).usage.target_share
                for pid, tg, rank in (("s1", 11, 1), ("s2", 7, 2)))
            # One appearance each, participation unknown: the case that keeps
            # the full prior, and the one a real lookback window is full of.
            bench = sum(
                build_player_usage([log(f"b{i}", 7, 0, None)],
                                   depth_rank=3, prior_scale=sc
                                   ).usage.target_share
                for i in range(12))
            return starters + bench

        self.assertGreater(pool(1.0), 1.0)
        self.assertLess(abs(pool(scale) - 1.0), abs(pool(1.0) - 1.0))


class TestAvailability(unittest.TestCase):
    """Opportunity is renormalised across whoever is on the field, so a player
    carried at full availability who does not dress takes his whole share out
    of the game. Measured over 2025 that was 20% of a team's targets."""

    def log(self, week, targets=6):
        from app.projections.usage_builder import PlayerGameLog
        return PlayerGameLog(player_id="p", name="p", position="WR",
                             team="SEA", week=week, targets=targets,
                             team_pass_attempts=35, receptions=4,
                             receiving_yards=50.0, snap_share=0.8)

    def build(self, weeks, team_weeks):
        from app.projections.usage_builder import build_player_usage
        return build_player_usage([self.log(w) for w in weeks],
                                  depth_rank=1,
                                  team_weeks=team_weeks).usage

    def test_an_ever_present_player_is_fully_available(self):
        u = self.build([5, 6, 7, 8], [5, 6, 7, 8])
        self.assertAlmostEqual(u.active_probability, 1.0, places=6)

    def test_a_long_absent_player_is_nearly_unavailable(self):
        from app.projections.usage_builder import MIN_AVAILABILITY
        u = self.build([1], [5, 6, 7, 8])
        self.assertAlmostEqual(u.active_probability, MIN_AVAILABILITY,
                               places=6)

    def test_it_is_never_exactly_zero(self):
        """A blanket zero makes the player's line a guaranteed under."""
        u = self.build([1], [5, 6, 7, 8])
        self.assertGreater(u.active_probability, 0.0)

    def test_recent_appearances_count_for_more(self):
        just_back = self.build([8], [5, 6, 7, 8])
        long_gone = self.build([5], [5, 6, 7, 8])
        self.assertGreater(just_back.active_probability,
                           long_gone.active_probability)

    def test_without_team_weeks_nothing_changes(self):
        u = self.build([5, 6], None)
        self.assertAlmostEqual(u.active_probability, 1.0, places=6)

    def test_injury_status_composes_with_availability(self):
        """Dressing rate and an injury report are independent reasons to sit;
        overwriting one with the other loses a real signal."""
        from app.projections.injury_engine import RoleProfile, propagate_injuries
        from app.projections.usage_builder import normalize_team_shares
        rotational = self.build([7, 8], [5, 6, 7, 8])
        self.assertLess(rotational.active_probability, 1.0)
        roles = {"p": RoleProfile(player_id="p", position="WR", depth_rank=1)}
        result = propagate_injuries(
            normalize_team_shares([rotational]), roles,
            {"p": {"report_status": "Questionable",
                   "practice_status": "Limited Participation in Practice"}})
        after = next(u for u in result.usages if u.player_id == "p")
        self.assertLess(after.active_probability,
                        rotational.active_probability)
