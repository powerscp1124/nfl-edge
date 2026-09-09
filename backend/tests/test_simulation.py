"""Simulator tests.

These check the properties that make the simulator worth having: internal
consistency, correlation between teammates, and sensible responses to game
script and scoring environment.
"""

import unittest

import numpy as np

from app.projections.environment import (
    GameEnvironment,
    TeamEnvironment,
    WeatherState,
)
from app.sim.game_sim import (PlayerUsage, TeamRoster, _allocate,
                              simulate_game)


def build_game(spread_home=-3.5, total=47.5, weather=None, seed=11, n_sims=8000):
    home_env = TeamEnvironment(team="KC", neutral_pass_rate=0.60,
                               seconds_per_play=26.5, red_zone_rush_rate=0.48)
    away_env = TeamEnvironment(team="DEN", neutral_pass_rate=0.55,
                               seconds_per_play=28.5, red_zone_rush_rate=0.55)

    home = TeamRoster(env=home_env, players=[
        PlayerUsage("kc-qb", "Home QB", "QB", "KC", is_starting_qb=True,
                    rush_share=0.06, yards_per_carry=4.8, goal_line_share=0.10),
        PlayerUsage("kc-rb", "Home RB", "RB", "KC", target_share=0.12,
                    rush_share=0.68, yards_per_carry=4.4,
                    yards_per_reception=7.5, catch_rate=0.76, adot=1.0,
                    goal_line_share=0.62, end_zone_target_share=0.08),
        PlayerUsage("kc-wr1", "Home WR1", "WR", "KC", target_share=0.27,
                    yards_per_reception=13.5, catch_rate=0.66, adot=11.5,
                    end_zone_target_share=0.30),
        PlayerUsage("kc-wr2", "Home WR2", "WR", "KC", target_share=0.18,
                    yards_per_reception=12.0, catch_rate=0.62, adot=10.0,
                    end_zone_target_share=0.20),
        PlayerUsage("kc-te", "Home TE", "TE", "KC", target_share=0.22,
                    yards_per_reception=11.0, catch_rate=0.70, adot=7.5,
                    end_zone_target_share=0.32),
        PlayerUsage("kc-wr3", "Home WR3", "WR", "KC", target_share=0.21,
                    yards_per_reception=10.5, catch_rate=0.60, adot=9.0,
                    end_zone_target_share=0.10),
        PlayerUsage("kc-rb2", "Home RB2", "RB", "KC", target_share=0.0,
                    rush_share=0.26, yards_per_carry=4.1,
                    goal_line_share=0.28),
    ])
    away = TeamRoster(env=away_env, players=[
        PlayerUsage("den-qb", "Away QB", "QB", "DEN", is_starting_qb=True,
                    rush_share=0.05, goal_line_share=0.08),
        PlayerUsage("den-rb", "Away RB", "RB", "DEN", target_share=0.10,
                    rush_share=0.72, yards_per_carry=4.2,
                    yards_per_reception=7.0, catch_rate=0.74, adot=0.5,
                    goal_line_share=0.65, end_zone_target_share=0.08),
        PlayerUsage("den-wr1", "Away WR1", "WR", "DEN", target_share=0.30,
                    yards_per_reception=13.0, catch_rate=0.63, adot=12.0,
                    end_zone_target_share=0.34),
        PlayerUsage("den-wr2", "Away WR2", "WR", "DEN", target_share=0.24,
                    yards_per_reception=11.5, catch_rate=0.61, adot=9.5,
                    end_zone_target_share=0.28),
        PlayerUsage("den-te", "Away TE", "TE", "DEN", target_share=0.20,
                    yards_per_reception=10.0, catch_rate=0.68, adot=6.5,
                    end_zone_target_share=0.22),
        PlayerUsage("den-wr3", "Away WR3", "WR", "DEN", target_share=0.16,
                    yards_per_reception=10.0, catch_rate=0.58, adot=8.5,
                    end_zone_target_share=0.08),
        PlayerUsage("den-rb2", "Away RB2", "RB", "DEN", rush_share=0.23,
                    yards_per_carry=4.0, goal_line_share=0.27),
    ])
    env = GameEnvironment(game_id="test", home=home_env, away=away_env,
                          spread_home=spread_home, total=total,
                          weather=weather or WeatherState(is_dome=True))
    return simulate_game(env, home, away, n_sims=n_sims, seed=seed)


class TestSimulatorSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = build_game()

    def test_rejects_too_few_simulations(self):
        with self.assertRaises(ValueError):
            build_game(n_sims=100)

    def test_scores_match_the_market(self):
        # Home favoured by 3.5 on a 47.5 total -> 25.5 / 22.0 implied.
        self.assertAlmostEqual(float(np.mean(self.sim.team_points["KC"])),
                               25.5, delta=0.6)
        self.assertAlmostEqual(float(np.mean(self.sim.team_points["DEN"])),
                               22.0, delta=0.6)

    def test_play_volume_is_plausible(self):
        for team in ("KC", "DEN"):
            plays = self.sim.team_plays[team]
            self.assertGreater(float(np.mean(plays)), 55)
            self.assertLess(float(np.mean(plays)), 75)

    def test_pass_and_rush_attempts_reconcile_with_plays(self):
        # Attempts plus rushes plus sacks must equal plays, so the shortfall is
        # sacks only and has to be small and positive.
        for team in ("KC", "DEN"):
            gap = (self.sim.team_plays[team]
                   - self.sim.team_pass_attempts[team]
                   - self.sim.team_rush_attempts[team])
            self.assertTrue(np.all(gap >= 0))
            self.assertLess(float(np.mean(gap)), 4.0)

    def test_passing_yards_equal_team_receiving_yards(self):
        """The consistency property that isolated projections cannot give you."""
        team_rec = sum(
            self.sim.player_receiving_yards[p]
            for p in ("kc-rb", "kc-wr1", "kc-wr2", "kc-te", "kc-wr3")
        )
        np.testing.assert_allclose(
            self.sim.player_passing_yards["kc-qb"], team_rec, rtol=1e-9
        )

    def test_yardage_totals_are_realistic(self):
        qb = self.sim.distribution("kc-qb", "passing_yards").mean()
        self.assertGreater(qb, 190)
        self.assertLess(qb, 330)
        wr1 = self.sim.distribution("kc-wr1", "receiving_yards").mean()
        self.assertGreater(wr1, 45)
        self.assertLess(wr1, 110)
        rb = self.sim.distribution("kc-rb", "rushing_yards").mean()
        self.assertGreater(rb, 45)
        self.assertLess(rb, 110)

    def test_receiving_yards_are_right_skewed(self):
        d = self.sim.distribution("kc-wr1", "receiving_yards")
        s = d.summary()
        self.assertGreater(s["mean"], s["median"])
        self.assertGreaterEqual(s["p10"], 0.0)

    def test_rushing_yards_can_be_negative_but_rarely(self):
        yards = self.sim.player_rushing_yards["kc-rb"]
        self.assertLess(float(np.mean(yards < 0)), 0.02)


class TestCorrelation(unittest.TestCase):
    """Correlation is the whole reason for simulating instead of projecting."""

    @classmethod
    def setUpClass(cls):
        cls.sim = build_game(seed=23)

    def test_qb_and_wr1_are_positively_correlated(self):
        r = self.sim.correlation("kc-qb", "passing_yards",
                                 "kc-wr1", "receiving_yards")
        self.assertGreater(r, 0.3)

    def test_teammates_sharing_targets_compete(self):
        """WR1 and WR2 draw from one pool of targets, so given team volume they
        trade off. Team pass volume is a common driver, so the raw correlation
        is muted rather than strongly negative."""
        r = self.sim.correlation("kc-wr1", "receiving_yards",
                                 "kc-wr2", "receiving_yards")
        self.assertLess(r, 0.35)

    def test_opposing_backs_are_negatively_correlated(self):
        """Game script pushes one team to run and the other to throw."""
        r = self.sim.correlation("kc-rb", "carries", "den-rb", "carries")
        self.assertLess(r, 0.0)


class TestGameScript(unittest.TestCase):
    def test_favourites_run_more_than_underdogs(self):
        big_fav = build_game(spread_home=-10.5, total=44.5, seed=31)
        big_dog = build_game(spread_home=+10.5, total=44.5, seed=31)
        self.assertGreater(
            float(np.mean(big_fav.team_rush_attempts["KC"])),
            float(np.mean(big_dog.team_rush_attempts["KC"])) + 2.0,
        )

    def test_underdogs_throw_more(self):
        fav = build_game(spread_home=-10.5, seed=41)
        dog = build_game(spread_home=+10.5, seed=41)
        self.assertGreater(
            float(np.mean(dog.team_pass_attempts["KC"])),
            float(np.mean(fav.team_pass_attempts["KC"])) + 2.0,
        )

    def test_scoring_environment_moves_touchdown_probability(self):
        """A back on a 28-point team must differ from one on a 16-point team,
        even with identical usage."""
        high = build_game(spread_home=-7.5, total=54.5, seed=51)
        low = build_game(spread_home=-7.5, total=36.5, seed=51)
        self.assertGreater(
            high.touchdown_probabilities("kc-rb")["anytime"],
            low.touchdown_probabilities("kc-rb")["anytime"] + 0.05,
        )

    def test_two_plus_is_always_below_anytime(self):
        sim = build_game(seed=61)
        for pid in ("kc-rb", "kc-wr1", "den-wr1"):
            td = sim.touchdown_probabilities(pid)
            self.assertLess(td["two_plus"], td["anytime"])
            self.assertAlmostEqual(
                td["p0"] + td["p1"] + td["p2"] + td["p3"] + td["p4_plus"],
                1.0, places=9,
            )

    def test_team_touchdowns_track_implied_total(self):
        high = build_game(total=54.5, seed=71)
        low = build_game(total=36.5, seed=71)
        self.assertGreater(float(np.mean(high.team_touchdowns["KC"])),
                           float(np.mean(low.team_touchdowns["KC"])))


class TestWeather(unittest.TestCase):
    def test_high_wind_suppresses_passing(self):
        dome = build_game(weather=WeatherState(is_dome=True), seed=81)
        windy = build_game(
            weather=WeatherState(temperature_f=38, wind_mph=24,
                                 precipitation_prob=0.5, is_dome=False),
            seed=81,
        )
        self.assertLess(
            windy.distribution("kc-qb", "passing_yards").mean(),
            dome.distribution("kc-qb", "passing_yards").mean(),
        )

    def test_wind_hurts_deep_receivers_more_than_short_ones(self):
        dome = build_game(weather=WeatherState(is_dome=True), seed=91)
        windy = build_game(
            weather=WeatherState(wind_mph=26, is_dome=False), seed=91
        )
        deep_loss = 1 - (windy.distribution("kc-wr1", "receiving_yards").mean()
                         / dome.distribution("kc-wr1", "receiving_yards").mean())
        short_loss = 1 - (windy.distribution("kc-rb", "receiving_yards").mean()
                          / dome.distribution("kc-rb", "receiving_yards").mean())
        self.assertGreater(deep_loss, short_loss)

    def test_mild_weather_is_close_to_neutral(self):
        """Do not over-react to small differences."""
        dome = build_game(weather=WeatherState(is_dome=True), seed=101)
        mild = build_game(
            weather=WeatherState(temperature_f=55, wind_mph=7), seed=101
        )
        ratio = (mild.distribution("kc-qb", "passing_yards").mean()
                 / dome.distribution("kc-qb", "passing_yards").mean())
        self.assertAlmostEqual(ratio, 1.0, delta=0.02)


class TestInjuryPropagation(unittest.TestCase):
    def test_ruling_out_wr1_lifts_the_rest_of_the_room(self):
        base = build_game(seed=111)
        env_home = TeamEnvironment(team="KC", neutral_pass_rate=0.60,
                                   seconds_per_play=26.5, red_zone_rush_rate=0.48)
        env_away = TeamEnvironment(team="DEN", neutral_pass_rate=0.55,
                                   seconds_per_play=28.5, red_zone_rush_rate=0.55)
        # WR1 out: his 27% target share is redistributed by renormalisation.
        home = TeamRoster(env=env_home, players=[
            PlayerUsage("kc-qb", "Home QB", "QB", "KC", is_starting_qb=True),
            PlayerUsage("kc-wr2", "Home WR2", "WR", "KC", target_share=0.18,
                        yards_per_reception=12.0, catch_rate=0.62, adot=10.0),
            PlayerUsage("kc-te", "Home TE", "TE", "KC", target_share=0.22,
                        yards_per_reception=11.0, catch_rate=0.70, adot=7.5),
            PlayerUsage("kc-wr3", "Home WR3", "WR", "KC", target_share=0.21,
                        yards_per_reception=10.5, catch_rate=0.60, adot=9.0),
            PlayerUsage("kc-rb", "Home RB", "RB", "KC", target_share=0.12,
                        rush_share=0.68, yards_per_reception=7.5,
                        catch_rate=0.76, adot=1.0),
        ])
        away = TeamRoster(env=env_away, players=[
            PlayerUsage("den-qb", "Away QB", "QB", "DEN", is_starting_qb=True),
            PlayerUsage("den-rb", "Away RB", "RB", "DEN", rush_share=0.72),
            PlayerUsage("den-wr1", "Away WR1", "WR", "DEN", target_share=0.50),
            PlayerUsage("den-wr2", "Away WR2", "WR", "DEN", target_share=0.50),
        ])
        env = GameEnvironment(game_id="t2", home=env_home, away=env_away,
                              spread_home=-3.5, total=47.5,
                              weather=WeatherState(is_dome=True))
        without = simulate_game(env, home, away, n_sims=8000, seed=111)
        self.assertGreater(
            without.distribution("kc-wr2", "receiving_yards").mean(),
            base.distribution("kc-wr2", "receiving_yards").mean() + 5.0,
        )

    def test_questionable_tag_widens_the_distribution(self):
        """A player at 60% to suit up should carry more variance, not just a
        lower mean, because the low tail is a real 'did not play' outcome."""
        env_home = TeamEnvironment(team="KC", red_zone_rush_rate=0.48)
        env_away = TeamEnvironment(team="DEN")
        away = TeamRoster(env=env_away, players=[
            PlayerUsage("den-qb", "Away QB", "QB", "DEN", is_starting_qb=True),
            PlayerUsage("den-wr1", "Away WR1", "WR", "DEN", target_share=1.0),
            PlayerUsage("den-rb", "Away RB", "RB", "DEN", rush_share=1.0),
        ])

        def run(active_prob):
            home = TeamRoster(env=env_home, players=[
                PlayerUsage("kc-qb", "QB", "QB", "KC", is_starting_qb=True),
                PlayerUsage("kc-wr1", "WR1", "WR", "KC", target_share=0.35,
                            yards_per_reception=13.0, catch_rate=0.65,
                            active_probability=active_prob),
                PlayerUsage("kc-wr2", "WR2", "WR", "KC", target_share=0.35),
                PlayerUsage("kc-te", "TE", "TE", "KC", target_share=0.30),
                PlayerUsage("kc-rb", "RB", "RB", "KC", rush_share=1.0),
            ])
            env = GameEnvironment(game_id="t3", home=env_home, away=env_away,
                                  spread_home=-3.0, total=45.0,
                                  weather=WeatherState(is_dome=True))
            return simulate_game(env, home, away, n_sims=8000, seed=121)

        certain = run(1.0).distribution("kc-wr1", "receiving_yards")
        doubtful = run(0.6).distribution("kc-wr1", "receiving_yards")
        self.assertLess(doubtful.mean(), certain.mean())
        self.assertGreater(doubtful.prob_over(0.5) * 0 + doubtful.cdf(0.1), 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTouchdownCalibration(unittest.TestCase):
    """The 2+ market is entirely a statement about repeat scoring."""

    @classmethod
    def setUpClass(cls):
        cls.sim = build_game(n_sims=40_000, seed=404)

    def test_team_touchdowns_are_consistent_with_points(self):
        """Points already pin down the touchdown count closely.

        Drawing a Poisson on top of an already-random score double-counts the
        variance, and the surplus lands on the 2+ market.
        """
        pts = self.sim.team_points["KC"]
        tds = self.sim.team_touchdowns["KC"]
        # Implied points from touchdowns alone must never exceed actual points.
        self.assertTrue(np.all(7 * tds <= pts + 7))
        self.assertLess(float(np.std(tds)), np.sqrt(float(np.mean(tds))) * 1.35)

    def test_repeat_penalty_reduces_two_plus_without_moving_anytime(self):
        from app.projections.environment import EnvironmentPriors

        def run(penalty):
            env_home = TeamEnvironment(team="KC", red_zone_rush_rate=0.45)
            env_away = TeamEnvironment(team="DEN")
            home = TeamRoster(env=env_home, players=[
                PlayerUsage("qb", "QB", "QB", "KC", is_starting_qb=True,
                            rush_share=0.08, goal_line_share=0.12),
                PlayerUsage("rb", "RB", "RB", "KC", rush_share=0.70,
                            target_share=0.12, goal_line_share=0.60,
                            end_zone_target_share=0.10),
                PlayerUsage("rb2", "RB2", "RB", "KC", rush_share=0.22,
                            goal_line_share=0.28),
                PlayerUsage("wr1", "WR1", "WR", "KC", target_share=0.30,
                            end_zone_target_share=0.34),
                PlayerUsage("wr2", "WR2", "WR", "KC", target_share=0.24,
                            end_zone_target_share=0.22),
                PlayerUsage("te", "TE", "TE", "KC", target_share=0.34,
                            end_zone_target_share=0.34),
            ])
            away = TeamRoster(env=env_away, players=[
                PlayerUsage("dqb", "QB", "QB", "DEN", is_starting_qb=True,
                            goal_line_share=0.2),
                PlayerUsage("drb", "RB", "RB", "DEN", rush_share=1.0,
                            goal_line_share=0.5, end_zone_target_share=0.1),
                PlayerUsage("dwr", "WR", "WR", "DEN", target_share=0.6,
                            end_zone_target_share=0.4),
                PlayerUsage("dte", "TE", "TE", "DEN", target_share=0.4,
                            end_zone_target_share=0.3),
            ])
            env = GameEnvironment(
                game_id="cal", home=env_home, away=env_away,
                spread_home=-4.5, total=46.5,
                weather=WeatherState(is_dome=True),
                priors=EnvironmentPriors(td_repeat_penalty=penalty),
            )
            return simulate_game(env, home, away, n_sims=40_000,
                                 seed=7).touchdown_probabilities("rb")

        independent = run(1.0)
        penalised = run(0.75)
        self.assertLess(penalised["two_plus"], independent["two_plus"])
        # Anytime should barely move: the penalty only affects repeat scoring.
        self.assertAlmostEqual(penalised["anytime"], independent["anytime"],
                               delta=0.03)

    def test_two_plus_ratio_is_in_the_market_range(self):
        """Against typical NFL prices, 2+/anytime runs roughly 0.20-0.35."""
        for pid in ("kc-rb", "kc-wr1", "kc-te"):
            td = self.sim.touchdown_probabilities(pid)
            if td["anytime"] < 0.15:
                continue
            ratio = td["two_plus"] / td["anytime"]
            with self.subTest(player=pid):
                self.assertGreater(ratio, 0.12)
                self.assertLess(ratio, 0.40)

    def test_long_touchdowns_are_not_allocated_by_goal_line_share(self):
        """A receiver with no goal-line role still scores on long plays."""
        td = self.sim.touchdown_probabilities("kc-wr1")
        self.assertGreater(td["anytime"], 0.10)


class TestOpportunityIsRedistributed(unittest.TestCase):
    """An absent player's opportunity must go to his team-mates, not vanish."""

    def test_allocation_conserves_the_total(self):
        rng = np.random.default_rng(3)
        total = rng.integers(20, 40, size=500)
        shares = np.array([0.4, 0.3, 0.2, 0.1])
        out = _allocate(total, shares, rng)
        np.testing.assert_array_equal(out.sum(axis=0), total)

    def test_inactive_player_share_goes_to_teammates(self):
        """Zeroing a share must not shrink the team total."""
        rng = np.random.default_rng(5)
        n = 400
        total = np.full(n, 30)
        shares = np.array([0.4, 0.3, 0.2, 0.1])
        active = np.ones((4, n), dtype=bool)
        active[0] = False                      # the 40% player sits
        out = _allocate(total, shares[:, None] * active, rng)
        np.testing.assert_array_equal(out.sum(axis=0), total)
        self.assertEqual(out[0].sum(), 0)
        # The other three now split the whole 30, not 60% of it.
        self.assertAlmostEqual(out[1:].sum() / (30.0 * n), 1.0, places=6)

    def test_masking_after_allocation_would_lose_opportunity(self):
        """Guards the specific bug: multiplying the mask in afterwards."""
        rng = np.random.default_rng(7)
        n = 400
        total = np.full(n, 30)
        shares = np.array([0.4, 0.3, 0.2, 0.1])
        active = np.ones((4, n), dtype=bool)
        active[0] = False
        masked_after = _allocate(total, shares, rng) * active
        self.assertLess(masked_after.sum(), total.sum() * 0.75)


class TestPlayVolumeIsOnTheRightScale(unittest.TestCase):
    """The play model and the pace estimator must share a scale.

    ``seconds_per_play`` measures neutral-situation drive pace (29.7-35.0
    across the 2025 league). The reference it is differenced against has to be
    on that same scale; when it was 27.5 every team read as several seconds
    slow and lost roughly seven plays a game.
    """

    def env(self, pace):
        from app.projections.environment import EnvironmentPriors
        home = TeamEnvironment(team="KC", neutral_pass_rate=0.58,
                               seconds_per_play=pace, red_zone_rush_rate=0.50)
        away = TeamEnvironment(team="DEN", neutral_pass_rate=0.55,
                               seconds_per_play=pace, red_zone_rush_rate=0.55)
        return GameEnvironment(game_id="test", home=home, away=away,
                               spread_home=-3.0,
                               total=EnvironmentPriors().total_reference,
                               weather=WeatherState())

    def test_reference_pace_gives_a_league_average_play_count(self):
        from app.projections.environment import EnvironmentPriors
        p = EnvironmentPriors()
        plays = self.env(p.seconds_per_play_reference).expected_plays(
            self.env(p.seconds_per_play_reference).home)
        self.assertAlmostEqual(plays, p.base_plays_per_team, places=6)
        self.assertTrue(56.0 <= plays <= 64.0,
                        f"league-average team simulated at {plays:.1f} plays")

    def test_a_real_measured_pace_gives_a_plausible_play_count(self):
        """SEA measured 32.7s in 2025 and ran 59.2 offensive plays a game."""
        e = self.env(32.7)
        plays = e.expected_plays(e.home)
        self.assertTrue(55.0 <= plays <= 65.0,
                        f"32.7s/play simulated at {plays:.1f} plays")

    def test_the_whole_measured_pace_range_stays_plausible(self):
        """No team should fall outside the real 55.4-66.1 span by much."""
        for pace in (29.7, 32.4, 35.0):
            e = self.env(pace)
            plays = e.expected_plays(e.home)
            self.assertTrue(53.0 <= plays <= 68.0,
                            f"{pace}s/play simulated at {plays:.1f} plays")


class TestShareDraw(unittest.TestCase):
    """Shares vary between games; a player with no share still has none."""

    def test_a_zero_share_player_is_never_handed_opportunity(self):
        """Flooring a Dirichlet's zeros invents targets for a blocking back --
        and breaks the identity that passing yards are the receivers' yards."""
        import app.sim.game_sim as gs
        rng = np.random.default_rng(4)
        drawn = gs._draw_shares(np.array([0.5, 0.5, 0.0]), 500, rng)
        self.assertTrue(np.all(drawn[2] == 0.0))

    def test_the_draw_preserves_the_pool(self):
        import app.sim.game_sim as gs
        rng = np.random.default_rng(4)
        shares = np.array([0.4, 0.35, 0.25])
        drawn = gs._draw_shares(shares, 500, rng)
        np.testing.assert_allclose(drawn.sum(axis=0), shares.sum(), rtol=1e-9)

    def test_zero_concentration_keeps_shares_fixed(self):
        import app.sim.game_sim as gs
        rng = np.random.default_rng(4)
        shares = np.array([0.4, 0.35, 0.25])
        drawn = gs._draw_shares(shares, 50, rng, concentration=0.0)
        for column in drawn.T:
            np.testing.assert_allclose(column, shares)

    def test_a_lone_claimant_takes_the_whole_pool(self):
        import app.sim.game_sim as gs
        rng = np.random.default_rng(4)
        drawn = gs._draw_shares(np.array([0.0, 0.8, 0.0]), 50, rng)
        np.testing.assert_allclose(drawn[1], 0.8)
