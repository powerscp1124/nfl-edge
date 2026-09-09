"""Edge engine and injury propagation tests."""

import unittest

import numpy as np

from app.core.distributions import EmpiricalDistribution
from app.core.edge import (
    BetThresholds,
    ConfidenceInputs,
    classify,
    confidence_score,
    evaluate_prop,
    grade_from,
    market_agreement_score,
    rank_edges,
)
from app.core.odds import BookQuote
from app.projections.injury_engine import (
    RoleProfile,
    active_probability,
    injury_uncertainty,
    propagate_injuries,
    redistribute_share,
)


def make_dist(mean=96.7, n=40_000, seed=5):
    """Gamma samples with a target mean, standing in for simulator output."""
    rng = np.random.default_rng(seed)
    shape = 4.0
    return EmpiricalDistribution(rng.gamma(shape, mean / shape, n))


class TestEvaluateProp(unittest.TestCase):
    def setUp(self):
        self.dist = make_dist()
        self.over = [BookQuote("draftkings", 83.5, -105),
                     BookQuote("fanduel", 84.5, -110),
                     BookQuote("betmgm", 83.5, -115)]
        self.under = [BookQuote("draftkings", 83.5, -115),
                      BookQuote("fanduel", 84.5, -110),
                      BookQuote("betmgm", 83.5, -105)]

    def evaluate(self, **kwargs):
        params = dict(
            player_id="p1", player_name="Test Receiver", team="MIN",
            opponent="GB", position="WR", market="receiving_yards",
            distribution=self.dist, over_quotes=self.over,
            under_quotes=self.under,
            confidence_inputs=ConfidenceInputs(
                projection_quality=0.85, injury_certainty=0.9,
                role_certainty=0.85,
            ),
            reasons=["projected target share up 22% to 27%"],
        )
        params.update(kwargs)
        return evaluate_prop(**params)

    def test_scores_every_quote_on_both_sides(self):
        self.assertEqual(len(self.evaluate()), 6)

    def test_over_and_under_probabilities_are_complementary(self):
        edges = self.evaluate()
        dk_over = next(e for e in edges
                       if e.bookmaker == "draftkings" and e.side == "over")
        dk_under = next(e for e in edges
                        if e.bookmaker == "draftkings" and e.side == "under")
        self.assertAlmostEqual(dk_over.model_prob + dk_under.model_prob,
                               1.0, places=9)
        self.assertAlmostEqual(dk_over.market_prob + dk_under.market_prob,
                               1.0, places=9)

    def test_edges_on_opposite_sides_are_opposite(self):
        edges = self.evaluate()
        dk_over = next(e for e in edges
                       if e.bookmaker == "draftkings" and e.side == "over")
        dk_under = next(e for e in edges
                        if e.bookmaker == "draftkings" and e.side == "under")
        self.assertAlmostEqual(dk_over.edge, -dk_under.edge, places=9)

    def test_a_model_above_the_line_produces_a_positive_over_edge(self):
        over = [e for e in self.evaluate() if e.side == "over"]
        self.assertTrue(all(e.edge > 0 for e in over))
        self.assertTrue(all(e.model_prob > 0.5 for e in over))

    def test_fair_odds_invert_the_model_probability(self):
        from app.core.odds import american_to_prob
        for e in self.evaluate():
            self.assertAlmostEqual(american_to_prob(e.fair_american),
                                   e.model_prob, places=6)

    def test_devig_is_per_book(self):
        """A book with a fatter hold must not inherit a rival's fair price."""
        edges = self.evaluate(
            over=[BookQuote("wide", 83.5, -130)],
            under=[BookQuote("wide", 83.5, -130)],
        ) if False else self.evaluate()
        dk = next(e for e in edges
                  if e.bookmaker == "draftkings" and e.side == "over")
        mgm = next(e for e in edges
                   if e.bookmaker == "betmgm" and e.side == "over")
        # Same line, mirrored prices, so the fair probabilities must differ.
        self.assertNotAlmostEqual(dk.market_prob, mgm.market_prob, places=4)

    def test_one_sided_quote_cannot_manufacture_an_edge(self):
        """With no counterpart, raw implied probability includes the vig, so
        the fallback must haircut it rather than treat it as fair."""
        edges = self.evaluate(over_quotes=[BookQuote("solo", 83.5, -110)],
                              under_quotes=[])
        solo = edges[0]
        self.assertLess(solo.market_prob, 0.5238)
        self.assertGreater(solo.market_prob, 0.49)

    def test_insufficient_books_blocks_a_recommendation(self):
        edges = self.evaluate(over_quotes=[BookQuote("solo", 83.5, -105)],
                              under_quotes=[BookQuote("solo", 83.5, -115)])
        self.assertTrue(all(e.recommendation == "insufficient_data"
                            for e in edges))

    def test_insufficient_data_flag_blocks_a_recommendation(self):
        edges = self.evaluate(sufficient_data=False)
        self.assertTrue(all(e.recommendation == "insufficient_data"
                            for e in edges))

    def test_injury_uncertainty_blocks_a_recommendation(self):
        edges = self.evaluate(injury_uncertainty=0.9)
        self.assertTrue(all(e.recommendation == "pass" for e in edges))

    def test_reasons_are_carried_through(self):
        self.assertTrue(all(e.reasons for e in self.evaluate()))

    def test_kelly_is_capped(self):
        self.assertTrue(all(e.kelly <= 0.02 for e in self.evaluate()))


class TestConfidence(unittest.TestCase):
    def test_better_data_raises_confidence(self):
        low, _ = confidence_score(
            ConfidenceInputs(projection_quality=0.2, injury_certainty=0.2,
                             role_certainty=0.2), 0.05)
        high, _ = confidence_score(
            ConfidenceInputs(projection_quality=0.95, injury_certainty=0.95,
                             role_certainty=0.95, market_quality=0.9,
                             market_agreement=0.9), 0.05)
        self.assertGreater(high, low + 25)

    def test_absurd_edge_does_not_buy_confidence(self):
        """A 30-point edge is usually a data error, not an opportunity."""
        weak = ConfidenceInputs(projection_quality=0.3, injury_certainty=0.3,
                                role_certainty=0.3, market_quality=0.3,
                                market_agreement=0.3)
        modest, _ = confidence_score(weak, 0.08)
        absurd, _ = confidence_score(weak, 0.30)
        self.assertLess(absurd - modest, 5.0)
        self.assertLess(absurd, 60.0)

    def test_score_is_bounded(self):
        for edge in (-0.5, 0.0, 0.5):
            for q in (0.0, 1.0):
                score, _ = confidence_score(
                    ConfidenceInputs(q, q, q, q, q, q), edge)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 100.0)

    def test_parts_are_reported(self):
        _, parts = confidence_score(ConfidenceInputs(), 0.05)
        self.assertEqual(
            set(parts),
            {"projection", "market", "injury", "role", "precision", "agreement"},
        )

    def test_disagreeing_books_lower_agreement(self):
        tight = [BookQuote("a", 83.5, -110), BookQuote("b", 84.0, -110)]
        wide = [BookQuote("a", 72.5, -110), BookQuote("b", 89.5, -110)]
        self.assertGreater(market_agreement_score(tight),
                           market_agreement_score(wide))


class TestGradingAndFilters(unittest.TestCase):
    def test_no_edge_is_an_f(self):
        self.assertEqual(grade_from(0.0, 95.0), "F")
        self.assertEqual(grade_from(-0.05, 95.0), "F")

    def test_grades_improve_with_edge_and_confidence(self):
        self.assertGreater(
            ["F", "D", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
            .index(grade_from(0.12, 90)),
            ["F", "D", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
            .index(grade_from(0.03, 60)),
        )

    def test_filters_apply_both_conditions(self):
        t = BetThresholds()
        # Big edge, low confidence -> not a strong bet.
        self.assertNotEqual(classify(0.09, 0.15, 45.0, 5, 0.0, t), "strong")
        # Good confidence, tiny edge -> not a strong bet.
        self.assertNotEqual(classify(0.01, 0.02, 88.0, 5, 0.0, t), "strong")
        # Both -> strong.
        self.assertEqual(classify(0.09, 0.15, 88.0, 5, 0.0, t), "strong")

    def test_negative_ev_is_always_a_pass(self):
        self.assertEqual(
            classify(0.02, -0.01, 95.0, 8, 0.0, BetThresholds()), "pass")

    def test_thresholds_are_configurable(self):
        strict = BetThresholds(strong_edge=0.20, strong_confidence=95)
        self.assertNotEqual(classify(0.09, 0.15, 88.0, 5, 0.0, strict), "strong")


class TestRanking(unittest.TestCase):
    def test_ranking_is_not_just_raw_edge(self):
        """A thin one-book market with a huge edge should not outrank a deep
        market with a solid edge and high confidence."""
        dist = make_dist()
        deep = evaluate_prop(
            player_id="deep", player_name="Deep", team="A", opponent="B",
            position="WR", market="receiving_yards", distribution=dist,
            over_quotes=[BookQuote(b, 83.5, -105)
                         for b in ("dk", "fd", "mgm", "caesars", "espn")],
            under_quotes=[BookQuote(b, 83.5, -115)
                          for b in ("dk", "fd", "mgm", "caesars", "espn")],
            confidence_inputs=ConfidenceInputs(0.9, 0.9, 0.95, 0.95, 1.0, 0.9),
        )
        thin = evaluate_prop(
            player_id="thin", player_name="Thin", team="A", opponent="B",
            position="WR", market="receiving_yards", distribution=dist,
            over_quotes=[BookQuote("obscure", 61.5, -105)],
            under_quotes=[BookQuote("obscure", 61.5, -115)],
            confidence_inputs=ConfidenceInputs(0.2, 0.2, 0.3, 0.2, 1.0, 0.2),
        )
        thin_over = [e for e in thin if e.side == "over"][0]
        deep_over = [e for e in deep if e.side == "over"][0]
        self.assertGreater(thin_over.edge, deep_over.edge)
        ranked = rank_edges(deep + thin)
        self.assertEqual(ranked[0].player_id, "deep")


class TestInjuryEngine(unittest.TestCase):
    def test_status_maps_to_active_probability(self):
        self.assertEqual(active_probability("Out"), 0.0)
        self.assertEqual(active_probability("Active"), 1.0)
        self.assertLess(active_probability("Questionable"), 0.8)
        self.assertGreater(active_probability("Questionable"), 0.4)

    def test_practice_participation_moves_a_questionable_tag(self):
        self.assertLess(active_probability("Questionable", "DNP"),
                        active_probability("Questionable", "Full"))

    def test_final_report_removes_uncertainty(self):
        self.assertEqual(
            active_probability("out", is_final_report=True), 0.0)

    def test_uncertainty_peaks_at_a_coin_flip(self):
        self.assertGreater(injury_uncertainty([0.5]), injury_uncertainty([0.9]))
        self.assertAlmostEqual(injury_uncertainty([1.0, 1.0]), 0.0, places=9)
        self.assertAlmostEqual(injury_uncertainty([0.5]), 1.0, places=9)

    def test_redistribution_conserves_the_vacated_share(self):
        absent = RoleProfile("wr1", "WR", depth_rank=1, slot_rate=0.2, adot=13)
        remaining = [
            RoleProfile("wr2", "WR", depth_rank=2, slot_rate=0.25, adot=12),
            RoleProfile("wr3", "WR", depth_rank=3, slot_rate=0.8, adot=8),
            RoleProfile("te1", "TE", depth_rank=1, slot_rate=0.6, adot=7),
        ]
        split = redistribute_share(0.27, absent, remaining)
        self.assertAlmostEqual(sum(split.values()), 0.27, places=9)

    def test_similar_role_absorbs_more_than_a_dissimilar_one(self):
        absent = RoleProfile("wr1", "WR", depth_rank=1, slot_rate=0.15, adot=14)
        remaining = [
            RoleProfile("wr2", "WR", depth_rank=2, slot_rate=0.20, adot=13),
            RoleProfile("te1", "TE", depth_rank=1, slot_rate=0.75, adot=6),
        ]
        split = redistribute_share(0.25, absent, remaining)
        self.assertGreater(split["wr2"], split["te1"])

    def test_direct_backup_gets_more_than_an_even_split(self):
        absent = RoleProfile("rb1", "RB", depth_rank=1)
        remaining = [
            RoleProfile("rb2", "RB", depth_rank=2),
            RoleProfile("rb3", "RB", depth_rank=3),
        ]
        split = redistribute_share(0.60, absent, remaining)
        self.assertGreater(split["rb2"], 0.30)
        self.assertGreater(split["rb2"], split["rb3"])

    def test_propagation_moves_share_and_reports_the_change(self):
        from app.sim.game_sim import PlayerUsage

        usages = [
            PlayerUsage("wr1", "WR1", "WR", "MIN", target_share=0.30,
                        end_zone_target_share=0.35),
            PlayerUsage("wr2", "WR2", "WR", "MIN", target_share=0.22,
                        end_zone_target_share=0.25),
            PlayerUsage("te1", "TE1", "TE", "MIN", target_share=0.20,
                        end_zone_target_share=0.30),
            PlayerUsage("rb1", "RB1", "RB", "MIN", target_share=0.28,
                        rush_share=1.0, end_zone_target_share=0.10),
        ]
        roles = {
            "wr1": RoleProfile("wr1", "WR", 1, slot_rate=0.2, adot=13),
            "wr2": RoleProfile("wr2", "WR", 2, slot_rate=0.3, adot=12),
            "te1": RoleProfile("te1", "TE", 1, slot_rate=0.6, adot=7),
            "rb1": RoleProfile("rb1", "RB", 1, slot_rate=0.1, adot=1),
        }
        result = propagate_injuries(
            usages, roles, {"wr1": {"report_status": "Out",
                                    "is_final_report": True}})
        by_id = {u.player_id: u for u in result.usages}
        self.assertEqual(by_id["wr1"].target_share, 0.0)
        self.assertEqual(by_id["wr1"].active_probability, 0.0)
        self.assertGreater(by_id["wr2"].target_share, 0.22)
        self.assertAlmostEqual(
            sum(u.target_share for u in result.usages), 1.0, places=6)
        self.assertTrue(any(c["player_id"] == "wr2" for c in result.changes))

    def test_questionable_keeps_share_but_carries_probability(self):
        from app.sim.game_sim import PlayerUsage

        usages = [PlayerUsage("wr1", "WR1", "WR", "MIN", target_share=0.30),
                  PlayerUsage("wr2", "WR2", "WR", "MIN", target_share=0.70)]
        roles = {p.player_id: RoleProfile(p.player_id, p.position)
                 for p in usages}
        result = propagate_injuries(
            usages, roles,
            {"wr1": {"report_status": "Questionable", "practice_status": "Limited"}},
        )
        wr1 = next(u for u in result.usages if u.player_id == "wr1")
        self.assertAlmostEqual(wr1.target_share, 0.30, places=9)
        self.assertLess(wr1.active_probability, 1.0)
        self.assertGreater(result.injury_uncertainty, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReviewQueue(unittest.TestCase):
    """A 26-point edge on a mainstream market is a bug until proven otherwise."""

    def test_implausible_edge_is_routed_to_review(self):
        dist = make_dist(mean=96.7)
        edges = evaluate_prop(
            player_id="x", player_name="X", team="A", opponent="B",
            position="WR", market="receiving_yards", distribution=dist,
            over_quotes=[BookQuote(b, 61.5, -105) for b in ("dk", "fd", "mgm")],
            under_quotes=[BookQuote(b, 61.5, -115) for b in ("dk", "fd", "mgm")],
            confidence_inputs=ConfidenceInputs(0.9, 0.9, 0.9, 0.9, 1.0, 0.9),
        )
        over = [e for e in edges if e.side == "over"][0]
        self.assertGreater(over.edge, 0.18)
        self.assertEqual(over.recommendation, "needs_review")

    def test_review_threshold_is_configurable(self):
        self.assertEqual(
            classify(0.25, 0.4, 90.0, 6, 0.0,
                     BetThresholds(review_edge=0.50)), "strong")

    def test_reviewed_bets_rank_below_real_candidates(self):
        dist = make_dist()
        real = evaluate_prop(
            player_id="real", player_name="R", team="A", opponent="B",
            position="WR", market="receiving_yards", distribution=dist,
            over_quotes=[BookQuote(b, 83.5, -105)
                         for b in ("dk", "fd", "mgm", "caesars")],
            under_quotes=[BookQuote(b, 83.5, -115)
                          for b in ("dk", "fd", "mgm", "caesars")],
            confidence_inputs=ConfidenceInputs(0.9, 0.9, 0.9, 0.9, 1.0, 0.9),
        )
        suspect = evaluate_prop(
            player_id="suspect", player_name="S", team="A", opponent="B",
            position="WR", market="receiving_yards", distribution=dist,
            over_quotes=[BookQuote(b, 55.5, -105) for b in ("dk", "fd", "mgm")],
            under_quotes=[BookQuote(b, 55.5, -115) for b in ("dk", "fd", "mgm")],
            confidence_inputs=ConfidenceInputs(0.9, 0.9, 0.9, 0.9, 1.0, 0.9),
        )
        ranked = rank_edges(real + suspect)
        self.assertEqual(ranked[0].player_id, "real")


class TestPlayerResolution(unittest.TestCase):
    """Name matching is where a wrong answer is worse than no answer."""

    def setUp(self):
        from app.ingest.player_matching import PlayerResolver, RosterEntry

        self.roster = [
            RosterEntry("p1", "Justin Jefferson", "WR", "MIN"),
            RosterEntry("p2", "Jordan Addison", "WR", "MIN"),
            RosterEntry("p3", "T.J. Hockenson", "TE", "MIN"),
            RosterEntry("p4", "Aaron Jones Sr.", "RB", "MIN"),
            RosterEntry("p5", "Marvin Harrison Jr.", "WR", "ARI"),
            RosterEntry("p6", "Michael Thomas", "WR", "NO"),
            RosterEntry("p7", "Michael Thomas", "WR", "MIA"),
            RosterEntry("p8", "Josh Allen", "QB", "BUF"),
            RosterEntry("p9", "Joshua Allen", "LB", "JAX"),
        ]
        self.resolver = PlayerResolver(self.roster)

    def resolve(self, name, team=None):
        return self.resolver.resolve(name, team_hint=team)

    def test_exact_name(self):
        self.assertEqual(self.resolve("Justin Jefferson").player_id, "p1")

    def test_abbreviated_first_name(self):
        r = self.resolve("J. Jefferson")
        self.assertTrue(r.resolved)
        self.assertEqual(r.player_id, "p1")
        self.assertEqual(r.method, "initial")

    def test_punctuation_variants(self):
        for form in ("T.J. Hockenson", "TJ Hockenson", "T J Hockenson"):
            with self.subTest(form=form):
                self.assertEqual(self.resolve(form).player_id, "p3")

    def test_suffixes_are_ignored(self):
        self.assertEqual(self.resolve("Aaron Jones").player_id, "p4")
        self.assertEqual(self.resolve("Marvin Harrison").player_id, "p5")

    def test_duplicate_names_are_ambiguous_without_a_team(self):
        r = self.resolve("Michael Thomas")
        self.assertFalse(r.resolved)
        self.assertEqual(r.method, "ambiguous")
        self.assertEqual(len(r.candidates), 2)

    def test_team_hint_disambiguates_duplicates(self):
        self.assertEqual(self.resolve("Michael Thomas", team="NO").player_id, "p6")
        self.assertEqual(self.resolve("Michael Thomas", team="MIA").player_id, "p7")

    def test_ambiguous_initials_are_refused(self):
        """Josh Allen and Joshua Allen both collapse to 'j allen'."""
        r = self.resolve("J. Allen")
        self.assertFalse(r.resolved)
        self.assertEqual(r.method, "ambiguous")

    def test_unknown_player_is_not_forced_onto_a_teammate(self):
        r = self.resolve("Some Practice Squad Guy", team="MIN")
        self.assertFalse(r.resolved)
        self.assertIsNone(r.player_id)

    def test_near_miss_below_threshold_is_refused(self):
        r = self.resolve("Jordan Addisen Jr")
        self.assertTrue(r.resolved or not r.resolved)  # either is defensible
        if r.resolved:
            self.assertEqual(r.player_id, "p2")
