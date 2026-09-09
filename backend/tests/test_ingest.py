"""Ingestion tests.

The transformations are tested against synthetic frames shaped like nflverse
output. That does not prove the real frames have these columns — only a live
run does — but it proves the logic is right, so a live failure is a data
problem and can be diagnosed as one.
"""

import os
import unittest
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.ingest.context_loader import (
    red_zone_rush_rate,
    seconds_per_play,
    ContextError,
    build_depth_ranks,
    build_game_logs,
    build_injury_statuses,
    build_roster,
    build_team_environment,
    extract_market_lines,
    normalize_weekly,
    pick_starting_qb,
    red_zone_usage,
    team_volume_by_week,
)
from app.ingest.teams import (
    DOME_TEAMS,
    TEAMS,
    UnknownTeamError,
    canonical_team,
    is_dome,
    teams_from_event,
)


class TestTeamMapping(unittest.TestCase):
    def test_canonical_codes_pass_through(self):
        for code in TEAMS:
            self.assertEqual(canonical_team(code), code)

    def test_full_names_resolve(self):
        self.assertEqual(canonical_team("Minnesota Vikings"), "MIN")
        self.assertEqual(canonical_team("San Francisco 49ers"), "SF")
        self.assertEqual(canonical_team("Los Angeles Chargers"), "LAC")
        self.assertEqual(canonical_team("Los Angeles Rams"), "LAR")

    def test_the_two_la_teams_do_not_collide(self):
        """The specific failure this table exists to prevent."""
        self.assertNotEqual(canonical_team("Los Angeles Rams"),
                            canonical_team("Los Angeles Chargers"))

    def test_relocations_map_to_current_codes(self):
        self.assertEqual(canonical_team("OAK"), "LV")
        self.assertEqual(canonical_team("SD"), "LAC")
        self.assertEqual(canonical_team("STL"), "LAR")
        self.assertEqual(canonical_team("Oakland Raiders"), "LV")
        self.assertEqual(canonical_team("San Diego Chargers"), "LAC")

    def test_provider_spelling_variants(self):
        self.assertEqual(canonical_team("JAC"), "JAX")
        self.assertEqual(canonical_team("WSH"), "WAS")
        self.assertEqual(canonical_team("GNB"), "GB")
        self.assertEqual(canonical_team("LA"), "LAR")

    def test_former_names_resolve(self):
        self.assertEqual(canonical_team("Washington Football Team"), "WAS")

    def test_nicknames_resolve(self):
        self.assertEqual(canonical_team("Vikings"), "MIN")
        self.assertEqual(canonical_team("49ers"), "SF")

    def test_unknown_teams_raise_rather_than_guess(self):
        with self.assertRaises(UnknownTeamError):
            canonical_team("Toronto Argonauts")
        with self.assertRaises(UnknownTeamError):
            canonical_team("")
        self.assertIsNone(canonical_team("Toronto Argonauts", strict=False))

    def test_dome_flags(self):
        self.assertTrue(is_dome("MIN"))
        self.assertTrue(is_dome("Minnesota Vikings"))
        self.assertFalse(is_dome("GB"))
        self.assertTrue(DOME_TEAMS <= set(TEAMS))

    def test_teams_from_odds_api_event(self):
        pair = teams_from_event({"home_team": "Minnesota Vikings",
                                 "away_team": "Green Bay Packers"})
        self.assertEqual((pair.home, pair.away), ("MIN", "GB"))


def weekly_frame():
    rows = []
    for week in range(1, 9):
        for pid, name, pos, team, tgt, car in [
            ("qb-min", "QB Min", "QB", "MIN", 0, 3),
            ("wr-min", "WR Min", "WR", "MIN", 10, 0),
            ("wr2-min", "WR2 Min", "WR", "MIN", 6, 0),
            ("rb-min", "RB Min", "RB", "MIN", 4, 16),
            ("qb-gb", "QB GB", "QB", "GB", 0, 2),
            ("wr-gb", "WR GB", "WR", "GB", 9, 0),
            ("rb-gb", "RB GB", "RB", "GB", 3, 15),
            ("k-min", "Kicker", "K", "MIN", 0, 0),   # filtered out
        ]:
            rows.append({
                "player_id": pid, "player_display_name": name,
                "position": pos, "recent_team": team, "week": week,
                "targets": tgt, "receptions": round(tgt * 0.65),
                "receiving_yards": tgt * 8.0,
                "receiving_air_yards": tgt * 9.0,
                "carries": car, "rushing_yards": car * 4.3,
            })
    return pd.DataFrame(rows)


class TestNormalizeWeekly(unittest.TestCase):
    def test_renames_and_filters(self):
        df = normalize_weekly(weekly_frame())
        self.assertIn("name", df.columns)
        self.assertIn("team", df.columns)
        self.assertNotIn("K", set(df["position"]))

    def test_fullbacks_become_running_backs(self):
        frame = weekly_frame()
        frame.loc[0, "position"] = "FB"
        self.assertNotIn("FB", set(normalize_weekly(frame)["position"]))

    def test_missing_required_column_raises(self):
        frame = weekly_frame().drop(columns=["position"])
        with self.assertRaises(ContextError):
            normalize_weekly(frame)

    def test_missing_optional_column_defaults_to_zero(self):
        frame = weekly_frame().drop(columns=["receiving_air_yards"])
        df = normalize_weekly(frame)
        self.assertIn("air_yards", df.columns)
        self.assertEqual(df["air_yards"].sum(), 0.0)

    def test_team_codes_are_canonicalised(self):
        frame = weekly_frame()
        frame["recent_team"] = frame["recent_team"].replace({"GB": "GNB"})
        self.assertEqual(set(normalize_weekly(frame)["team"]), {"MIN", "GB"})


class TestTeamVolume(unittest.TestCase):
    def test_volumes_sum_across_players(self):
        vol = team_volume_by_week(normalize_weekly(weekly_frame()))
        min_week1 = vol[(vol["team"] == "MIN") & (vol["week"] == 1)].iloc[0]
        self.assertAlmostEqual(min_week1["team_rush_attempts"], 19.0)
        # 20 targets scaled up for throwaways.
        self.assertAlmostEqual(min_week1["team_pass_attempts"], 21.2, places=6)

    def test_shares_derived_from_this_total_sum_to_one(self):
        weekly = normalize_weekly(weekly_frame())
        vol = team_volume_by_week(weekly)
        merged = weekly.merge(vol, on=["team", "week"])
        min_wk = merged[(merged["team"] == "MIN") & (merged["week"] == 1)]
        share = (min_wk["carries"] / min_wk["team_rush_attempts"]).sum()
        self.assertAlmostEqual(share, 1.0, places=9)


class TestRedZoneUsage(unittest.TestCase):
    def test_counts_inside_the_five_and_ten(self):
        pbp = pd.DataFrame([
            {"play_type": "run", "yardline_100": 3, "week": 1, "posteam": "MIN",
             "rusher_player_id": "rb-min", "receiver_player_id": None},
            {"play_type": "run", "yardline_100": 12, "week": 1, "posteam": "MIN",
             "rusher_player_id": "rb-min", "receiver_player_id": None},
            {"play_type": "pass", "yardline_100": 8, "week": 1, "posteam": "MIN",
             "rusher_player_id": None, "receiver_player_id": "wr-min"},
        ])
        rz = red_zone_usage(pbp)
        rb = rz[rz["player_id"] == "rb-min"].iloc[0]
        self.assertEqual(rb["gl_carries"], 1.0)   # the 12-yard-line run excluded
        wr = rz[rz["player_id"] == "wr-min"].iloc[0]
        self.assertEqual(wr["ez_targets"], 1.0)

    def test_empty_pbp_returns_empty_frame(self):
        self.assertTrue(red_zone_usage(pd.DataFrame()).empty)


class TestBuildGameLogs(unittest.TestCase):
    def setUp(self):
        self.weekly = normalize_weekly(weekly_frame())
        self.volume = team_volume_by_week(self.weekly)

    def build(self, **kw):
        return build_game_logs(self.weekly, self.volume, pd.DataFrame(),
                               pd.DataFrame(), ["MIN", "GB"], **kw)

    def test_groups_by_team_and_player(self):
        logs = self.build()
        self.assertEqual(set(logs), {"MIN", "GB"})
        self.assertIn("wr-min", logs["MIN"])
        self.assertEqual(len(logs["MIN"]["wr-min"]), 8)

    def test_through_week_excludes_the_target_week(self):
        """The point-in-time guard for backtests."""
        logs = self.build(through_week=5)
        weeks = [r["week"] for r in logs["MIN"]["wr-min"]]
        self.assertEqual(max(weeks), 4)
        self.assertNotIn(5, weeks)

    def test_lookback_window_is_applied(self):
        logs = self.build(lookback_weeks=3)
        self.assertLessEqual(len(logs["MIN"]["wr-min"]), 3)

    def test_no_data_before_the_cutoff_raises(self):
        with self.assertRaises(ContextError):
            self.build(through_week=1)

    def test_logs_carry_team_volume(self):
        row = self.build()["MIN"]["wr-min"][0]
        self.assertGreater(row["team_pass_attempts"], 0)
        self.assertGreater(row["team_rush_attempts"], 0)


class TestTeamEnvironment(unittest.TestCase):
    def test_pass_rate_reflects_actual_volume(self):
        weekly = normalize_weekly(weekly_frame())
        vol = team_volume_by_week(weekly)
        env = build_team_environment(weekly, vol, "MIN")
        self.assertGreater(env["neutral_pass_rate"], 0.4)
        self.assertLess(env["neutral_pass_rate"], 0.75)
        self.assertTrue(env["_pace_is_approximate"])

    def test_missing_team_raises(self):
        weekly = normalize_weekly(weekly_frame())
        vol = team_volume_by_week(weekly)
        with self.assertRaises(ContextError):
            build_team_environment(weekly, vol, "KC")


class TestMarketLines(unittest.TestCase):
    def payload(self, spreads=(-2.5, -3.0, -2.5), totals=(47.5, 47.0, 48.0)):
        books = []
        for s, t in zip(spreads, totals):
            books.append({"key": f"book{s}{t}", "markets": [
                {"key": "spreads", "outcomes": [
                    {"name": "Minnesota Vikings", "point": s},
                    {"name": "Green Bay Packers", "point": -s}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": t}, {"name": "Under", "point": t}]},
            ]})
        return [{"id": "e1", "home_team": "Minnesota Vikings",
                 "away_team": "Green Bay Packers", "bookmakers": books}]

    def test_uses_the_median_across_books(self):
        out = extract_market_lines(self.payload(), "e1")
        self.assertAlmostEqual(out["spread_home"], -2.5)
        self.assertAlmostEqual(out["total"], 47.5)

    def test_one_stale_book_does_not_move_the_median(self):
        out = extract_market_lines(
            self.payload(spreads=(-2.5, -3.0, -9.5)), "e1")
        self.assertAlmostEqual(out["spread_home"], -3.0)

    def test_spread_is_taken_from_the_home_side(self):
        out = extract_market_lines(self.payload(spreads=(-6.5,) * 3), "e1")
        self.assertAlmostEqual(out["spread_home"], -6.5)

    def test_missing_event_raises(self):
        with self.assertRaises(ContextError):
            extract_market_lines(self.payload(), "nope")

    def test_missing_lines_block_the_projection(self):
        payload = [{"id": "e1", "home_team": "Minnesota Vikings",
                    "away_team": "Green Bay Packers", "bookmakers": []}]
        with self.assertRaises(ContextError):
            extract_market_lines(payload, "e1")


class TestDepthAndInjuries(unittest.TestCase):
    def test_latest_depth_row_wins(self):
        depth = pd.DataFrame([
            {"player_id": "wr-min", "team": "MIN", "depth_team": 2,
             "observed_at": datetime(2026, 9, 1, tzinfo=timezone.utc)},
            {"player_id": "wr-min", "team": "MIN", "depth_team": 1,
             "observed_at": datetime(2026, 9, 5, tzinfo=timezone.utc)},
        ])
        self.assertEqual(build_depth_ranks(depth, ["MIN"])["wr-min"], 1)

    def test_depth_respects_as_of(self):
        depth = pd.DataFrame([
            {"player_id": "wr-min", "team": "MIN", "depth_team": 2,
             "observed_at": datetime(2026, 9, 1, tzinfo=timezone.utc)},
            {"player_id": "wr-min", "team": "MIN", "depth_team": 1,
             "observed_at": datetime(2026, 9, 5, tzinfo=timezone.utc)},
        ])
        ranks = build_depth_ranks(
            depth, ["MIN"], as_of=datetime(2026, 9, 3, tzinfo=timezone.utc))
        self.assertEqual(ranks["wr-min"], 2)

    def test_injury_statuses_are_extracted(self):
        inj = pd.DataFrame([
            {"player_id": "wr-min", "team": "MIN", "report_status": "Out",
             "practice_status": "DNP", "week": 5},
        ])
        out = build_injury_statuses(inj, ["MIN"])
        self.assertEqual(out["wr-min"]["report_status"], "Out")
        self.assertEqual(out["wr-min"]["practice_status"], "DNP")

    def test_blank_status_is_not_an_injury(self):
        inj = pd.DataFrame([
            {"player_id": "wr-min", "team": "MIN", "report_status": np.nan,
             "week": 5},
        ])
        self.assertEqual(build_injury_statuses(inj, ["MIN"]), {})

    def test_other_teams_are_ignored(self):
        inj = pd.DataFrame([
            {"player_id": "x", "team": "KC", "report_status": "Out", "week": 5},
        ])
        self.assertEqual(build_injury_statuses(inj, ["MIN", "GB"]), {})


class TestRosterAndQB(unittest.TestCase):
    def setUp(self):
        weekly = normalize_weekly(weekly_frame())
        self.logs = build_game_logs(weekly, team_volume_by_week(weekly),
                                    pd.DataFrame(), pd.DataFrame(),
                                    ["MIN", "GB"])

    def test_roster_covers_every_player(self):
        roster = build_roster(self.logs)
        ids = {r["player_id"] for r in roster}
        self.assertIn("wr-min", ids)
        self.assertIn("rb-gb", ids)
        self.assertTrue(all(r["team"] in ("MIN", "GB") for r in roster))

    def test_starting_qb_is_identified_per_team(self):
        qbs = pick_starting_qb(self.logs)
        self.assertEqual(qbs["MIN"], "qb-min")
        self.assertEqual(qbs["GB"], "qb-gb")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPaceAndRedZone(unittest.TestCase):
    """Both were hardcoded; both are now measured, with honest fallbacks."""

    def pbp(self, n_run=30, n_pass=30):
        rows = []
        clock = 1800
        for i in range(n_run + n_pass):
            clock -= 26
            rows.append({
                "game_id": "g1", "posteam": "MIN", "qtr": 2,
                "score_differential": 3, "yardline_100": 15,
                "play_type": "run" if i < n_run else "pass",
                "game_seconds_remaining": clock,
            })
        return pd.DataFrame(rows)

    def test_red_zone_rush_rate_is_measured(self):
        from app.ingest.context_loader import red_zone_rush_rate
        rate = red_zone_rush_rate(self.pbp(n_run=45, n_pass=15), "MIN")
        self.assertAlmostEqual(rate, 0.75, delta=0.02)

    def test_run_heavy_and_pass_heavy_teams_differ(self):
        from app.ingest.context_loader import red_zone_rush_rate
        run_heavy = red_zone_rush_rate(self.pbp(50, 10), "MIN")
        pass_heavy = red_zone_rush_rate(self.pbp(10, 50), "MIN")
        self.assertGreater(run_heavy, pass_heavy + 0.4)

    def test_thin_sample_falls_back_to_league_average(self):
        from app.ingest.context_loader import red_zone_rush_rate
        self.assertEqual(red_zone_rush_rate(self.pbp(5, 3), "MIN"), 0.50)

    def test_pace_is_measured_from_the_clock(self):
        from app.ingest.context_loader import seconds_per_play
        self.assertAlmostEqual(seconds_per_play(self.pbp(), "MIN"), 26.0, delta=1.0)

    def test_pace_falls_back_when_columns_are_missing(self):
        from app.ingest.context_loader import seconds_per_play
        frame = self.pbp().drop(columns=["game_seconds_remaining"])
        self.assertEqual(seconds_per_play(frame, "MIN"), 27.5)

    def bimodal_pbp(self, n_drives=12, plays_per_drive=6):
        """Real pace: the clock stops on an incompletion, runs after a run.

        Gaps are therefore bimodal -- 5s after a stopped clock, 40s after a
        running one. Within a drive that is 40,5,40,5,40: mean 26.0s, median
        40.0s. The median is the bug: it sits inside the upper cluster and,
        once clamped, reads 35s for every team in the league.

        Drives are separated by 50s, a plausible three-and-out, which is short
        enough to survive the 60s filter and so cannot be excluded by gap size
        alone -- only by grouping on the drive.
        """
        rows = []
        clock = 1800
        for drive in range(n_drives):
            for i in range(plays_per_drive):
                stopped = i % 2 == 0
                clock -= 5 if stopped else 40
                rows.append({
                    "game_id": "g1", "posteam": "MIN", "qtr": 2,
                    "score_differential": 3, "yardline_100": 40,
                    "play_type": "pass" if stopped else "run",
                    "drive": drive + 1,
                    "game_seconds_remaining": clock,
                })
            clock -= 45  # the opponent's possession, plus 5 for the next snap
        return pd.DataFrame(rows)

    def test_pace_is_not_biased_by_the_bimodal_clock(self):
        """The median lands in the upper cluster and reads far too slow."""
        from app.ingest.context_loader import seconds_per_play
        pace = seconds_per_play(self.bimodal_pbp(), "MIN")
        self.assertAlmostEqual(pace, 26.0, delta=0.5)
        # The shipped median returned 40.0 here, which the clamp then reported
        # as 35.0 -- indistinguishable from every other team in the league.
        self.assertLess(pace, 35.0)

    def test_pace_gap_never_spans_a_change_of_possession(self):
        """A three-and-out is too short to exclude by gap size alone."""
        from app.ingest.context_loader import seconds_per_play
        frame = self.bimodal_pbp()
        with_drives = seconds_per_play(frame, "MIN")
        without = seconds_per_play(frame.drop(columns=["drive"]), "MIN")
        self.assertLess(with_drives, without)

    def test_pace_prefers_drive_time_of_possession(self):
        """With the drive columns, pace is TOP over the plays it covers."""
        from app.ingest.context_loader import seconds_per_play
        frame = self.bimodal_pbp()
        frame["drive_time_of_possession"] = "2:30"   # 150s / 6 plays = 25.0
        self.assertAlmostEqual(seconds_per_play(frame, "MIN"), 25.0, delta=0.01)


class TestWeatherWiring(unittest.TestCase):
    """A forecast carries provenance that WeatherState does not accept.

    `fetch_context` used to pass `weather=None` unconditionally, so every live
    projection ran on neutral conditions and said so in a warning. Wiring the
    fetch in without filtering would instead raise on construction.
    """

    def test_the_field_list_matches_WeatherState(self):
        """If WeatherState gains a field, this list has to gain it too."""
        import dataclasses
        from app.ingest.context_loader import WEATHER_STATE_FIELDS
        from app.projections.environment import WeatherState
        actual = {f.name for f in dataclasses.fields(WeatherState)}
        self.assertEqual(set(WEATHER_STATE_FIELDS), actual)

    def test_a_real_forecast_constructs_a_WeatherState(self):
        from app.ingest.context_loader import WEATHER_STATE_FIELDS
        from app.projections.environment import WeatherState
        forecast = {"temperature_f": 41.0, "wind_mph": 18.0,
                    "precipitation_prob": 0.4, "is_dome": False,
                    "source": "open-meteo"}
        state = WeatherState(**{k: forecast[k] for k in WEATHER_STATE_FIELDS
                                if k in forecast})
        self.assertAlmostEqual(state.effective_wind, 18.0)

    def test_an_unavailable_forecast_still_constructs(self):
        from app.ingest.context_loader import WEATHER_STATE_FIELDS
        from app.projections.environment import WeatherState
        forecast = {"temperature_f": 60.0, "wind_mph": 0.0,
                    "precipitation_prob": 0.0, "is_dome": False,
                    "source": "unavailable", "reason": "timeout"}
        state = WeatherState(**{k: forecast[k] for k in WEATHER_STATE_FIELDS
                                if k in forecast})
        self.assertEqual(state.effective_wind, 0.0)

    def test_a_dome_reports_no_wind(self):
        from app.ingest.context_loader import WEATHER_STATE_FIELDS
        from app.projections.environment import WeatherState
        forecast = {"temperature_f": 70.0, "wind_mph": 0.0,
                    "precipitation_prob": 0.0, "is_dome": True,
                    "source": "dome"}
        state = WeatherState(**{k: forecast[k] for k in WEATHER_STATE_FIELDS
                                if k in forecast})
        self.assertTrue(state.is_dome)
        self.assertEqual(state.effective_wind, 0.0)


class TestNonPlayerMarkets(unittest.TestCase):
    """Books price things that are not players."""

    def test_team_defences_are_not_players(self):
        from app.ingest.player_matching import is_non_player_market
        for name in ("Seattle Seahawks Defense", "Seattle Seahawks D/ST",
                     "New England Patriots D/ST", "New England Patriots"):
            self.assertTrue(is_non_player_market(name), name)

    def test_novelty_markets_are_not_players(self):
        from app.ingest.player_matching import is_non_player_market
        self.assertTrue(is_non_player_market("No Scorer"))

    def test_real_players_are_players(self):
        from app.ingest.player_matching import is_non_player_market
        for name in ("Jaxon Smith-Njigba", "Cooper Kupp", "A.J. Brown",
                     "Rhamondre Stevenson", "Drake Maye"):
            self.assertFalse(is_non_player_market(name), name)


class TestRosterIncludesTheDepthChart(unittest.TestCase):
    """A rookie with no stat line is a data gap, not a matching failure."""

    def logs(self):
        return {"SEA": {"p1": [{"name": "Played Player", "position": "WR",
                                "week": 5}]}}

    def depth(self):
        return pd.DataFrame([
            {"gsis_id": "p1", "team": "SEA", "dt": "2026-01-01",
             "player_name": "Played Player", "pos_abb": "WR", "pos_rank": 1},
            {"gsis_id": "p2", "team": "SEA", "dt": "2026-01-01",
             "player_name": "Rookie Receiver", "pos_abb": "WR", "pos_rank": 4},
            {"gsis_id": "p3", "team": "SEA", "dt": "2026-01-01",
             "player_name": "Some Cornerback", "pos_abb": "LCB",
             "pos_rank": 1},
        ])

    def test_a_player_with_no_stat_line_is_still_resolvable(self):
        from app.ingest.context_loader import build_roster
        names = {r["full_name"] for r in
                 build_roster(self.logs(), depth=self.depth(), teams=["SEA"])}
        self.assertIn("Rookie Receiver", names)

    def test_defenders_are_not_added(self):
        from app.ingest.context_loader import build_roster
        names = {r["full_name"] for r in
                 build_roster(self.logs(), depth=self.depth(), teams=["SEA"])}
        self.assertNotIn("Some Cornerback", names)

    def test_players_are_not_duplicated(self):
        from app.ingest.context_loader import build_roster
        roster = build_roster(self.logs(), depth=self.depth(), teams=["SEA"])
        ids = [r["player_id"] for r in roster]
        self.assertEqual(len(ids), len(set(ids)))

    def test_omitting_the_depth_chart_keeps_the_old_behaviour(self):
        from app.ingest.context_loader import build_roster
        self.assertEqual(len(build_roster(self.logs())), 1)


class TestPlayByPlayIsPointInTime(unittest.TestCase):
    """Pace and red-zone tendency come from play-by-play, and that frame has
    to respect the same cutoff the weekly frame does."""

    def pbp(self):
        rows = []
        clock = 1800
        for week in (1, 2, 9):
            for i in range(60):
                clock -= 26 if week < 9 else 5   # week 9 is wildly faster
                rows.append({"game_id": f"g{week}", "posteam": "MIN",
                             "qtr": 2, "score_differential": 3,
                             "yardline_100": 40, "week": week,
                             "play_type": "run" if i % 2 else "pass",
                             "game_seconds_remaining": clock})
            clock = 1800
        return pd.DataFrame(rows)

    def test_a_later_week_cannot_change_an_earlier_projection(self):
        from app.ingest.context_loader import seconds_per_play
        full = seconds_per_play(self.pbp(), "MIN")
        to_date = seconds_per_play(self.pbp()[self.pbp()["week"] < 3], "MIN")
        self.assertNotAlmostEqual(full, to_date, places=3)
        self.assertAlmostEqual(to_date, 26.0, delta=1.0)


class TestRotationPool(unittest.TestCase):
    """Opportunity pools are renormalised to 1.0, so who is in the pool
    decides how much is left for everyone else."""

    def test_a_starter_is_in_the_rotation(self):
        from app.ingest.context_loader import in_rotation
        self.assertTrue(in_rotation("WR", 1, 0.83))

    def test_a_deep_reserve_on_a_fifth_of_the_snaps_is_not(self):
        from app.ingest.context_loader import in_rotation
        self.assertFalse(in_rotation("WR", 6, 0.21))

    def test_snaps_alone_can_keep_a_low_ranked_player(self):
        """A TE3 playing half the snaps is in the rotation whatever the chart
        says."""
        from app.ingest.context_loader import in_rotation
        self.assertTrue(in_rotation("TE", 3, 0.51))

    def test_rank_alone_can_keep_a_lightly_used_starter(self):
        from app.ingest.context_loader import in_rotation
        self.assertTrue(in_rotation("RB", 1, 0.10))

    def test_a_rank_from_another_position_group_is_not_evidence(self):
        """Listed RB5, projected WR: participation has to decide alone."""
        from app.ingest.context_loader import in_rotation
        self.assertFalse(in_rotation("WR", None, 0.21))
        self.assertTrue(in_rotation("WR", None, 0.60))

    def test_a_priced_player_is_never_pruned(self):
        """Pruning must not cost market coverage."""
        from app.ingest.context_loader import in_rotation
        self.assertFalse(in_rotation("WR", 7, 0.09))
        self.assertTrue(in_rotation("WR", 7, 0.09, has_market=True))

    def test_quarterbacks_are_never_pruned(self):
        from app.ingest.context_loader import in_rotation
        self.assertTrue(in_rotation("QB", None, None))

    def test_unknown_participation_does_not_keep_a_deep_reserve(self):
        from app.ingest.context_loader import in_rotation
        self.assertFalse(in_rotation("WR", 6, None))


class TestDepthChartPositionGroups(unittest.TestCase):
    """A rank is only meaningful inside its own position group."""

    def frame(self):
        return pd.DataFrame([
            {"gsis_id": "rb1", "team": "SEA", "dt": "2026-01-01",
             "pos_abb": "RB", "pos_rank": 1},
            {"gsis_id": "rb2", "team": "SEA", "dt": "2026-01-01",
             "pos_abb": "RB", "pos_rank": 2},
            {"gsis_id": "fb1", "team": "SEA", "dt": "2026-01-01",
             "pos_abb": "FB", "pos_rank": 1},
            {"gsis_id": "wr1", "team": "SEA", "dt": "2026-01-01",
             "pos_abb": "WR", "pos_rank": 1},
        ])

    def test_a_fullback_does_not_outrank_the_starting_back(self):
        """FB1 taken at face value hands a blocker the RB1 prior."""
        from app.ingest.context_loader import build_depth_ranks
        ranks = build_depth_ranks(self.frame(), ["SEA"])
        self.assertEqual(ranks["rb1"], 1)
        self.assertGreater(ranks["fb1"], ranks["rb2"])

    def test_a_fullback_is_grouped_with_the_backs(self):
        from app.ingest.context_loader import build_depth_positions
        pos = build_depth_positions(self.frame(), ["SEA"])
        self.assertEqual(pos["fb1"], "RB")
        self.assertEqual(pos["wr1"], "WR")

    def test_primary_labels_keep_their_own_rank(self):
        from app.ingest.context_loader import build_depth_ranks
        ranks = build_depth_ranks(self.frame(), ["SEA"])
        self.assertEqual(ranks["wr1"], 1)
        self.assertEqual(ranks["rb2"], 2)

    def test_the_prior_a_fullback_receives_is_a_backup_prior(self):
        from app.ingest.context_loader import build_depth_ranks
        from app.projections.usage_builder import opportunity_prior
        ranks = build_depth_ranks(self.frame(), ["SEA"])
        starter = opportunity_prior("RB", ranks["rb1"])["rush_share"]
        blocker = opportunity_prior("RB", ranks["fb1"])["rush_share"]
        self.assertLess(blocker, starter)


class TestSnapCrosswalk(unittest.TestCase):
    """Snap counts are keyed by PFR id, everything else by gsis id."""

    def snaps(self):
        return pd.DataFrame([
            {"pfr_player_id": "SmitJa00", "week": 1, "offense_pct": 0.88},
            {"pfr_player_id": "KuppCo00", "week": 1, "offense_pct": 0.75},
        ])

    def id_map(self):
        return pd.DataFrame([
            {"gsis_id": "00-001", "pfr_id": "SmitJa00"},
            {"gsis_id": "00-002", "pfr_id": "KuppCo00"},
        ])

    def test_pfr_ids_are_translated_to_gsis_ids(self):
        from app.ingest.context_loader import snap_shares
        out = snap_shares(self.snaps(), id_map=self.id_map())
        self.assertEqual(sorted(out["player_id"]), ["00-001", "00-002"])

    def test_without_the_map_there_is_no_player_id_to_join_on(self):
        """The silent failure: build_game_logs then skips the merge entirely."""
        from app.ingest.context_loader import snap_shares
        out = snap_shares(self.snaps())
        self.assertNotIn("player_id", out.columns)

    def test_snap_share_actually_reaches_the_game_log(self):
        """A renamed column would pass the unit test and still merge nothing."""
        from app.ingest.context_loader import (build_game_logs, snap_shares,
                                               team_volume_by_week)
        weekly = normalize_weekly(weekly_frame())
        volume = team_volume_by_week(weekly)
        pid = str(weekly["player_id"].iloc[0])
        week = int(weekly["week"].iloc[0])
        snaps = pd.DataFrame([{"pfr_player_id": "AbcdEf00", "week": week,
                               "offense_pct": 0.91}])
        id_map = pd.DataFrame([{"gsis_id": pid, "pfr_id": "AbcdEf00"}])
        snap = snap_shares(snaps, id_map=id_map)
        logs = build_game_logs(weekly, volume, pd.DataFrame(), snap,
                               [str(weekly["team"].iloc[0])])
        shares = [row["snap_share"]
                  for players in logs.values()
                  for rows in players.values()
                  for row in rows]
        self.assertIn(0.91, shares)


class TestSnapShareIsPointInTime(unittest.TestCase):
    """snap_share lives in player_game_stats, which is never readable during a
    backtest. In the live path it rides the through_week guard on the weekly
    frame; this pins that so it cannot drift."""

    def test_snaps_after_through_week_never_reach_the_log(self):
        from app.ingest.context_loader import (build_game_logs, snap_shares,
                                               team_volume_by_week)
        weekly = normalize_weekly(weekly_frame())
        volume = team_volume_by_week(weekly)
        pid = str(weekly["player_id"].iloc[0])
        team = str(weekly["team"].iloc[0])
        weeks = sorted(int(w) for w in weekly["week"].unique())
        cutoff = weeks[-1]
        snaps = pd.DataFrame([
            {"pfr_player_id": "AbcdEf00", "week": w,
             "offense_pct": 0.99 if w >= cutoff else 0.40}
            for w in weeks
        ])
        id_map = pd.DataFrame([{"gsis_id": pid, "pfr_id": "AbcdEf00"}])
        snap = snap_shares(snaps, id_map=id_map)
        logs = build_game_logs(weekly, volume, pd.DataFrame(), snap, [team],
                               through_week=cutoff)
        shares = [row["snap_share"]
                  for players in logs.values()
                  for rows in players.values()
                  for row in rows]
        self.assertNotIn(0.99, shares)

    def test_unresolved_snap_share_is_none_not_zero(self):
        """A failed crosswalk must not read as 'never on the field'."""
        from app.ingest.context_loader import (build_game_logs,
                                               team_volume_by_week)
        weekly = normalize_weekly(weekly_frame())
        volume = team_volume_by_week(weekly)
        logs = build_game_logs(weekly, volume, pd.DataFrame(), pd.DataFrame(),
                               [str(weekly["team"].iloc[0])])
        shares = [row["snap_share"]
                  for players in logs.values()
                  for rows in players.values()
                  for row in rows]
        self.assertTrue(all(v is None for v in shares))


class TestInjuryStatusRecency(unittest.TestCase):
    """The latest report, not the latest non-null value of every column."""

    def frame(self):
        return pd.DataFrame([
            {"gsis_id": "p1", "team": "SEA", "week": 3,
             "report_status": "Questionable", "practice_status": None},
            {"gsis_id": "p1", "team": "SEA", "week": 21,
             "report_status": None,
             "practice_status": "Full Participation in Practice"},
            {"gsis_id": "p1", "team": "SEA", "week": 22,
             "report_status": None,
             "practice_status": "Full Participation in Practice"},
            {"gsis_id": "p2", "team": "SEA", "week": 22,
             "report_status": "Out", "practice_status": None},
        ])

    def test_a_recovered_player_is_not_still_questionable(self):
        """groupby().last() reached back to week 3 and re-applied it."""
        out = build_injury_statuses(self.frame(), ["SEA"])
        self.assertNotIn("p1", out)

    def test_a_genuinely_injured_player_is_kept(self):
        out = build_injury_statuses(self.frame(), ["SEA"])
        self.assertEqual(out["p2"]["report_status"], "Out")


class FakePolarsFrame:
    """Stands in for a Polars frame: exposes .columns, .select, .to_pandas."""

    def __init__(self, df):
        self._df = df

    @property
    def columns(self):
        return list(self._df.columns)

    def select(self, cols):
        return FakePolarsFrame(self._df[cols])

    def to_pandas(self):
        return self._df.copy()


class TestPandasBoundary(unittest.TestCase):
    """nflreadpy returns Polars; the conversion happens in exactly one place."""

    def frame(self):
        return pd.DataFrame({"game_id": ["g1"], "play_type": ["run"],
                             "epa": [0.4], "unused_column": [1]})

    def test_converts_polars_to_pandas(self):
        from app.ingest.nflverse import to_pandas
        out = to_pandas(FakePolarsFrame(self.frame()), "test")
        self.assertIsInstance(out, pd.DataFrame)

    def test_stamps_source_and_time(self):
        from app.ingest.nflverse import to_pandas
        out = to_pandas(FakePolarsFrame(self.frame()), "nflverse_pbp")
        self.assertEqual(out["_source"].iloc[0], "nflverse_pbp")
        self.assertIn("_ingested_at", out.columns)

    def test_selects_columns_before_converting(self):
        from app.ingest.nflverse import to_pandas
        out = to_pandas(FakePolarsFrame(self.frame()), "test",
                        columns=["game_id", "epa"])
        self.assertNotIn("unused_column", out.columns)
        self.assertIn("epa", out.columns)

    def test_missing_columns_are_skipped_not_raised(self):
        """nflverse renames fields between seasons; one absence must not kill
        an ingest."""
        from app.ingest.nflverse import to_pandas
        out = to_pandas(FakePolarsFrame(self.frame()), "test",
                        columns=["game_id", "a_column_that_does_not_exist"])
        self.assertIn("game_id", out.columns)

    def test_accepts_a_pandas_frame_directly(self):
        from app.ingest.nflverse import to_pandas
        out = to_pandas(self.frame(), "test", columns=["game_id"])
        self.assertEqual(list(out.columns)[:1], ["game_id"])


class TestDotenvLoader(unittest.TestCase):
    """A .env that is silently ignored is worse than no .env."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / ".env"
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def write(self, text):
        self.path.write_text(text)
        return self.path

    def test_loads_simple_pairs(self):
        from app.env import load_dotenv
        path = self.write("FOO=bar\nBAZ=qux\n")
        loaded = load_dotenv(path)
        self.assertEqual(set(loaded), {"FOO", "BAZ"})
        self.assertEqual(os.environ["FOO"], "bar")

    def test_ignores_comments_and_blank_lines(self):
        from app.env import load_dotenv
        path = self.write("# a comment\n\nFOO=bar\n")
        self.assertEqual(load_dotenv(path), ["FOO"])

    def test_strips_quotes(self):
        from app.env import load_dotenv
        load_dotenv(self.write('FOO="bar"\nBAZ=\'qux\'\n'))
        self.assertEqual(os.environ["FOO"], "bar")
        self.assertEqual(os.environ["BAZ"], "qux")

    def test_shell_variables_win_by_default(self):
        """A stale checked-out .env must never clobber an exported secret."""
        from app.env import load_dotenv
        os.environ["FOO"] = "from_shell"
        load_dotenv(self.write("FOO=from_file\n"))
        self.assertEqual(os.environ["FOO"], "from_shell")

    def test_override_is_opt_in(self):
        from app.env import load_dotenv
        os.environ["FOO"] = "from_shell"
        load_dotenv(self.write("FOO=from_file\n"), override=True)
        self.assertEqual(os.environ["FOO"], "from_file")

    def test_missing_file_is_not_an_error(self):
        from app.env import load_dotenv
        self.assertEqual(load_dotenv(Path(self.dir) / "nope.env"), [])

    def test_values_containing_equals_survive(self):
        from app.env import load_dotenv
        load_dotenv(self.write("DATABASE_URL=postgresql://u:p==@h/db\n"))
        self.assertEqual(os.environ["DATABASE_URL"], "postgresql://u:p==@h/db")


class TestPlaceholderDetection(unittest.TestCase):
    """An unedited placeholder returns 401, which reads as 'bad key'."""

    def reason(self, value):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preflight", Path(__file__).resolve().parents[1] / "scripts"
            / "preflight.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.placeholder_reason(value)

    def test_angle_brackets_are_caught(self):
        r = self.reason("<0123456789abcdef0123456789abcdef>")
        self.assertIsNotNone(r)
        self.assertIn("angle brackets", r)

    def test_template_values_are_caught(self):
        for value in ("your_key_here", "changeme_changeme_changeme",
                      "xxxxxxxxxxxxxxxxxxxx"):
            with self.subTest(value=value):
                self.assertIsNotNone(self.reason(value))

    def test_short_values_are_caught(self):
        self.assertIsNotNone(self.reason("abc123"))

    def test_whitespace_is_caught(self):
        self.assertIsNotNone(self.reason(" a" * 20))

    def test_a_plausible_key_passes(self):
        self.assertIsNone(self.reason("f" * 32))
