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

    def test_environment_flags_approximation_only_without_pbp(self):
        from app.ingest.context_loader import build_team_environment
        weekly = normalize_weekly(weekly_frame())
        vol = team_volume_by_week(weekly)
        without = build_team_environment(weekly, vol, "MIN")
        with_pbp = build_team_environment(weekly, vol, "MIN", pbp=self.pbp())
        self.assertTrue(without["_pace_is_approximate"])
        self.assertFalse(with_pbp["_pace_is_approximate"])


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
