"""Paper journal tests.

A forward record is only evidence if it cannot be revised after the fact and
cannot double-count. These check both, plus the settlement rules that keep a
player who never took the field from scoring as a winning under.
"""

import tempfile
import unittest
from pathlib import Path

from app.paper.journal import (
    Pick,
    journal,
    record,
    settle,
    settled_rows,
    summary,
    unsettled,
)

KICKOFF = "2026-09-14T17:00:00+00:00"
DECIDED = "2026-09-13T17:00:00+00:00"


def pick(**kw):
    base = dict(decision_at=DECIDED, history_season=2025, event_id="evt1",
                kickoff=KICKOFF, player_id="p1", market="receiving_yards",
                side="over", line=50.5, american=-110, season=2026, week=1,
                model_prob=0.55, would_bet=True)
    base.update(kw)
    return Pick(**base)


class TestJournal(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = Path(self.dir.name) / "j.sqlite3"

    def tearDown(self):
        self.dir.cleanup()

    def test_a_pick_is_recorded(self):
        with journal(self.db) as c:
            self.assertEqual(record(c, [pick()]), 1)
            self.assertEqual(summary(c)["recorded"], 1)

    def test_recording_the_same_slate_twice_does_not_double_count(self):
        """A weekly job that reruns must not inflate the record.

        The rerun carries a *different* wall-clock decision time, which is what
        actually happens in production -- an earlier version of this test held
        that field fixed and so passed while the real job duplicated 32 rows.
        """
        with journal(self.db) as c:
            record(c, [pick(decision_at="2026-09-13T09:00:00+00:00")])
            again = record(c, [pick(decision_at="2026-09-13T17:44:31+00:00")])
            self.assertEqual(again, 0)
            self.assertEqual(summary(c)["recorded"], 1)

    def test_re_recording_on_a_later_day_is_a_new_observation(self):
        """Prices move; a projection made a day later is genuinely new."""
        with journal(self.db) as c:
            record(c, [pick(decision_at="2026-09-12T17:00:00+00:00")])
            record(c, [pick(decision_at="2026-09-13T17:00:00+00:00")])
            self.assertEqual(summary(c)["recorded"], 2)

    def test_a_different_book_or_line_is_a_different_pick(self):
        with journal(self.db) as c:
            record(c, [pick(bookmaker="a"), pick(bookmaker="b"),
                       pick(bookmaker="a", line=52.5)])
            self.assertEqual(summary(c)["recorded"], 3)

    def test_everything_is_stored_not_only_what_would_be_backed(self):
        """Calibration measured only on threshold-clearing bets conditions on
        the model's own confidence."""
        with journal(self.db) as c:
            record(c, [pick(would_bet=True),
                       pick(side="under", would_bet=False)])
            s = summary(c)
            self.assertEqual(s["recorded"], 2)
            self.assertEqual(s["would_bet"], 1)

    def test_only_games_past_kickoff_are_settleable(self):
        with journal(self.db) as c:
            record(c, [pick()])
            self.assertEqual(len(unsettled(c, before="2026-09-13T00:00:00")), 0)
            self.assertEqual(len(unsettled(c, before="2026-09-15T00:00:00")), 1)

    def test_settling_marks_the_row_and_stops_reoffering_it(self):
        with journal(self.db) as c:
            record(c, [pick()])
            row = unsettled(c, before="2026-09-15T00:00:00")[0]
            settle(c, row["id"], actual=61.0, result="win", pnl=0.909)
            self.assertEqual(len(unsettled(c, before="2026-09-15T00:00:00")), 0)
            self.assertEqual(summary(c)["settled"], 1)

    def test_settled_rows_can_be_filtered_to_actual_wagers(self):
        with journal(self.db) as c:
            record(c, [pick(bookmaker="a", would_bet=True),
                       pick(bookmaker="b", would_bet=False)])
            for row in unsettled(c, before="2026-09-15T00:00:00"):
                settle(c, row["id"], 61.0, "win",
                       0.909 if row["would_bet"] else 0.0)
            self.assertEqual(len(settled_rows(c)), 2)
            self.assertEqual(len(settled_rows(c, only_bets=True)), 1)

    def test_a_recorded_projection_is_never_revised(self):
        """Re-recording with a different probability must not overwrite the
        original: a paper record that can be edited afterwards is not
        evidence."""
        with journal(self.db) as c:
            record(c, [pick(model_prob=0.55)])
            record(c, [pick(model_prob=0.99)])
            row = list(c.execute("SELECT model_prob FROM picks"))[0]
            self.assertAlmostEqual(row["model_prob"], 0.55)


if __name__ == "__main__":
    unittest.main()
