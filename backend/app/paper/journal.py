"""An append-only journal of projections made before kickoff.

The model has no demonstrated edge -- four full-season backtests returned
between +0.35% and -7.65%, and the projection adds no explanatory power over
the book's line. So this records what it *would* have done and settles it
afterwards. Nothing here places a wager, and there is deliberately no code
path that could.

The point is out-of-sample evidence. Every constant in the model was fitted
against 2023-2025; a season recorded forward is the only test that cannot be
contaminated by that. Watch the calibration slope rather than ROI -- it moved
measurably across changes ROI could not separate from noise on 2,000 bets.

SQLite rather than Postgres on purpose: `db/schema.sql` has never been applied
to a live database, and a journal that needs a server standing up is a journal
that does not get written.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at     TEXT    NOT NULL,
    decision_at     TEXT    NOT NULL,
    -- The calendar day the projection was made. Uniqueness keys on this
    -- rather than the exact timestamp: a weekly job that retries after a
    -- failure must not journal the slate twice, and wall-clock time makes
    -- every rerun a fresh row. Re-recording on a later day is a genuinely
    -- new observation and is allowed.
    decision_day    TEXT    NOT NULL,
    history_season  INTEGER NOT NULL,
    event_id        TEXT    NOT NULL,
    game_id         TEXT,
    kickoff         TEXT    NOT NULL,
    season          INTEGER,
    week            INTEGER,
    player_id       TEXT    NOT NULL,
    player_name     TEXT,
    team            TEXT,
    opponent        TEXT,
    market          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    line            REAL    NOT NULL,
    bookmaker       TEXT,
    american        REAL    NOT NULL,
    model_prob      REAL,
    market_prob     REAL,
    edge            REAL,
    confidence      REAL,
    recommendation  TEXT,
    projection      REAL,
    -- Whether the shipped thresholds would have backed it. Everything the
    -- model evaluated is stored, not only what it liked: calibration measured
    -- on the subset that cleared a threshold conditions on the model's own
    -- confidence and hides the overconfidence being looked for.
    would_bet       INTEGER NOT NULL DEFAULT 0,
    settled         INTEGER NOT NULL DEFAULT 0,
    actual          REAL,
    result          TEXT,
    pnl             REAL,
    UNIQUE (event_id, player_id, market, side, line, bookmaker, decision_day)
);
CREATE INDEX IF NOT EXISTS picks_unsettled ON picks (settled, kickoff);
CREATE INDEX IF NOT EXISTS picks_lookup ON picks (season, week, player_id);
"""

DEFAULT_PATH = Path("paper_journal.sqlite3")


@dataclass
class Pick:
    """One projection, recorded before kickoff."""

    decision_at: str
    history_season: int
    event_id: str
    kickoff: str
    player_id: str
    market: str
    side: str
    line: float
    american: float
    player_name: str | None = None
    team: str | None = None
    opponent: str | None = None
    bookmaker: str | None = None
    game_id: str | None = None
    season: int | None = None
    week: int | None = None
    model_prob: float | None = None
    market_prob: float | None = None
    edge: float | None = None
    confidence: float | None = None
    recommendation: str | None = None
    projection: float | None = None
    would_bet: bool = False


@contextmanager
def journal(path: str | Path = DEFAULT_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record(conn: sqlite3.Connection, picks: Iterable[Pick]) -> int:
    """Write picks, ignoring ones already journalled.

    Idempotent by (event, player, market, side, line, book, decision time), so
    re-running a slate cannot double-count. A recorded projection is never
    revised: revising one after the fact is how a paper record stops being
    evidence.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [(
        now, p.decision_at, (p.decision_at or now)[:10], p.history_season, p.event_id, p.game_id, p.kickoff,
        p.season, p.week, p.player_id, p.player_name, p.team, p.opponent,
        # SQLite treats NULLs as distinct in a UNIQUE constraint, so a missing
        # bookmaker would defeat deduplication and let a rerun of the weekly
        # job silently inflate the record.
        p.market, p.side, float(p.line), p.bookmaker or "", float(p.american),
        p.model_prob, p.market_prob, p.edge, p.confidence, p.recommendation,
        p.projection, int(bool(p.would_bet)),
    ) for p in picks]
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO picks (
               recorded_at, decision_at, decision_day, history_season,
               event_id, game_id,
               kickoff, season, week, player_id, player_name, team, opponent,
               market, side, line, bookmaker, american, model_prob,
               market_prob, edge, confidence, recommendation, projection,
               would_bet)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    return conn.total_changes - before


def unsettled(conn: sqlite3.Connection, before: str | None = None
              ) -> list[sqlite3.Row]:
    """Recorded picks whose game has kicked off but which are not yet graded."""
    cutoff = before or datetime.now(timezone.utc).isoformat()
    return list(conn.execute(
        "SELECT * FROM picks WHERE settled = 0 AND kickoff < ? "
        "ORDER BY kickoff", (cutoff,)))


def settle(conn: sqlite3.Connection, pick_id: int, actual: float,
           result: str, pnl: float) -> None:
    conn.execute(
        "UPDATE picks SET settled = 1, actual = ?, result = ?, pnl = ? "
        "WHERE id = ?", (float(actual), result, float(pnl), int(pick_id)))


def settled_rows(conn: sqlite3.Connection, only_bets: bool = False
                 ) -> list[sqlite3.Row]:
    sql = "SELECT * FROM picks WHERE settled = 1 AND result IN ('win','loss')"
    if only_bets:
        sql += " AND would_bet = 1"
    return list(conn.execute(sql))


def summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) AS recorded,
                  SUM(would_bet) AS would_bet,
                  SUM(settled) AS settled,
                  SUM(CASE WHEN settled = 0 THEN 1 ELSE 0 END) AS pending
           FROM picks""").fetchone()
    return {k: (row[k] or 0) for k in row.keys()}
