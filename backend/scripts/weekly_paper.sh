#!/usr/bin/env bash
# Weekly paper-trade job: settle what has played, record what is coming.
#
#   ./scripts/weekly_paper.sh            # uses the defaults below
#   HISTORY_SEASON=2026 ./scripts/weekly_paper.sh
#
# Safe to run repeatedly: recording is idempotent per calendar day, and
# settlement skips rows already graded. Nothing here places a wager.
#
# Note the two seasons. HISTORY_SEASON is the season whose data builds the
# projection; OUTCOME_SEASON is where the results land. Early in 2026 those
# differ, because the current season has no history yet. Once 2026 has enough
# weeks behind it, set HISTORY_SEASON=2026.
set -euo pipefail

cd "$(dirname "$0")/.."

HISTORY_SEASON="${HISTORY_SEASON:-2025}"
OUTCOME_SEASON="${OUTCOME_SEASON:-2026}"
JOURNAL="${JOURNAL:-$(pwd)/../paper_journal.sqlite3}"
LOG_DIR="${LOG_DIR:-$(pwd)/../paper_logs}"
PY="$(pwd)/.venv/bin/python"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(date -u +%Y-%m-%d).log"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "journal: $JOURNAL"

  # Settlement must never block recording. If outcomes are unavailable -- no
  # nflverse file yet in September, or a transient fetch failure -- the picks
  # simply stay pending and get graded next week. Losing a week of *recording*
  # is unrecoverable; losing a week of settling is not.
  echo "--- settle (outcomes from $OUTCOME_SEASON) ---"
  "$PY" scripts/paper_trade.py settle --season "$OUTCOME_SEASON" \
      --db "$JOURNAL" || echo "settle failed; continuing to record"

  echo "--- record (history from $HISTORY_SEASON) ---"
  "$PY" scripts/paper_trade.py record --season "$HISTORY_SEASON" --db "$JOURNAL"

  echo "--- report ---"
  "$PY" scripts/paper_trade.py report --db "$JOURNAL"
  echo
} 2>&1 | tee -a "$LOG"

echo "logged to $LOG"
