#!/usr/bin/env python3
"""Harvest historical prop lines and their outcomes, without simulating.

    python scripts/harvest_lines.py --season 2024 --out lines-2024.csv

Signal testing asks whether some fact predicts a player beating his line. That
needs the line, the outcome, and the fact -- not a projection. Running the full
simulator to obtain them costs two hours a season and contributes nothing to
the answer, so this fetches the same historical odds the backtest would and
joins them straight to the settled stat line.

Player resolution is deliberately coarse here: names are matched against the
whole season's player index rather than a per-game roster, because for this
purpose an unmatched name is a dropped row, not a mispriced bet.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv  # noqa: E402

load_dotenv()

from app.backtest.engine import decision_timestamp  # noqa: E402
from app.ingest.odds_api import (  # noqa: E402
    OddsAPIClient,
    normalize_event_props,
)
from app.ingest.player_matching import (  # noqa: E402
    is_non_player_market,
    normalize_name,
)
from app.ingest.teams import canonical_team  # noqa: E402

MARKETS = {"passing_yards", "rushing_yards", "receiving_yards"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset-minutes", type=int, default=1440)
    ap.add_argument("--weeks", default="1-22")
    args = ap.parse_args()

    lo, hi = (args.weeks.split("-") + [args.weeks])[:2]
    weeks = set(range(int(lo), int(hi) + 1))

    from app.ingest.nflverse import load_schedules, load_weekly_stats
    weekly = load_weekly_stats([args.season])
    sched = load_schedules([args.season])

    # Season-wide name index, and the settled stat lines to join against.
    index, actuals = {}, {}
    for _, r in weekly.iterrows():
        pid = str(r["player_id"])
        name = r.get("player_display_name") or r.get("name")
        if isinstance(name, str):
            index.setdefault(normalize_name(name), pid)
        actuals[(pid, int(r["week"]))] = {
            "passing_yards": float(r.get("passing_yards") or 0),
            "rushing_yards": float(r.get("rushing_yards") or 0),
            "receiving_yards": float(r.get("receiving_yards") or 0),
            "team": canonical_team(str(r["team"]), strict=False)
            if not pd.isna(r.get("team")) else None,
        }

    client = OddsAPIClient()
    games = sched[(sched["season"] == args.season)
                  & (sched["week"].isin(weeks))].dropna(subset=["home_score"])
    rows, by_week, seen_games, unmatched = [], {}, 0, 0
    for _, g in games.sort_values(["week"]).iterrows():
        week = int(g["week"])
        kickoff = pd.to_datetime(f"{g['gameday']} {g.get('gametime') or '13:00'}")
        kickoff = kickoff.tz_localize("US/Eastern",
                                      ambiguous=True).tz_convert("UTC")
        decided = decision_timestamp(kickoff.to_pydatetime(),
                                     args.offset_minutes)
        stamp = decided.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if week not in by_week:
            try:
                slate = client.historical_events(stamp)
            except Exception as exc:  # noqa: BLE001
                print(f"  week {week}: events failed ({exc})")
                by_week[week] = {}
                slate = []
            by_week[week] = {
                (canonical_team(e.get("home_team") or "", strict=False),
                 canonical_team(e.get("away_team") or "", strict=False)):
                e.get("id") for e in slate}
        home = canonical_team(str(g["home_team"]))
        away = canonical_team(str(g["away_team"]))
        event_id = by_week[week].get((home, away))
        if not event_id:
            continue
        try:
            payload = client.historical_event_props(event_id, stamp)
        except Exception as exc:  # noqa: BLE001
            print(f"  {g['game_id']}: props failed ({exc})")
            continue
        quotes = normalize_event_props(payload.get("data", payload))
        seen_games += 1
        best = {}
        for q in quotes:
            if q.is_alternate or q.stat not in MARKETS: continue
            if is_non_player_market(q.player_name): continue
            pid = index.get(normalize_name(q.player_name))
            if pid is None:
                unmatched += 1
                continue
            best.setdefault((pid, q.stat), []).append(q.line)
        for (pid, market), lines in best.items():
            settled = actuals.get((pid, week))
            if settled is None: continue
            rows.append({
                "season": args.season, "week": week,
                "game_id": str(g["game_id"]), "player_id": pid,
                "market": market,
                "line": float(pd.Series(lines).median()),
                "actual": settled[market], "team": settled["team"],
                "home": home, "away": away,
            })
        if seen_games % 20 == 0:
            print(f"  {seen_games} games, {len(rows)} lines")

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"{args.season}: {seen_games} games, {len(rows)} player-market lines "
          f"({unmatched} names unmatched) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
