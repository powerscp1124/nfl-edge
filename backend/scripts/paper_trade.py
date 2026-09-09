#!/usr/bin/env python3
"""Paper-trade loop: record projections before kickoff, settle them after.

    python scripts/paper_trade.py record --season 2025      # the upcoming slate
    python scripts/paper_trade.py settle --season 2026      # grade what has played
    python scripts/paper_trade.py report

Nothing here places a bet, and there is no flag that would. The model has no
demonstrated edge -- see the README -- so the purpose is evidence, not action.

Every constant in the model was fitted against 2023-2025, which makes a season
recorded forward the only test that cannot be contaminated by that fitting.
The number to watch is the calibration slope from `scripts/calibrate_probs.py`:
it separates signal from noise in hundreds of bets, where ROI needs thousands.

`--season` is the season whose *history* builds the projection. For a week 1
game that is the previous completed season; the current one has no data yet.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv  # noqa: E402

load_dotenv()

from app.backtest.runner import (  # noqa: E402
    MARKET_STAT_COLUMNS,
    grade,
    select,
)
from app.backtest.engine import BacktestConfig  # noqa: E402
from app.core.edge import ConfidenceInputs, evaluate_prop  # noqa: E402
from app.core.odds import expected_value  # noqa: E402
from app.ingest.context_loader import fetch_context  # noqa: E402
from app.ingest.odds_api import OddsAPIClient, normalize_event_props  # noqa: E402
from app.ingest.player_matching import (  # noqa: E402
    PlayerResolver,
    RosterEntry,
    is_non_player_market,
)
from app.paper.journal import (  # noqa: E402
    DEFAULT_PATH,
    Pick,
    journal,
    record,
    settle,
    settled_rows,
    summary,
    unsettled,
)
from app.projections.pipeline import (  # noqa: E402
    anchored_distribution,
    build_environment,
    build_rosters,
    group_quotes,
)
from app.sim.game_sim import simulate_game  # noqa: E402


def _kickoff_of(event: dict):
    try:
        return datetime.fromisoformat(
            str(event.get("commence_time")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def do_record(args) -> int:
    client = OddsAPIClient()
    events = client.list_events()
    if args.event_id:
        events = [e for e in events if e.get("id") == args.event_id]
    elif args.within_days:
        # The events endpoint returns most of the season, not the coming week,
        # so a weekly job without a horizon would simulate 270-odd games.
        horizon = datetime.now(timezone.utc) + timedelta(days=args.within_days)
        events = [e for e in events
                  if (e.get("commence_time") or "")
                  and _kickoff_of(e) is not None
                  and _kickoff_of(e) <= horizon]
    if args.limit:
        events = events[:args.limit]
    if not events:
        print("No upcoming events to record.")
        return 0
    print(f"{len(events)} upcoming event(s); history season {args.season}")

    config = BacktestConfig(seasons=[args.season], min_edge=args.min_edge,
                            min_confidence=args.min_confidence)
    now = datetime.now(timezone.utc)
    total = 0
    with journal(args.db) as conn:
        for event in events:
            eid = event.get("id")
            kickoff = event.get("commence_time") or ""
            try:
                picks = project_event(client, eid, event, args, config, now)
            except Exception as exc:  # noqa: BLE001
                print(f"  {eid[:8]} {event.get('away_team')} @ "
                      f"{event.get('home_team')}: {type(exc).__name__}: {exc}")
                continue
            written = record(conn, picks)
            backed = sum(1 for p in picks if p.would_bet)
            total += written
            print(f"  {event.get('away_team')} @ {event.get('home_team')}"
                  f"  {kickoff[:16]}  {written} new row(s), "
                  f"{backed} would be backed")
    print(f"\nRecorded {total} new projection(s). No wager was placed.")
    return 0


def project_event(client, event_id, event, args, config, now) -> list[Pick]:
    payload = client.event_props(event_id)
    quotes = normalize_event_props(payload)
    built = fetch_context(event_id, season=args.season)
    context = built.context
    roster = [RosterEntry(**r) for r in context["roster"]]
    resolver = PlayerResolver(roster)
    names = {(q.player_name, None) for q in quotes
             if not is_non_player_market(q.player_name)}
    resolved, _ = resolver.resolve_many(names) if names else ({}, [])

    rosters, _notes, role_changes = build_rosters(
        context, resolver, market_players=frozenset(resolved.values()))
    env = build_environment(context, rosters)
    home, away = context["home_team"], context["away_team"]
    sim = simulate_game(env, rosters[home][0], rosters[away][0],
                        n_sims=args.sims, seed=args.seed)

    name_of = {r.player_id: r.full_name for r in roster}
    pos_of = {r.player_id: r.position for r in roster}
    team_of = {r.player_id: r.team for r in roster}
    injury_unc = {t: r.injury_uncertainty for t, (_, r) in rosters.items()}

    grouped = group_quotes(quotes, resolved)
    edges = []
    for (pid, stat, _side) in list(grouped):
        over = grouped.get((pid, stat, "over"), [])
        under = grouped.get((pid, stat, "under"), [])
        if not over and not under:
            continue
        try:
            dist = sim.distribution(pid, stat)
        except KeyError:
            continue
        dist = anchored_distribution(dist, over, under)
        team = team_of.get(pid, home)
        edges.extend(evaluate_prop(
            player_id=pid, player_name=name_of.get(pid, pid), team=team,
            opponent=away if team == home else home,
            position=pos_of.get(pid, "WR"), market=stat, distribution=dist,
            over_quotes=over, under_quotes=under,
            confidence_inputs=ConfidenceInputs(
                projection_quality=0.62,
                injury_certainty=1.0 - injury_unc.get(team, 0.0),
                role_certainty=0.55 if role_changes.get(pid) else 0.80)))

    # Best price per player/market/side, as the slice does.
    best: dict[tuple, object] = {}
    for e in edges:
        key = (e.player_id, e.market, e.side)
        if key not in best or e.ev > best[key].ev:
            best[key] = e
    chosen = {id(e) for e in select(list(best.values()), config)}

    stamp = now.isoformat()
    return [Pick(
        decision_at=stamp, history_season=args.season, event_id=event_id,
        game_id=context.get("game_id"), kickoff=event.get("commence_time") or "",
        player_id=e.player_id, player_name=e.player_name, team=e.team,
        opponent=e.opponent, market=e.market, side=e.side, line=e.line,
        bookmaker=e.bookmaker, american=e.american, model_prob=e.model_prob,
        market_prob=e.market_prob, edge=e.edge, confidence=e.confidence,
        recommendation=e.recommendation, projection=e.projection,
        would_bet=id(e) in chosen,
    ) for e in best.values()]


def do_settle(args) -> int:
    from app.ingest.nflverse import load_weekly_stats
    try:
        weekly = load_weekly_stats([args.season])
    except Exception as exc:  # noqa: BLE001
        # nflverse publishes a season's stats only once games have been played,
        # so early in a season the file does not exist yet. That is the normal
        # state of a weekly job in September, not an error: say so and leave
        # the picks pending rather than aborting before anything is recorded.
        print(f"No outcome data for {args.season} yet "
              f"({type(exc).__name__}). Nothing settled; picks stay pending.")
        return 0
    if weekly.empty:
        print(f"No outcome data for {args.season} yet. "
              "Nothing settled; picks stay pending.")
        return 0
    id_col = "player_id" if "player_id" in weekly.columns else "gsis_id"
    actuals: dict[tuple, dict] = {}
    for _, r in weekly.iterrows():
        key = (str(r[id_col]), int(r["week"]))
        actuals[key] = {m: float(r.get(c) or 0.0)
                        for m, c in MARKET_STAT_COLUMNS.items()
                        if c in weekly.columns}

    graded = skipped = 0
    with journal(args.db) as conn:
        rows = unsettled(conn)
        print(f"{len(rows)} pick(s) past kickoff and awaiting settlement")
        for row in rows:
            week = row["week"] or args.week
            if week is None:
                skipped += 1
                continue
            stat = actuals.get((row["player_id"], int(week)), {}).get(row["market"])
            if stat is None:
                skipped += 1      # did not play: void rather than score it
                continue
            result = grade(row["side"], row["line"], stat)
            if result == "void":
                skipped += 1
                continue
            if not row["would_bet"] or result == "push":
                pnl = 0.0
            elif result == "win":
                pnl = expected_value(1.0, row["american"], 1.0)
            else:
                pnl = -1.0
            settle(conn, row["id"], stat, result, pnl)
            graded += 1
    print(f"Settled {graded}, skipped {skipped} (no stat line, or no week).")
    return 0


def do_report(args) -> int:
    with journal(args.db) as conn:
        counts = summary(conn)
        print("=== Journal ===")
        for k in ("recorded", "would_bet", "settled", "pending"):
            print(f"  {k:12} {counts.get(k, 0)}")
        rows = settled_rows(conn)
        bets = [r for r in rows if r["would_bet"]]
        if not rows:
            print("\nNothing settled yet.")
            return 0

        print("\n=== All settled projections (calibration population) ===")
        show(rows)
        if bets:
            print("\n=== Paper wagers only (what the thresholds backed) ===")
            show(bets, with_pnl=True)
        else:
            print("\nNo projection cleared the thresholds. With an anchored "
                  "model that is the expected result, not a failure.")
    return 0


def show(rows, with_pnl: bool = False) -> None:
    probs = np.array([r["model_prob"] for r in rows], dtype=float)
    won = np.array([1.0 if r["result"] == "win" else 0.0 for r in rows])
    print(f"  n = {len(rows)}   beat-the-line rate = "
          f"{100 * np.mean([1.0 if r['actual'] > r['line'] else 0.0 for r in rows]):.1f}%")
    print(f"  Brier = {np.mean((probs - won) ** 2):.5f}   "
          f"(a flat 50% scores 0.25000)")
    if with_pnl:
        pnl = float(np.sum([r["pnl"] or 0.0 for r in rows]))
        print(f"  paper P&L = {pnl:+.2f} units over {len(rows)} wagers "
              f"({100 * pnl / len(rows):+.2f}% ROI)   win rate "
              f"{100 * won.mean():.1f}%")
    print("  These are paper results. No wager was placed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("record", "settle", "report"))
    ap.add_argument("--season", type=int,
                    help="season whose history builds the projection")
    ap.add_argument("--week", type=int, default=None,
                    help="fallback week for settlement")
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--within-days", type=int, default=8,
                    help="only record games kicking off within this many days "
                         "(the events endpoint returns most of the season)")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--min-confidence", type=float, default=60.0)
    ap.add_argument("--db", default=str(DEFAULT_PATH))
    args = ap.parse_args()
    if args.mode in ("record", "settle") and args.season is None:
        ap.error("--season is required for record and settle")
    return {"record": do_record, "settle": do_settle,
            "report": do_report}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
