#!/usr/bin/env python3
"""Replay past games through the live projection path and score the result.

    python scripts/backtest.py --season 2025 --weeks 5-8 --dry-run
    python scripts/backtest.py --season 2025 --weeks 5-8

``--dry-run`` costs nothing and touches no odds endpoint: it builds every
game's context and projections from nflverse alone and reports what the run
*would* do. Use it to prove the wiring before spending credits.

Two things this deliberately does not do:

* It does not grade ``needs_review`` or ``pass`` selections. Those are not
  wagers, and counting them would report the record of a strategy the shipped
  thresholds forbid.
* It does not read a stat line before the decision timestamp. Projections are
  rebuilt with ``through_week`` set to the game's own week, so the model sees
  the same history it would have seen on the day.

Known compromise, printed at run time: the spread and total come from the
nflverse schedule, which stores *closing* numbers. The player projection never
sees them until after its own history cutoff, but they are the one input that
is not strictly as-of. Wire historical game odds if that matters for your use.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv  # noqa: E402

load_dotenv()

from app.backtest.engine import BacktestConfig  # noqa: E402
from app.backtest.runner import (  # noqa: E402
    MARKET_STAT_COLUMNS,
    actual_values,
    run_backtest,
)
from app.core.edge import ConfidenceInputs, evaluate_prop  # noqa: E402
from app.ingest.context_loader import build_context  # noqa: E402
from app.ingest.odds_api import (  # noqa: E402
    OddsAPIClient,
    normalize_event_props,
)
from app.ingest.player_matching import (  # noqa: E402
    PlayerResolver,
    RosterEntry,
    is_non_player_market,
)
from app.projections.pipeline import (  # noqa: E402
    anchored_distribution,
    build_environment,
    build_rosters,
    group_quotes,
)
from app.ingest.teams import canonical_team  # noqa: E402
from app.sim.game_sim import simulate_game  # noqa: E402


def parse_weeks(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(w) for w in spec.split(",")]


def schedule_games(schedules: pd.DataFrame, season: int,
                   weeks: list[int]) -> list[dict]:
    """Completed games only: an unplayed game has nothing to grade against."""
    df = schedules[(schedules["season"] == season)
                   & (schedules["week"].isin(weeks))]
    games = []
    for _, r in df.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            continue
        kickoff = _kickoff(r)
        if kickoff is None:
            continue
        games.append({
            "game_id": str(r["game_id"]),
            "season": int(r["season"]),
            "week": int(r["week"]),
            "kickoff": kickoff,
            # Schedules carry provider spellings and pre-relocation codes
            # ("LA" for the Rams); everything downstream is keyed by the
            # canonical code.
            "home": canonical_team(str(r["home_team"])),
            "away": canonical_team(str(r["away_team"])),
            # nflverse stores the spread from the home side with a positive
            # number meaning the home team is favoured; the model uses the
            # betting convention, where a favourite carries a negative number.
            "spread_home": (None if pd.isna(r.get("spread_line"))
                            else -float(r["spread_line"])),
            "total": (None if pd.isna(r.get("total_line"))
                      else float(r["total_line"])),
        })
    return games


def _kickoff(row) -> datetime | None:
    day, time = row.get("gameday"), row.get("gametime")
    if pd.isna(day):
        return None
    stamp = f"{day} {time if not pd.isna(time) else '13:00'}"
    try:
        return (pd.to_datetime(stamp)
                .tz_localize("US/Eastern", ambiguous=True)
                .tz_convert("UTC").to_pydatetime())
    except Exception:  # noqa: BLE001
        return None


def synthetic_game_lines(game: dict, event_id: str) -> list[dict]:
    """The schedule's closing spread and total, shaped like an odds payload.

    Built in the payload's own shape so ``extract_market_lines`` stays the one
    place that reads a spread, rather than growing a second path that could
    drift from it.
    """
    return [{
        "id": event_id,
        "home_team": game["home"],
        "away_team": game["away"],
        "bookmakers": [{
            "key": "nflverse_schedule",
            "markets": [
                {"key": "spreads", "outcomes": [
                    {"name": game["home"], "point": game["spread_home"]}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": game["total"]}]},
            ],
        }],
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", default="1-18")
    parser.add_argument("--offset-minutes", type=int, default=1440)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=60.0)
    parser.add_argument("--staking", choices=("flat", "kelly"), default="flat")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many games")
    parser.add_argument("--clv", action="store_true",
                        help="also fetch each game's closing prices and score "
                             "closing line value (one extra call per game)")
    parser.add_argument("--save-bets", default=None,
                        help="write every graded candidate to this CSV, so "
                             "calibration can be refit without re-simulating")
    parser.add_argument("--dry-run", action="store_true",
                        help="no odds calls; proves the wiring for free")
    parser.add_argument("--no-anchor", action="store_true",
                        help="project from the model alone, ignoring "
                             "the market line")
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    config = BacktestConfig(
        seasons=[args.season],
        decision_offset_minutes=args.offset_minutes,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
        staking=args.staking,
    )

    from app.ingest.nflverse import (
        load_depth_charts,
        load_id_map,
        load_injuries,
        load_play_by_play,
        load_schedules,
        load_snap_counts,
        load_weekly_stats,
    )

    print(f"Loading {args.season} nflverse frames once for the whole run...")
    weekly = load_weekly_stats([args.season])
    pbp = load_play_by_play([args.season])
    snaps = load_snap_counts([args.season])
    depth = load_depth_charts([args.season])
    injuries = load_injuries([args.season])
    id_map = load_id_map()
    schedules = load_schedules([args.season])

    from app.ingest.player_matching import normalize_name
    season_ids = {}
    for _, row in weekly.iterrows():
        label = row.get("player_display_name") or row.get("name")
        if isinstance(label, str):
            season_ids.setdefault(normalize_name(label), str(row["player_id"]))

    games = schedule_games(schedules, args.season, weeks)
    if args.limit:
        games = sorted(games, key=lambda g: g["kickoff"])[:args.limit]
    print(f"{len(games)} completed game(s) in weeks {args.weeks}")
    print("NOTE: spread and total come from the schedule's closing numbers; "
          "every other input is as-of the decision time.")

    client = None if args.dry_run else OddsAPIClient()
    events_by_week: dict[int, dict] = {}

    def event_id_for(game, decided_at):
        if client is None:
            return f"dry-{game['game_id']}"
        week = game["week"]
        if week not in events_by_week:
            slate = client.historical_events(
                decided_at.astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
            events_by_week[week] = {
                (e.get("home_team"), e.get("away_team")): e.get("id")
                for e in slate
            }
        for (home, away), eid in events_by_week[week].items():
            if (canonical_team(home or "", strict=False) == game["home"]
                    and canonical_team(away or "", strict=False)
                    == game["away"]):
                return eid
        raise RuntimeError(f"no historical event for {game['game_id']}")

    def propose(game, decided_at):
        event_id = event_id_for(game, decided_at)
        if client is None:
            quotes = []
        else:
            payload = client.historical_event_props(
                event_id,
                decided_at.astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
            quotes = normalize_event_props(payload.get("data", payload))

        built = build_context(
            event_payload={"id": event_id, "home_team": game["home"],
                           "away_team": game["away"],
                           "commence_time": game["kickoff"].isoformat()},
            game_lines=synthetic_game_lines(game, event_id),
            weekly=weekly, pbp=pbp, snaps=snaps, depth=depth,
            injuries=injuries, id_map=id_map,
            through_week=game["week"], as_of=decided_at,
        )
        context = built.context
        roster = [RosterEntry(**r) for r in context["roster"]]
        resolver = PlayerResolver(roster)
        names = {(q.player_name, None) for q in quotes
                 if not is_non_player_market(q.player_name)}
        resolved, _ = resolver.resolve_many(names) if names else ({}, [])

        rosters, _notes, role_changes = build_rosters(
            context, resolver, market_players=frozenset(resolved.values()))
        env = build_environment(context, rosters)
        sim = simulate_game(env, rosters[game["home"]][0],
                            rosters[game["away"]][0],
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
            if not args.no_anchor:
                # The projection is the market line unless something argues
                # otherwise; see app/projections/anchor.py for why.
                dist = anchored_distribution(dist, over, under)
            team = team_of.get(pid, game["home"])
            edges.extend(evaluate_prop(
                player_id=pid, player_name=name_of.get(pid, pid), team=team,
                opponent=game["away"] if team == game["home"] else game["home"],
                position=pos_of.get(pid, "WR"), market=stat, distribution=dist,
                over_quotes=over, under_quotes=under,
                confidence_inputs=ConfidenceInputs(
                    projection_quality=0.62,
                    injury_certainty=1.0 - injury_unc.get(team, 0.0),
                    role_certainty=0.55 if role_changes.get(pid) else 0.80,
                ),
            ))

        # One wager per player/market/side at the best available price. The
        # slice does this before ranking; without it each book's quote on the
        # same prop is graded as a separate bet, which multiplies the sample
        # by the number of books and counts one outcome a dozen times.
        best: dict[tuple, object] = {}
        for e in edges:
            key = (e.player_id, e.market, e.side)
            if key not in best or e.ev > best[key].ev:
                best[key] = e
        return list(best.values())

    def actuals_for(game):
        return actual_values(weekly, game["season"], game["week"])

    def closing_for(game):
        """Consensus closing price per (player, market, side, line).

        Beating the close is the best short-run evidence that a model has
        found real inefficiency, because unlike win rate it does not depend on
        whether the bets happened to land -- it converges in hundreds of bets
        rather than thousands.

        The **best** closing price is used, not the median, because the bet
        side takes the best available price too. Comparing best-of-N at bet
        time against median-of-N at the close is positively biased with no
        skill involved: the maximum of a sample exceeds its median by
        construction. That artefact produced a spurious +1.24pp of "CLV" that
        had zero correlation with whether the bets won (r = -0.001), and it
        grew as the selection was filtered toward better prices. Like must be
        compared with like.
        """
        if client is None:
            return {}
        try:
            payload = client.historical_event_props(
                event_id_for(game, game["kickoff"]),
                game["kickoff"].astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception:  # noqa: BLE001
            return {}
        prices = {}
        for q in normalize_event_props(payload.get("data", payload)):
            if q.is_alternate or q.stat not in MARKET_STAT_COLUMNS:
                continue
            pid = season_ids.get(normalize_name(q.player_name))
            if pid is None:
                continue
            side = "over" if q.side.startswith("o") else "under"
            prices.setdefault((pid, q.stat, side, float(q.line)),
                              []).append(float(q.american))
        # Most favourable price for the bettor on that side.
        return {k: float(np.max(v)) for k, v in prices.items()}

    def on_game(game, bets, error):
        flag = f"  ERROR {error}" if error else ""
        print(f"  {game['game_id']:22} {len(bets):3d} bet(s){flag}")

    result, bets = run_backtest(games, config, propose=propose,
                                actuals_for=actuals_for, on_game=on_game,
                                closing_for=closing_for if args.clv else None,
                                grade_all=bool(args.save_bets))

    if args.save_bets:
        import csv
        from dataclasses import asdict
        rows = [asdict(b) for b in bets]
        with open(args.save_bets, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])) if rows \
                else None
            if writer:
                writer.writeheader()
                writer.writerows(rows)
        picked = sum(1 for b in bets if b.selected)
        print(f"\nWrote {len(rows)} graded candidate(s) "
              f"({picked} selected) to {args.save_bets}")

    if args.clv:
        scored = sum(1 for b in bets if b.selected and b.clv_available)
        placed = sum(1 for b in bets if b.selected)
        print(f"\nClosing prices matched for {scored}/{placed} bet(s) "
              f"at the same line")

    print("\n=== Result ===")
    for key, value in result.summary().items():
        print(f"  {key:16} {value}")
    if result.calibration:
        print("\n=== Calibration ===")
        print(f"  {'bin':>12} {'n':>5} {'predicted':>10} {'actual':>8}")
        for row in result.calibration:
            if not row.get("n"):
                continue
            label = f"{row['bin_low']:.0%}-{row['bin_high']:.0%}"
            print(f"  {label:>12} {row['n']:5d} "
                  f"{row.get('predicted') or 0:10.3f} "
                  f"{row.get('observed') or 0:8.3f}")
    if result.by_bucket:
        print("\n=== By edge bucket ===")
        for bucket, stats in sorted(result.by_bucket.items()):
            print(f"  {bucket:>10} n={stats['n']:4d} roi={stats['roi']:+.3f} "
                  f"win={stats['win_rate']:.3f} units={stats['units']:+.2f}")
    if result.n_bets == 0:
        print("\n  No qualifying bets. With the thresholds this strict that is "
              "a result, not a failure -- but it is also no evidence either "
              "way about the model.")
    print("\nProjections are probabilistic estimates with real uncertainty. "
          "No wager is guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
