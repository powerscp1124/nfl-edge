#!/usr/bin/env python3
"""Vertical slice: one game, odds to ranked edges, printed to console.

Runs the entire chain that the production job will run, with no database and no
API server, so every unvalidated assumption is exercised at once:

    odds -> player resolution -> usage -> injuries -> simulation
         -> distributions -> de-vig -> edge -> ranking

Two modes:

    --fixture             Runs against a checked-in payload shaped like a real
                          Odds API response. Verifies the plumbing offline and
                          costs no API credits.

    --event-id <id>       Runs against the live API. Needs ODDS_API_KEY.

The point of the fixture mode is that when you first run this live, the only
thing that can be wrong is the *data* — the wiring is already proven. Watch two
numbers on the live run: the player match rate, and the distribution of edges.
Match rate below about 90% means the resolver needs work before anything else
matters. Edges clustered above 10 points mean the priors are wrong, not that
you have found a market inefficiency.

    python scripts/slice.py --fixture
    python scripts/slice.py --event-id abc123 --sims 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv  # noqa: E402

# Load .env before anything reads os.environ. Shell variables always win.
_loaded = load_dotenv()

from app.core.edge import (  # noqa: E402
    BetThresholds,
    ConfidenceInputs,
    evaluate_prop,
    rank_edges,
)
from app.core.odds import format_american  # noqa: E402
from app.ingest.odds_api import normalize_event_props  # noqa: E402
from app.ingest.player_matching import (  # noqa: E402
    PlayerResolver,
    is_non_player_market,
    normalize_name,
    RosterEntry,
    match_rate,
)
from app.projections.pipeline import (  # noqa: E402
    anchored_distribution,
    MARKET_LABELS,
    build_environment,
    build_rosters,
    group_quotes,
)
from app.sim.game_sim import simulate_game  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

DISCLAIMER = (
    "Projections are probabilistic estimates with real uncertainty. "
    "No wager is guaranteed."
)


# --------------------------------------------------------------------------- #
def load_payload(args) -> dict:
    if args.fixture:
        path = FIXTURE_DIR / "event_odds_sample.json"
        if not path.exists():
            raise SystemExit(f"Fixture missing at {path}")
        return json.loads(path.read_text())

    from app.ingest.odds_api import OddsAPIClient

    client = OddsAPIClient()
    payload = client.event_props(args.event_id)
    print(f"API credits remaining: {client.quota.remaining} "
          f"(this call cost {client.quota.last_cost})")
    return payload


def load_context(args) -> dict:
    """Roster, game logs, market lines, injuries.

    Fixture mode reads a file. Live mode builds the same structure from
    nflverse and The Odds API through ``fetch_context``. Both paths produce an
    identical dict, so everything downstream is provider-agnostic.
    """
    if args.fixture:
        return json.loads((FIXTURE_DIR / "game_context_sample.json").read_text())

    from app.ingest.context_loader import fetch_context

    built = fetch_context(args.event_id, season=args.season,
                          through_week=args.through_week)
    for warning in built.warnings:
        print(f"  context warning  {warning}")
    return built.context


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--through-week", type=int, default=None,
        help="Backtest guard: use only data from before this week.")
    parser.add_argument("--no-anchor", action="store_true",
                        help="project from the model alone, "
                             "ignoring the market line")
    args = parser.parse_args()

    if not args.fixture and not args.event_id:
        parser.error("pass --fixture or --event-id")

    payload = load_payload(args)
    context = load_context(args)
    quotes = normalize_event_props(payload)
    print(f"\nParsed {len(quotes)} odds rows from "
          f"{len({q.bookmaker for q in quotes})} books")

    # --- 1. Player resolution ------------------------------------------- #
    roster = [RosterEntry(**r) for r in context["roster"]]
    resolver = PlayerResolver(roster)
    team_of = {r.full_name: r.team for r in roster}
    market_names = {q.player_name for q in quotes}
    team_markets = {n for n in market_names if is_non_player_market(n)}
    names = {(n, None) for n in market_names - team_markets}
    resolved, unresolved = resolver.resolve_many(names)

    # A resolved player with no game log is a data gap, not a matching
    # failure. Counting the two together hides both.
    with_history = {pid for team in context["game_logs"].values()
                    for pid in team}
    no_history = {n: pid for n, pid in resolved.items()
                  if pid not in with_history}

    rate = match_rate(resolved, unresolved)
    print(f"Player resolution: {len(resolved)}/{len(names)} matched "
          f"({rate:.1%})")
    if team_markets:
        print(f"  {len(team_markets)} team/novelty market(s) skipped: "
              f"{', '.join(sorted(team_markets)[:3])}"
              f"{' ...' if len(team_markets) > 3 else ''}")
    if no_history:
        print(f"  {len(no_history)} matched but have no game log this season "
              f"(cannot be projected): {', '.join(sorted(no_history)[:4])}"
              f"{' ...' if len(no_history) > 4 else ''}")
    league = context.get("league_players", {})
    churn, missing = [], []
    for u in unresolved:
        elsewhere = league.get(normalize_name(u.raw_name))
        (churn if elsewhere else missing).append((u, elsewhere))
    for u, elsewhere in churn:
        print(f"  ROSTER CHANGE  {u.raw_name:24} played for {elsewhere} in "
              "this season's data; not on either roster here")
    for u, _ in missing:
        print(f"  UNRESOLVED  {u.raw_name:24} [{u.method}] "
              f"not in this season's data at all "
              f"(candidates={[c.split(' (')[0] for c in u.candidates[:2]]})")
    if rate < 0.90:
        print("  ! Match rate below 90%. Fix the resolver before trusting "
              "any edge on this slate.")

    # --- 2. Rosters, injuries, environment ------------------------------- #
    rosters, notes, role_changes = build_rosters(
        context, resolver, market_players=frozenset(resolved.values()))
    env = build_environment(context, rosters)
    home, away = context["home_team"], context["away_team"]

    print(f"\n{away} @ {home}   spread {env.spread_home:+g}   "
          f"total {env.total:g}")
    ih, ia = env.implied_totals()
    print(f"Implied totals: {home} {ih:.1f}, {away} {ia:.1f}")
    w = env.weather
    print("Weather: dome" if w.is_dome else
          f"Weather: {w.temperature_f:.0f}F, wind {w.wind_mph:.0f}mph")

    for team, (_, result) in rosters.items():
        for change in result.changes:
            print(f"  injury shift  {team} {change['player_id']:16} "
                  f"{change['attribute']} {change['before']:.3f} -> "
                  f"{change['after']:.3f}")
    for pid, c in role_changes.items():
        print(f"  role change   {pid:16} target share "
              f"{c['baseline']:.1%} -> {c['recent']:.1%} (z={c['z_score']:.1f})")
    for note in notes:
        print(f"  data note     {note}")

    # --- 3. Simulate ------------------------------------------------------ #
    sim = simulate_game(env, rosters[home][0], rosters[away][0],
                        n_sims=args.sims, seed=args.seed)
    print(f"\nSimulated {sim.n_sims:,} games")
    for warn in sim.warnings:
        print(f"  ! {warn}")
    for team in (home, away):
        print(f"  {team}: {np.mean(sim.team_points[team]):.1f} pts, "
              f"{np.mean(sim.team_plays[team]):.1f} plays, "
              f"{np.mean(sim.team_pass_attempts[team]):.1f} att, "
              f"{np.mean(sim.team_rush_attempts[team]):.1f} rush")

    # --- 4. Edges --------------------------------------------------------- #
    injury_unc = {t: r.injury_uncertainty for t, (_, r) in rosters.items()}
    grouped = group_quotes(quotes, resolved)
    name_of = {r.player_id: r.full_name for r in roster}
    pos_of = {r.player_id: r.position for r in roster}
    team_by_id = {r.player_id: r.team for r in roster}

    all_edges = []
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

        team = team_by_id.get(pid, home)
        opponent = away if team == home else home
        change = role_changes.get(pid)
        reasons = []
        if change:
            reasons.append(
                f"target share {change['baseline']:.0%} -> {change['recent']:.0%} "
                f"over last {change['games_since_change']} games"
            )
        reasons.append(
            f"projected {np.mean(sim.team_plays[team]):.0f} team plays, "
            f"{np.mean(sim.team_pass_attempts[team]):.0f} pass attempts"
        )

        edges = evaluate_prop(
            player_id=pid,
            player_name=name_of.get(pid, pid),
            team=team,
            opponent=opponent,
            position=pos_of.get(pid, "WR"),
            market=stat,
            distribution=dist,
            over_quotes=over,
            under_quotes=under,
            confidence_inputs=ConfidenceInputs(
                projection_quality=0.62,   # no NGS or PFF in this slice
                injury_certainty=1.0 - injury_unc.get(team, 0.0),
                role_certainty=0.55 if change else 0.80,
            ),
            reasons=reasons,
            thresholds=BetThresholds(),
            injury_uncertainty=injury_unc.get(team, 0.0),
            model_version="SLICE-0.1.0",
        )
        all_edges.extend(e for e in edges if e.side == "over" or True)

    # Deduplicate to the best price per player/market/side.
    best: dict[tuple, object] = {}
    for e in all_edges:
        key = (e.player_id, e.market, e.side)
        if key not in best or e.ev > best[key].ev:
            best[key] = e
    ranked = rank_edges([e for e in best.values() if e.edge >= args.min_edge])

    # --- 5. Report -------------------------------------------------------- #
    print(f"\n{'PLAYER':22} {'MARKET':9} {'SIDE':5} {'LINE':>6} {'BOOK':11} "
          f"{'ODDS':>6} {'PROJ':>7} {'P':>6} {'MKT':>6} {'EDGE':>7} "
          f"{'EV':>7} {'CONF':>5} {'GR':>3}  REC")
    print("-" * 128)
    for e in ranked[:20]:
        print(f"{e.player_name[:22]:22} {MARKET_LABELS[e.market]:9} "
              f"{e.side:5} {e.line:6.1f} {e.bookmaker[:11]:11} "
              f"{format_american(e.american):>6} {e.projection:7.1f} "
              f"{e.model_prob:6.1%} {e.market_prob:6.1%} {e.edge:+7.1%} "
              f"{e.ev:+7.1%} {e.confidence:5.0f} {e.grade:>3}  "
              f"{e.recommendation}")

    counts = defaultdict(int)
    for e in ranked:
        counts[e.recommendation] += 1
    print(f"\n{dict(counts)}")

    if ranked:
        arr = np.array([e.edge for e in ranked])
        print(f"Edge distribution: median {np.median(arr):+.1%}, "
              f"p90 {np.quantile(arr, 0.9):+.1%}, max {arr.max():+.1%}")
        if np.median(np.abs(arr)) > 0.10:
            print("  ! Median absolute edge above 10 points. That is a "
                  "calibration problem, not a slate full of value.")

    top = next((e for e in ranked
                if e.recommendation in ("strong", "moderate")), None)
    if top:
        print(f"\nWHY  {top.player_name} {top.side.upper()} {top.line:g} "
              f"{MARKET_LABELS[top.market]}")
        print(f"  Model {top.projection:.1f} (median {top.median:.1f}, "
              f"P25 {top.p25:.1f}, P75 {top.p75:.1f})")
        print(f"  {top.model_prob:.1%} vs market {top.market_prob:.1%} -> "
              f"{top.edge:+.1%} edge, fair {top.display_fair}, "
              f"best {top.display_odds} at {top.bookmaker}")
        for r in top.reasons:
            print(f"  - {r}")
        print(f"  Confidence {top.confidence:.0f}: {top.confidence_parts}")

    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
