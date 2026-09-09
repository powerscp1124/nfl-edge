"""Shared projection pipeline: context + odds -> rosters, environment, quotes.

Lifted out of ``scripts/slice.py`` so the backtest runs the *same* code the
live slice runs. A backtest that re-implements the projection path measures a
model nobody ships; every difference between the two is a way for the backtest
to be optimistic about the thing actually being bet.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.odds import BookQuote
from app.ingest.player_matching import PlayerResolver
from app.projections.environment import (
    GameEnvironment,
    TeamEnvironment,
    WeatherState,
)
from app.projections.injury_engine import RoleProfile, propagate_injuries
from app.projections.usage_builder import (
    build_player_usage,
    flag_role_changes,
    normalize_team_shares,
)
from app.projections.anchor import LineAnchor
from app.sim.game_sim import TeamRoster

# Markets the simulator produces a distribution for.
MARKET_LABELS = {
    "passing_yards": "Pass Yds",
    "rushing_yards": "Rush Yds",
    "receiving_yards": "Rec Yds",
}


def build_rosters(context: dict, resolver: PlayerResolver,
                  market_players: frozenset = frozenset()):
    """Build both teams' PlayerUsage from game logs, then apply injuries."""
    from app.ingest.context_loader import (MIN_ROTATION_PLAYERS,
                                           in_rotation)
    from app.projections.usage_builder import (PlayerGameLog,
                                                roster_prior_scale)

    rosters, all_notes, role_changes = {}, [], {}
    depth_positions = context.get("depth_position", {})

    def trusted_rank(pid, position):
        """The depth-chart rank, or None when it is a rank for another group."""
        listed = depth_positions.get(pid)
        if listed is not None and listed != position:
            return None
        return context["depth_rank"].get(pid)

    def rank_for(pid, position):
        """Depth-chart rank, but only when it is a rank for this position.

        nflverse ranks within its own label. A player listed RB5 who projects
        as a receiver would otherwise be handed the WR5 prior -- a rank from a
        different depth chart. When they disagree, fall back to the default.
        """
        listed = depth_positions.get(pid)
        if listed is not None and listed != position:
            return 3
        return context["depth_rank"].get(pid, 3)


    for team, players in context["game_logs"].items():
        usages, roles, statuses = [], {}, {}
        prior_mass = []

        # Depth-chart priors are per-player figures that a whole roster
        # silently sums past 1.0. Scale them to fit before they are blended.
        # Prune to the players who will actually take snaps before anything
        # is estimated: prior mass handed to the rest comes straight out of
        # the pool the starters share.
        kept = {}
        for pid, logs_raw in players.items():
            rows = sorted(logs_raw, key=lambda r: r["week"])
            pos = rows[-1]["position"]
            snaps = [r["snap_share"] for r in rows
                     if r.get("snap_share") is not None]
            if in_rotation(pos, trusted_rank(pid, pos),
                           sum(snaps) / len(snaps) if snaps else None,
                           has_market=pid in market_players):
                kept[pid] = logs_raw
        if len(kept) >= MIN_ROTATION_PLAYERS:
            dropped = len(players) - len(kept)
            if dropped:
                all_notes.append(
                    f"{team}: {dropped} of {len(players)} players outside the "
                    "rotation excluded from the opportunity pool")
            players = kept

        entries = []
        for pid, logs_raw in players.items():
            pos = sorted(logs_raw, key=lambda r: r["week"])[-1]["position"]
            if pos in ("WR", "TE", "RB"):
                entries.append((pos, rank_for(pid, pos)))
        prior_scale = roster_prior_scale(entries)
        team_weeks = sorted({row["week"] for rows in players.values()
                             for row in rows})
        for pid, logs_raw in players.items():
            logs = [PlayerGameLog(**row) for row in logs_raw]
            built = build_player_usage(
                logs, depth_rank=rank_for(pid, logs[-1].position),
                prior_scale=prior_scale, team_weeks=team_weeks)
            usage = built.usage
            if usage.position == "QB" and pid == context["starting_qb"][team]:
                usage = type(usage)(**{**vars(usage), "is_starting_qb": True})
            usages.append(usage)
            prior_mass.append(built.prior_mass)
            all_notes.extend(f"{usage.name}: {n}" for n in built.notes)

            change = flag_role_changes(logs)
            if change:
                role_changes[pid] = change

            roles[pid] = RoleProfile(
                player_id=pid,
                position=usage.position,
                depth_rank=rank_for(pid, usage.position),
                slot_rate=context["slot_rate"].get(pid, 0.3),
                adot=usage.adot,
            )
            if pid in context["injuries"]:
                statuses[pid] = context["injuries"][pid]

        usages = normalize_team_shares(usages, prior_mass=prior_mass)
        result = propagate_injuries(usages, roles, statuses)
        env = TeamEnvironment(team=team, **context["team_env"][team])
        rosters[team] = (TeamRoster(env=env, players=result.usages), result)

    return rosters, all_notes, role_changes


def build_environment(context: dict, rosters) -> GameEnvironment:
    home, away = context["home_team"], context["away_team"]
    weather = WeatherState(**context["weather"])
    return GameEnvironment(
        game_id=context["game_id"],
        home=rosters[home][0].env,
        away=rosters[away][0].env,
        spread_home=context["spread_home"],
        total=context["total"],
        weather=weather,
    )


def group_quotes(quotes, resolved: dict[str, str]):
    """Bucket normalised odds rows by (player_id, stat, side)."""
    grouped = defaultdict(list)
    for q in quotes:
        pid = resolved.get(q.player_name)
        if pid is None or q.is_alternate:
            continue
        if q.stat not in MARKET_LABELS:
            continue
        side = "over" if q.side.startswith("o") else "under"
        grouped[(pid, q.stat, side)].append(
            BookQuote(bookmaker=q.bookmaker, line=q.line, american=q.american)
        )
    return grouped


def anchored_distribution(distribution, over_quotes, under_quotes,
                          anchor: LineAnchor | None = None,
                          deviations=None):
    """Recentre a simulated distribution on the market's consensus line.

    Uses the median line across books rather than one book's number: a single
    stale half-point would otherwise drag the anchor, and the anchor is now
    the projection rather than a comparison for it.
    """
    from app.core.odds import consensus_line

    quotes = list(over_quotes) + list(under_quotes)
    if not quotes:
        return distribution
    line = consensus_line(quotes)["median"]
    return (anchor or LineAnchor()).apply(distribution, line, deviations)
