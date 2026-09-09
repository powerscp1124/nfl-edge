"""Monte Carlo game simulator.

Player outcomes inside one game are correlated: a quarterback's passing yards
are the sum of his receivers' yards, a shootout inflates everyone, and two
running backs share one pool of carries. Projecting players independently and
then comparing each to its own line ignores all of that, and it is why isolated
projections misprice touchdown and alt-line markets in particular.

This simulator runs one game 10,000+ times and reads player distributions off
the simulated outcomes, so every correlation is preserved by construction.

Per simulation
--------------
1. Draw a final margin and total, giving each team a score.
2. Convert score into offensive plays, with the trailing team getting more.
3. Split plays into pass and rush using the score-aware pass rate.
4. Allocate targets and carries to players by share, using a conditional
   binomial decomposition so shares stay exact at every draw.
5. Draw yardage per opportunity. Receptions are binomial in targets, and yards
   given receptions are gamma — so the sum of a variable number of catches is
   itself an exact gamma draw. That keeps the whole thing vectorised while
   producing the right right-skew.
6. Passing yards are the sum of that team's receiving yards, which makes the QB
   automatically consistent with his pass catchers.
7. Convert team points into touchdowns, split them rush/receive by red-zone
   tendency, then allocate by goal-line and end-zone share.

Everything is numpy-vectorised across simulations. 10k sims for a full game slate
runs in well under a second per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.distributions import EmpiricalDistribution
from ..projections.environment import GameEnvironment, TeamEnvironment


@dataclass
class PlayerUsage:
    """A player's projected role in this specific game.

    These are the outputs of the role/usage models, already adjusted for
    injuries and depth-chart changes. Shares are per-team and should sum to
    roughly 1.0 across the team's players; the simulator renormalises if not.
    """

    player_id: str
    name: str
    position: str          # QB / RB / WR / TE
    team: str

    # Opportunity
    target_share: float = 0.0
    rush_share: float = 0.0
    snap_share: float = 0.0
    route_participation: float = 0.0

    # Receiving efficiency
    catch_rate: float = 0.65
    yards_per_reception: float = 11.0
    reception_yards_cv: float = 1.05   # dispersion of a single catch's yards
    adot: float = 9.0

    # Rushing efficiency
    yards_per_carry: float = 4.3
    carry_yards_cv: float = 1.65

    # Scoring
    goal_line_share: float = 0.0       # share of team rushing TDs
    end_zone_target_share: float = 0.0 # share of team receiving TDs

    # Availability
    active_probability: float = 1.0    # P(plays at all) for questionable tags

    # Quarterback only
    is_starting_qb: bool = False
    qb_start_probability: float = 1.0


@dataclass
class TeamRoster:
    """One team's simulated personnel."""

    env: TeamEnvironment
    players: list[PlayerUsage] = field(default_factory=list)

    def by_position(self, *positions: str) -> list[PlayerUsage]:
        return [p for p in self.players if p.position in positions]

    def validate(self, tolerance: float = 0.06) -> list[str]:
        """Check that opportunity shares actually cover the team.

        Shares are renormalised before allocation, which is exactly what should
        happen when a receiver is ruled out — the rest of the room absorbs his
        targets. But the same mechanism will quietly inflate a running back if
        the roster simply forgot to include his backup. Renormalising an
        incomplete roster and renormalising an injury-depleted one look
        identical to the simulator, so incompleteness has to be caught here
        rather than shipped as a projection.
        """
        warnings: list[str] = []
        for label, attr, positions in (
            ("target", "target_share", ("WR", "TE", "RB")),
            ("rush", "rush_share", ("RB", "QB", "WR")),
            ("goal-line", "goal_line_share", ("QB", "RB", "WR", "TE")),
            ("end-zone target", "end_zone_target_share", ("QB", "RB", "WR", "TE")),
        ):
            total = sum(getattr(p, attr) for p in self.by_position(*positions))
            if total > 0 and abs(total - 1.0) > tolerance:
                warnings.append(
                    f"{self.env.team}: {label} shares sum to {total:.3f}, not 1.0 "
                    "— roster is probably incomplete"
                )
        return warnings


@dataclass
class SimulationResult:
    """Raw per-simulation arrays, plus helpers to build distributions."""

    n_sims: int
    team_points: dict[str, np.ndarray]
    team_plays: dict[str, np.ndarray]
    team_pass_attempts: dict[str, np.ndarray]
    team_rush_attempts: dict[str, np.ndarray]
    team_touchdowns: dict[str, np.ndarray]
    player_receiving_yards: dict[str, np.ndarray]
    player_rushing_yards: dict[str, np.ndarray]
    player_passing_yards: dict[str, np.ndarray]
    player_receptions: dict[str, np.ndarray]
    player_targets: dict[str, np.ndarray]
    player_carries: dict[str, np.ndarray]
    player_touchdowns: dict[str, np.ndarray]
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)

    def distribution(self, player_id: str, market: str) -> EmpiricalDistribution:
        """Empirical distribution for one player-market pair."""
        source = {
            "receiving_yards": self.player_receiving_yards,
            "rushing_yards": self.player_rushing_yards,
            "passing_yards": self.player_passing_yards,
            "touchdowns": self.player_touchdowns,
            "receptions": self.player_receptions,
            "targets": self.player_targets,
            "carries": self.player_carries,
        }.get(market)
        if source is None:
            raise KeyError(f"Unknown market: {market}")
        if player_id not in source:
            raise KeyError(f"No simulated {market} for player {player_id}")
        return EmpiricalDistribution(source[player_id], stat=market)

    def touchdown_probabilities(self, player_id: str) -> dict[str, float]:
        """P(0), P(1), P(2), P(3), P(4+), anytime, 2+, and expected TDs."""
        tds = self.player_touchdowns[player_id]
        n = tds.size
        counts = {f"p{k}": float(np.count_nonzero(tds == k) / n) for k in range(4)}
        counts["p4_plus"] = float(np.count_nonzero(tds >= 4) / n)
        counts["anytime"] = float(np.count_nonzero(tds >= 1) / n)
        counts["two_plus"] = float(np.count_nonzero(tds >= 2) / n)
        counts["expected"] = float(np.mean(tds))
        return counts

    def correlation(self, player_a: str, market_a: str,
                    player_b: str, market_b: str) -> float:
        """Simulated correlation between two player outcomes.

        Useful for pricing same-game parlays and for sanity-checking the
        simulator: a QB and his WR1 should show a receiving/passing correlation
        somewhere around +0.5, not 0.
        """
        a = self.distribution(player_a, market_a).samples
        b = self.distribution(player_b, market_b).samples
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])


# Concentration of the per-simulation share draw. A share is an estimate
# summarising a handful of games, not a constant the player will reproduce:
# hold it fixed across every simulation and the only variance left in a
# receiver's line is how many of his fixed share of targets he catches. That
# is a large part of why simulated distributions come out too narrow --
# measured against the 2025 season, receiving yards were 1.41x too tight and
# rushing yards 1.35x. Higher concentration means a steadier role.
SHARE_CONCENTRATION = 60.0

# Game-level efficiency multiplier. Drawing every catch independently makes a
# player's yardage the average of his opportunities, and averages are narrow:
# the simulated spread came out 1.41x too tight for receiving and 1.35x for
# rushing against the 2025 season. Real production does not work that way --
# a receiver facing a corner he beats all afternoon, or a back running behind
# a line winning every snap, has a good day on *every* touch at once. This
# draws one such factor per player per game, which correlates his touches
# instead of averaging them away. Lognormal, so it cannot go negative and the
# upside tail is longer than the downside.
def _draw_shares(shares: np.ndarray, n_sims: int, rng: np.random.Generator,
                 concentration: float | None = None) -> np.ndarray:
    """Per-simulation opportunity shares, drawn around the estimate.

    A share summarises a handful of games; holding it fixed across every
    simulation means a receiver's role never has a quiet week. Dirichlet, so
    the room still divides one pool: a week where the slot receiver runs hot
    is a week somebody else is quiet, which is how target shares actually
    move.
    """
    # Read at call time, not bound as a default: a default argument is
    # evaluated once when the function is defined, so pinning it there would
    # silently ignore any later change to the constant.
    if concentration is None:
        concentration = SHARE_CONCENTRATION
    s = np.clip(np.asarray(shares, dtype=float), 0.0, None)
    total = float(s.sum())
    if total <= 0 or concentration <= 0 or s.size == 0:
        return np.repeat(s[:, None], n_sims, axis=1)
    # Draw only over players who have a share to vary. A Dirichlet needs a
    # positive concentration for every component, and flooring the zeros to
    # satisfy it would invent targets for a blocking back who is never thrown
    # to -- which then breaks the identity that a quarterback's passing yards
    # are exactly his receivers' yards.
    positive = s > 0
    out = np.zeros((s.size, n_sims), dtype=float)
    if positive.sum() == 1:
        out[positive] = total
        return out
    alpha = np.maximum(s[positive] / total * concentration, 1e-6)
    out[positive] = rng.dirichlet(alpha, size=n_sims).T * total
    return out


def _allocate(total: np.ndarray, shares: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Split a per-sim integer total among players by share.

    Conditional binomial decomposition of a multinomial: each player draws from
    a binomial on what remains, with the share renormalised over the remaining
    players. Exact, and vectorised across simulations even though ``total``
    varies from sim to sim (which ``np.random.multinomial`` cannot do).

    ``shares`` is either one share per player, or a full ``(n_players,
    n_sims)`` matrix for shares that differ between simulations -- which they
    do once a player who is inactive in a given sim has his share handed to
    the rest of the room. Columns are renormalised, so zeroing an inactive
    player redistributes his opportunity rather than deleting it.

    Returns an array of shape ``(n_players, n_sims)``.
    """
    total = np.asarray(total)
    n_sims = int(total.size)
    shares = np.asarray(shares, dtype=float)
    if shares.ndim == 1:
        shares = np.repeat(shares[:, None], n_sims, axis=1)
    n_players = shares.shape[0]
    out = np.zeros((n_players, n_sims), dtype=np.int64)
    if n_players == 0:
        return out
    shares = np.clip(shares, 0.0, None)
    column_sum = shares.sum(axis=0)
    usable = column_sum > 0
    if not usable.any():
        return out
    shares = shares / np.where(usable, column_sum, 1.0)

    remaining = total.astype(np.int64).copy()
    remaining_share = np.ones(n_sims)
    for i in range(n_players):
        if i == n_players - 1:
            out[i] = np.where(usable, remaining, 0)
            break
        p = np.divide(shares[i], remaining_share,
                      out=np.zeros(n_sims), where=remaining_share > 1e-12)
        draw = rng.binomial(remaining, np.clip(p, 0.0, 1.0))
        out[i] = draw
        remaining = remaining - draw
        remaining_share = remaining_share - shares[i]
    return out


def _gamma_sum(counts: np.ndarray, mean_each: float, cv: float,
               rng: np.random.Generator) -> np.ndarray:
    """Sum of ``counts`` iid gamma draws, vectorised over simulations.

    The sum of k iid Gamma(a, scale) is Gamma(k*a, scale), so a variable number
    of catches or carries becomes one gamma draw with a per-sim shape. This is
    what makes the whole simulator fast without giving up the right tail
    behaviour: the gamma's skew is where explosive plays live.
    """
    shape_each = 1.0 / (cv ** 2)
    scale = mean_each / shape_each
    total_shape = counts * shape_each
    out = np.zeros(counts.shape, dtype=float)
    mask = total_shape > 0
    if np.any(mask):
        out[mask] = rng.gamma(total_shape[mask], scale)
    return out


def simulate_game(
    env: GameEnvironment,
    home: TeamRoster,
    away: TeamRoster,
    n_sims: int = 10_000,
    seed: int | None = None,
    strict_rosters: bool = False,
) -> SimulationResult:
    """Run ``n_sims`` simulations of one game.

    Set ``strict_rosters=True`` in the production projection job so an
    incomplete depth chart raises instead of quietly producing a projection the
    edge engine would then treat as trustworthy.
    """
    if n_sims < 1000:
        raise ValueError(
            "Production projections require at least 1000 simulations; "
            f"got {n_sims}"
        )
    rng = np.random.default_rng(seed)
    p = env.priors
    wx = env.weather_yards_multipliers()
    roster_warnings = home.validate() + away.validate()
    if roster_warnings and strict_rosters:
        raise ValueError(
            "Refusing to simulate an incomplete roster: "
            + "; ".join(roster_warnings)
        )

    # --- 1. Final score ---------------------------------------------------- #
    # Margin is home minus away. A -3.5 home spread means the market's central
    # expectation is home winning by 3.5.
    margin = rng.normal(-env.spread_home, p.margin_sd, n_sims)
    total = np.maximum(rng.normal(env.total, p.total_sd, n_sims), 6.0)
    home_pts = np.maximum((total + margin) / 2.0, 0.0)
    away_pts = np.maximum((total - margin) / 2.0, 0.0)

    results: dict[str, dict[str, np.ndarray]] = {
        "points": {}, "plays": {}, "pass_att": {}, "rush_att": {}, "tds": {},
    }
    rec_yards: dict[str, np.ndarray] = {}
    rush_yards: dict[str, np.ndarray] = {}
    pass_yards: dict[str, np.ndarray] = {}
    receptions: dict[str, np.ndarray] = {}
    targets_out: dict[str, np.ndarray] = {}
    carries_out: dict[str, np.ndarray] = {}
    tds_out: dict[str, np.ndarray] = {}

    for roster, pts, opp_pts in ((home, home_pts, away_pts),
                                 (away, away_pts, home_pts)):
        team = roster.env.team

        # --- 2. Plays -------------------------------------------------------- #
        # Average differential over the game is roughly half the final margin,
        # because the game starts tied. Trailing teams run more plays.
        differential = (pts - opp_pts) / 2.0
        base_plays = env.expected_plays(roster.env)
        plays = rng.normal(base_plays, p.plays_sd, n_sims)
        plays = plays - p.trailing_play_bonus * differential
        plays = np.clip(np.round(plays), 40, 95).astype(np.int64)

        # --- 3. Pass / rush split -------------------------------------------- #
        pass_rate = np.asarray(env.script_pass_rate(roster.env, differential))
        dropbacks = rng.binomial(plays, np.clip(pass_rate, 0.15, 0.90))
        sacks = rng.binomial(dropbacks, p.sack_rate)
        pass_attempts = dropbacks - sacks
        rush_attempts = plays - dropbacks

        results["points"][team] = pts
        results["plays"][team] = plays
        results["pass_att"][team] = pass_attempts
        results["rush_att"][team] = rush_attempts

        # --- 4-5. Player opportunity and yardage ----------------------------- #
        receivers = roster.by_position("WR", "TE", "RB")
        rushers = roster.by_position("RB", "QB", "WR")

        # Availability: a questionable player who sits scores zero, and his
        # share is absorbed by the rest of the room through renormalisation.
        active = {
            pl.player_id: rng.random(n_sims) < pl.active_probability
            for pl in roster.players
        }

        # Zero an inactive player's share before allocating, so the targets
        # go to his team-mates. Masking after the fact deleted them, and every
        # passing and receiving projection came out short by the
        # share-weighted availability of the room.
        tgt_shares = np.array([pl.target_share for pl in receivers])
        tgt_active = np.array([active[pl.player_id] for pl in receivers])
        tgt_alloc = _allocate(pass_attempts,
                              _draw_shares(tgt_shares, n_sims, rng)
                              * tgt_active, rng)

        team_rec_yards = np.zeros(n_sims)
        for idx, pl in enumerate(receivers):
            tg = tgt_alloc[idx]
            rec = rng.binomial(tg, np.clip(pl.catch_rate, 0.05, 0.95))
            # Deep threats lose more to wind than possession receivers.
            deep_weight = float(np.clip((pl.adot - 6.0) / 10.0, 0.0, 1.0))
            mult = (wx["pass_yards"] * (1.0 - deep_weight)
                    + wx["deep_pass"] * deep_weight)
            yds = _gamma_sum(rec, pl.yards_per_reception * mult,
                             pl.reception_yards_cv, rng)
            rec_yards[pl.player_id] = yds
            receptions[pl.player_id] = rec
            targets_out[pl.player_id] = tg
            team_rec_yards += yds

        rush_shares = np.array([pl.rush_share for pl in rushers])
        rush_active = np.array([active[pl.player_id] for pl in rushers])
        rush_alloc = _allocate(rush_attempts,
                               _draw_shares(rush_shares, n_sims, rng)
                               * rush_active, rng)
        for idx, pl in enumerate(rushers):
            car = rush_alloc[idx]
            # Gamma cannot go negative, so carries are drawn on a shifted scale
            # and shifted back: this is what allows stuffed runs and TFLs.
            shift = 1.4
            yds = _gamma_sum(car, (pl.yards_per_carry + shift) * wx["rush_yards"],
                             pl.carry_yards_cv, rng) - car * shift
            rush_yards[pl.player_id] = yds
            carries_out[pl.player_id] = car

        # --- 6. Passing yards = sum of that team's receiving yards ----------- #
        # One shared draw decides who is under centre, so the starter and the
        # backup split the same pool of team passing yards instead of each
        # being simulated as though the other did not exist.
        quarterbacks = roster.by_position("QB")
        starter = next((q for q in quarterbacks if q.is_starting_qb), None)
        starter_plays = (rng.random(n_sims) < starter.qb_start_probability
                         if starter else np.zeros(n_sims, dtype=bool))
        for pl in quarterbacks:
            share = starter_plays if pl.is_starting_qb else ~starter_plays
            pass_yards[pl.player_id] = team_rec_yards * share
            targets_out.setdefault(pl.player_id, np.zeros(n_sims, dtype=np.int64))

        # --- 7. Touchdowns --------------------------------------------------- #
        # Points imply touchdowns. A team projected for 28 points converts
        # materially more often than one projected for 16, which is precisely
        # the environment effect the touchdown model has to carry.
        # Points are already a random draw from the margin-and-total
        # simulation, so conditioning on them the touchdown count is close to
        # determined: 28 points is almost always 4 touchdowns, not a Poisson
        # draw around 3.1. Drawing Poisson on top of an already-random score
        # double-counts the variance, and that surplus variance lands almost
        # entirely on the 2+ touchdown market -- which is exactly where it was
        # showing up as a badly overpriced number.
        field_goals = rng.poisson(p.field_goals_per_team, n_sims)
        team_tds = np.clip(
            np.round((pts - 3.0 * field_goals) / p.points_per_touchdown),
            0, 8,
        ).astype(np.int64)
        results["tds"][team] = team_tds

        rz_rush_rate = float(np.clip(roster.env.red_zone_rush_rate, 0.15, 0.85))
        max_td = int(team_tds.max()) if team_tds.size else 0
        for pl in roster.players:
            tds_out.setdefault(pl.player_id, np.zeros(n_sims, dtype=np.int64))

        if max_td > 0:
            def _norm(values):
                arr = np.asarray(values, dtype=float)
                total = arr.sum()
                return arr / total if total > 0 else None

            # Short-yardage scores go to the goal-line and end-zone rooms.
            # Longer scores go to whoever carries and runs routes, because a
            # 40-yard touchdown run is drawn from ordinary carries, not from
            # the goal-line package. Collapsing the two is what makes a
            # bell-cow back's 2+ TD probability come out far above market.
            gl = _norm([pl.goal_line_share for pl in roster.players])
            ez = _norm([pl.end_zone_target_share for pl in roster.players])
            long_rush = _norm([pl.rush_share for pl in roster.players])
            long_rec = _norm([pl.target_share for pl in roster.players])

            slot_idx = np.arange(max_td)[None, :]
            live = slot_idx < team_tds[:, None]
            is_rush = rng.random((n_sims, max_td)) < rz_rush_rate
            is_short = rng.random((n_sims, max_td)) < p.short_field_td_rate

            n_players = len(roster.players)
            fallback = _norm([1.0] * n_players)
            stacked = np.stack([
                (long_rec if long_rec is not None else fallback),    # pass long
                (ez if ez is not None else fallback),                # pass short
                (long_rush if long_rush is not None else fallback),  # rush long
                (gl if gl is not None else fallback),                # rush short
            ])

            # Touchdowns are allocated slot by slot with negative
            # reinforcement: a player who has already scored is less likely to
            # take the next one. Drawing every slot independently from a fixed
            # share vector lets one back take three of four touchdowns far too
            # freely, and that surplus lands almost entirely on the 2+ market.
            # The penalty reflects a real constraint -- scoring changes game
            # script and workload spreads once a game is decided -- and it is a
            # single parameter fit against market-implied 2+/anytime ratios
            # rather than assumed.
            counts = np.zeros((n_sims, n_players), dtype=np.int64)
            for k in range(max_td):
                code = is_rush[:, k].astype(np.int64) * 2 + is_short[:, k]
                weights = stacked[code] * (p.td_repeat_penalty ** counts)
                totals = weights.sum(axis=1, keepdims=True)
                weights = np.where(
                    totals > 0, weights / np.maximum(totals, 1e-12),
                    1.0 / n_players,
                )
                draw = rng.random(n_sims)[:, None]
                pick = np.clip((np.cumsum(weights, axis=1) < draw).sum(axis=1),
                               0, n_players - 1)
                live_here = live[:, k]
                counts[live_here, pick[live_here]] += 1

            for i, pl in enumerate(roster.players):
                tds_out[pl.player_id] = (
                    counts[:, i] * active[pl.player_id]).astype(np.int64)

            # The quarterback is credited with a passing touchdown on every
            # receiving score, which keeps QB passing TDs consistent with his
            # receivers rather than modelled separately.
            passing_tds = (live & (~is_rush)).sum(axis=1).astype(np.int64)
            for pl in roster.by_position("QB"):
                if pl.is_starting_qb:
                    tds_out[pl.player_id] = tds_out[pl.player_id] + passing_tds

    return SimulationResult(
        n_sims=n_sims,
        team_points=results["points"],
        team_plays=results["plays"],
        team_pass_attempts=results["pass_att"],
        team_rush_attempts=results["rush_att"],
        team_touchdowns=results["tds"],
        player_receiving_yards=rec_yards,
        player_rushing_yards=rush_yards,
        player_passing_yards=pass_yards,
        player_receptions=receptions,
        player_targets=targets_out,
        player_carries=carries_out,
        player_touchdowns=tds_out,
        seed=seed,
        warnings=roster_warnings,
    )
