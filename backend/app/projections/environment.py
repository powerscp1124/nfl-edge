"""Game environment: translating market and team inputs into play volume.

This layer answers "how many plays, how many of them are passes, and how does
that change as the game unfolds" before any player is considered. Everything
downstream is opportunity share applied to these totals, which is what makes the
projections explainable: a receiving-yards number can always be decomposed back
into team pass attempts, target share, and yards per target.

The coefficients below are starting priors, chosen to be in the right region for
modern NFL football. They are meant to be **refit from play-by-play**, not left
as constants — see ``fit_environment_priors`` in app/projections/fitting.py.
Every one of them is exposed in ``EnvironmentPriors`` so a refit replaces them
without touching the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvironmentPriors:
    """League-level relationships, refit each season from play-by-play."""

    # Fitted against 2025 play-by-play, all 32 teams (2026-09-08). Before that
    # these were round-number priors, and the reference in particular was on a
    # different scale from what ``seconds_per_play`` actually measures: it
    # returns neutral-situation drive pace, which runs 29.7-35.0 across the
    # league, against a reference of 27.5. Every team therefore read as several
    # seconds slow, and at 1.9 plays per second that cost ~7 plays a game --
    # a league-average team was simulated at 53.6 plays against an actual 60.7.
    base_plays_per_team: float = 60.7            # league mean offensive plays
    plays_per_point_of_total: float = 0.22       # higher totals -> more plays
    total_reference: float = 44.5
    seconds_per_play_reference: float = 32.4     # league mean of the estimator
    # Measured slope of plays on pace: 0.67 plays per second slower. Weakly
    # identified (r = -0.30 over one season of 32 teams), and deliberately
    # lower than the 1.9 it replaces: at 1.9 the model reproduced the real
    # spread of team play counts but was no more accurate than predicting the
    # league mean, so the extra spread was noise. Game-to-game variation comes
    # from plays_sd and the trailing-team bonus, not from this. Refit it
    # against more than one season before leaning on it.
    pace_play_sensitivity: float = 0.67
    trailing_play_bonus: float = 0.28            # extra plays per point trailed
    league_neutral_pass_rate: float = 0.575
    pass_rate_per_point_trailed: float = 0.0125  # logit-space script response
    wind_pass_rate_penalty: float = 0.0035       # per mph above threshold
    wind_threshold_mph: float = 12.0
    precip_pass_rate_penalty: float = 0.020
    cold_pass_rate_penalty: float = 0.0006       # per degree below threshold
    cold_threshold_f: float = 32.0
    sack_rate: float = 0.068
    points_per_touchdown: float = 7.0            # TD plus a made extra point
    field_goals_per_team: float = 1.7            # league average per team-game
    # Weight applied to a player's share for each touchdown he has already
    # scored in the same simulated game. 1.0 is fully independent allocation,
    # which overstates repeat scoring. Fit against market-implied 2+/anytime
    # ratios rather than assumed.
    td_repeat_penalty: float = 0.75
    # Fraction of touchdowns scored from short yardage (inside about the 5).
    # The rest are longer scores -- a 30-yard run, a deep ball -- and they are
    # NOT allocated by goal-line share. Treating every touchdown as a
    # goal-line touchdown concentrates scoring on the bell-cow back far more
    # than the market does, which shows up as a badly overpriced 2+ TD market.
    short_field_td_rate: float = 0.55
    margin_sd: float = 13.2
    total_sd: float = 10.4
    plays_sd: float = 5.5


@dataclass
class WeatherState:
    """Weather at kickoff. Domes short-circuit every adjustment."""

    temperature_f: float = 60.0
    wind_mph: float = 0.0
    precipitation_prob: float = 0.0
    is_dome: bool = False

    @property
    def effective_wind(self) -> float:
        return 0.0 if self.is_dome else max(self.wind_mph, 0.0)


@dataclass
class TeamEnvironment:
    """One team's side of a game environment."""

    team: str
    neutral_pass_rate: float = 0.575
    seconds_per_play: float = 27.5
    off_efficiency: float = 0.0        # EPA/play above league average
    def_efficiency_faced: float = 0.0  # opponent defensive EPA/play allowed
    red_zone_rush_rate: float = 0.50
    proe: float = 0.0                  # pass rate over expectation


@dataclass
class GameEnvironment:
    """Everything a game simulation needs before players are involved."""

    game_id: str
    home: TeamEnvironment
    away: TeamEnvironment
    spread_home: float          # -3.5 means home favoured by 3.5
    total: float
    weather: WeatherState = field(default_factory=WeatherState)
    priors: EnvironmentPriors = field(default_factory=EnvironmentPriors)

    def implied_totals(self) -> tuple[float, float]:
        """Implied points for (home, away)."""
        home = self.total / 2.0 - self.spread_home / 2.0
        return float(home), float(self.total - home)

    def expected_plays(self, team: TeamEnvironment) -> float:
        """Expected offensive plays before game script is simulated."""
        p = self.priors
        from_total = p.plays_per_point_of_total * (self.total - p.total_reference)
        from_pace = p.pace_play_sensitivity * (
            p.seconds_per_play_reference - team.seconds_per_play
        )
        return float(p.base_plays_per_team + from_total + from_pace)

    def weather_pass_rate_adjustment(self) -> float:
        """Additive shift to pass rate from weather, in probability points.

        Deliberately small. Wind is the only weather variable with a large,
        well-evidenced effect on passing, and even a 20mph wind moves pass rate
        by only a couple of points. Over-reacting to a 15-degree temperature
        difference is a common way to break a model.
        """
        w = self.weather
        if w.is_dome:
            return 0.0
        p = self.priors
        adj = 0.0
        if w.effective_wind > p.wind_threshold_mph:
            adj -= p.wind_pass_rate_penalty * (w.effective_wind - p.wind_threshold_mph)
        adj -= p.precip_pass_rate_penalty * max(w.precipitation_prob, 0.0)
        if w.temperature_f < p.cold_threshold_f:
            adj -= p.cold_pass_rate_penalty * (p.cold_threshold_f - w.temperature_f)
        return float(adj)

    def weather_yards_multipliers(self) -> dict[str, float]:
        """Multiplicative efficiency effects from weather.

        Wind hurts deep passing far more than short passing, so the deep
        multiplier is separate. Rushing efficiency is close to weather-neutral;
        rushing *volume* is where bad weather shows up, and that is handled
        through the pass-rate adjustment instead.
        """
        w = self.weather
        if w.is_dome:
            return {"pass_yards": 1.0, "deep_pass": 1.0, "rush_yards": 1.0}
        wind = w.effective_wind
        excess = max(wind - self.priors.wind_threshold_mph, 0.0)
        return {
            "pass_yards": float(np.clip(1.0 - 0.0055 * excess, 0.80, 1.0)),
            "deep_pass": float(np.clip(1.0 - 0.0140 * excess, 0.55, 1.0)),
            "rush_yards": float(np.clip(1.0 - 0.0012 * excess
                                        - 0.010 * w.precipitation_prob, 0.92, 1.0)),
        }

    def script_pass_rate(self, team: TeamEnvironment, point_differential):
        """Pass rate given a running score differential.

        ``point_differential`` is the team's score minus opponent's. Trailing
        teams pass more and leading teams run more, and the response is applied
        in logit space so the rate stays inside (0, 1) at any margin. This is the
        single most important mechanism in the simulator for rushing props: a
        back on a 7-point favourite sees materially more fourth-quarter carries
        than the same back on a 7-point underdog.

        Accepts a scalar or an array of differentials, so the simulator can
        evaluate all 10,000 simulations in one call.
        """
        p = self.priors
        diff = np.asarray(point_differential, dtype=float)
        base = np.clip(team.neutral_pass_rate + team.proe
                       + self.weather_pass_rate_adjustment(), 0.25, 0.85)
        logit = np.log(base / (1.0 - base))
        logit = logit - p.pass_rate_per_point_trailed * diff * 4.0
        rate = 1.0 / (1.0 + np.exp(-logit))
        return float(rate) if rate.ndim == 0 else rate

    def summary(self) -> dict:
        home_pts, away_pts = self.implied_totals()
        return {
            "game_id": self.game_id,
            "spread_home": self.spread_home,
            "total": self.total,
            "implied_home": round(home_pts, 2),
            "implied_away": round(away_pts, 2),
            "expected_plays_home": round(self.expected_plays(self.home), 1),
            "expected_plays_away": round(self.expected_plays(self.away), 1),
            "weather_pass_adjustment": round(self.weather_pass_rate_adjustment(), 4),
            "weather_multipliers": {
                k: round(v, 4) for k, v in self.weather_yards_multipliers().items()
            },
            "is_dome": self.weather.is_dome,
            "wind_mph": self.weather.effective_wind,
        }
