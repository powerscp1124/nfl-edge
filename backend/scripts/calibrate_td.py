#!/usr/bin/env python3
"""Fit the touchdown repeat penalty against market-implied 2+ rates.

The 2+ touchdown market is where a simulator's allocation assumptions show up
most sharply, because it is entirely a statement about repeat scoring. Two
models can agree exactly on every anytime price and disagree by a factor of two
on 2+.

IMPORTANT: the reference curve below is hand-entered from typical NFL prices,
not scraped. It is good enough to remove an obvious bias and not good enough to
trust as calibration. Replace ``market_two_plus`` with real de-vigged
anytime/2+ pairs from stored ``odds_snapshots`` before believing the fitted
value -- ``load_market_pairs`` is where that goes.

    python scripts/calibrate_td.py
"""
import sys; sys.path.insert(0,'.')
import numpy as np
from app.projections.environment import GameEnvironment, TeamEnvironment, WeatherState, EnvironmentPriors
from app.sim.game_sim import PlayerUsage, TeamRoster, simulate_game

def run(gl, rush, penalty):
    pr=EnvironmentPriors(td_repeat_penalty=penalty)
    he=TeamEnvironment(team="KC",neutral_pass_rate=0.58,red_zone_rush_rate=0.42)
    ae=TeamEnvironment(team="DEN")
    home=TeamRoster(env=he,players=[
        PlayerUsage("qb","QB","QB","KC",is_starting_qb=True,rush_share=0.08,goal_line_share=0.10),
        PlayerUsage("rb","RB","RB","KC",target_share=0.12,rush_share=rush,yards_per_carry=4.4,
                    goal_line_share=gl,end_zone_target_share=0.08),
        PlayerUsage("rb2","RB2","RB","KC",rush_share=0.92-rush,goal_line_share=0.90-gl),
        PlayerUsage("wr1","WR1","WR","KC",target_share=0.28,end_zone_target_share=0.32),
        PlayerUsage("wr2","WR2","WR","KC",target_share=0.24,end_zone_target_share=0.22),
        PlayerUsage("te","TE","TE","KC",target_share=0.36,end_zone_target_share=0.38),
    ])
    away=TeamRoster(env=ae,players=[
        PlayerUsage("dqb","QB","QB","DEN",is_starting_qb=True,goal_line_share=0.15),
        PlayerUsage("drb","RB","RB","DEN",rush_share=1.0,goal_line_share=0.55,end_zone_target_share=0.10),
        PlayerUsage("dwr","WR","WR","DEN",target_share=0.6,end_zone_target_share=0.45),
        PlayerUsage("dte","TE","TE","DEN",target_share=0.4,end_zone_target_share=0.30),
    ])
    env=GameEnvironment(game_id="t",home=he,away=ae,spread_home=-4.5,total=46.5,
                        weather=WeatherState(is_dome=True),priors=pr)
    sim=simulate_game(env,home,away,n_sims=40000,seed=5)
    return sim.touchdown_probabilities("rb"), sim.touchdown_probabilities("wr1")

def load_market_pairs():
    """Real de-vigged (anytime, 2+) pairs from stored odds.

    Returns None until wired. Query odds_snapshots for player_anytime_td and
    player_tds_over at the same timestamp and book, de-vig each against its
    market, and return the pairs. Fitting against real pairs is the only way
    this number means anything.
    """
    return None


# Placeholder reference curve. Hand-entered, not scraped. See module docstring.
FALLBACK_ANYTIME = [0.30, 0.35, 0.45, 0.55, 0.65, 0.72]
FALLBACK_TWO_PLUS = [0.055, 0.070, 0.100, 0.150, 0.200, 0.245]


def market_two_plus(anytime):
    pairs = load_market_pairs()
    if pairs:
        xs, ys = zip(*sorted(pairs))
        return np.interp(anytime, xs, ys)
    return np.interp(anytime, FALLBACK_ANYTIME, FALLBACK_TWO_PLUS)

if load_market_pairs() is None:
    print("WARNING: fitting against the hand-entered fallback curve, not real\n"
          "         market prices. Wire load_market_pairs() before trusting\n"
          "         the fitted value.\n")

for penalty in (1.0, 0.85, 0.75, 0.65, 0.55, 0.45):
    errs=[]
    rows=[]
    for gl,rush in [(0.75,0.75),(0.62,0.68),(0.50,0.55),(0.38,0.45),(0.25,0.35)]:
        rb,wr=run(gl,rush,penalty)
        for t in (rb,wr):
            exp=market_two_plus(t['anytime'])
            errs.append(t['two_plus']-exp)
        rows.append((rb['anytime'],rb['two_plus'],market_two_plus(rb['anytime'])))
    print(f"penalty={penalty:.2f}  mean signed error={np.mean(errs):+.4f}  "
          f"mean abs={np.mean(np.abs(errs)):.4f}")
    if penalty in (0.75, 0.65):
        for a,t,m in rows:
            print(f"    anytime {a:.3f}  model 2+ {t:.3f}  market 2+ {m:.3f}")
import sys; sys.path.insert(0,'.')
import numpy as np
from app.projections.environment import GameEnvironment, TeamEnvironment, WeatherState, EnvironmentPriors
from app.sim.game_sim import PlayerUsage, TeamRoster, simulate_game

def run(gl, rush, penalty):
    pr=EnvironmentPriors(td_repeat_penalty=penalty)
    he=TeamEnvironment(team="KC",neutral_pass_rate=0.58,red_zone_rush_rate=0.42)
    ae=TeamEnvironment(team="DEN")
    home=TeamRoster(env=he,players=[
        PlayerUsage("qb","QB","QB","KC",is_starting_qb=True,rush_share=0.08,goal_line_share=0.10),
        PlayerUsage("rb","RB","RB","KC",target_share=0.12,rush_share=rush,yards_per_carry=4.4,
                    goal_line_share=gl,end_zone_target_share=0.08),
        PlayerUsage("rb2","RB2","RB","KC",rush_share=0.92-rush,goal_line_share=0.90-gl),
        PlayerUsage("wr1","WR1","WR","KC",target_share=0.28,end_zone_target_share=0.32),
        PlayerUsage("wr2","WR2","WR","KC",target_share=0.24,end_zone_target_share=0.22),
        PlayerUsage("te","TE","TE","KC",target_share=0.36,end_zone_target_share=0.38),
    ])
    away=TeamRoster(env=ae,players=[
        PlayerUsage("dqb","QB","QB","DEN",is_starting_qb=True,goal_line_share=0.15),
        PlayerUsage("drb","RB","RB","DEN",rush_share=1.0,goal_line_share=0.55,end_zone_target_share=0.10),
        PlayerUsage("dwr","WR","WR","DEN",target_share=0.6,end_zone_target_share=0.45),
        PlayerUsage("dte","TE","TE","DEN",target_share=0.4,end_zone_target_share=0.30),
    ])
    env=GameEnvironment(game_id="t",home=he,away=ae,spread_home=-4.5,total=46.5,
                        weather=WeatherState(is_dome=True),priors=pr)
    sim=simulate_game(env,home,away,n_sims=40000,seed=5)
    return sim.touchdown_probabilities("rb"), sim.touchdown_probabilities("wr1")

def load_market_pairs():
    """Real de-vigged (anytime, 2+) pairs from stored odds.

    Returns None until wired. Query odds_snapshots for player_anytime_td and
    player_tds_over at the same timestamp and book, de-vig each against its
    market, and return the pairs. Fitting against real pairs is the only way
    this number means anything.
    """
    return None


# Placeholder reference curve. Hand-entered, not scraped. See module docstring.
FALLBACK_ANYTIME = [0.30, 0.35, 0.45, 0.55, 0.65, 0.72]
FALLBACK_TWO_PLUS = [0.055, 0.070, 0.100, 0.150, 0.200, 0.245]


def market_two_plus(anytime):
    pairs = load_market_pairs()
    if pairs:
        xs, ys = zip(*sorted(pairs))
        return np.interp(anytime, xs, ys)
    return np.interp(anytime, FALLBACK_ANYTIME, FALLBACK_TWO_PLUS)

if load_market_pairs() is None:
    print("WARNING: fitting against the hand-entered fallback curve, not real\n"
          "         market prices. Wire load_market_pairs() before trusting\n"
          "         the fitted value.\n")

for penalty in (1.0, 0.85, 0.75, 0.65, 0.55, 0.45):
    errs=[]
    rows=[]
    for gl,rush in [(0.75,0.75),(0.62,0.68),(0.50,0.55),(0.38,0.45),(0.25,0.35)]:
        rb,wr=run(gl,rush,penalty)
        for t in (rb,wr):
            exp=market_two_plus(t['anytime'])
            errs.append(t['two_plus']-exp)
        rows.append((rb['anytime'],rb['two_plus'],market_two_plus(rb['anytime'])))
    print(f"penalty={penalty:.2f}  mean signed error={np.mean(errs):+.4f}  "
          f"mean abs={np.mean(np.abs(errs)):.4f}")
    if penalty in (0.75, 0.65):
        for a,t,m in rows:
            print(f"    anytime {a:.3f}  model 2+ {t:.3f}  market 2+ {m:.3f}")
