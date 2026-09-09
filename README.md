# NFL Prop Engine

Player prop and touchdown projections compared against sportsbook markets, built
around one output:

> "63% chance of going over 81.5 yards while the market implies 51%"

rather than "projected for 94 yards."

Projections are probabilistic estimates with real uncertainty. No wager is
guaranteed, and historical performance does not predict future results.

## Read this first

The model has now been run against live data and backtested over three
seasons. **It has no demonstrated edge on these markets.** The central
measurement, on 2025 (n = 2,801 receiving props, 1,345 rushing):

| market | corr(our projection, actual) | corr(book line, actual) |
|---|---|---|
| passing | 0.125 | 0.316 |
| receiving | 0.434 | 0.578 |
| rushing | 0.530 | 0.655 |

Regress the outcome on **both** the projection and the line, and the
coefficient on the projection is **-0.012** for receiving and **-0.003** for
rushing. R² using both inputs equals R² using the line alone, to three decimal
places. The projection contains no information the market price does not
already carry.

Everything else in this document follows from that. Four full-season backtests
returned ROI between +0.35% and -7.65%, all indistinguishable from breakeven
or worse. Four candidate signals were tested and all came back null. The
engineering is sound; the information is not there.

This is written down rather than quietly dropped because the most expensive
thing a betting model can do is look plausible.

## Status

**301 unit tests pass.** The pipeline runs end to end against live data.

| Module | State |
|---|---|
| `core/odds.py` | Complete, tested |
| `core/distributions.py` | Complete, tested |
| `core/edge.py` | Complete, tested |
| `core/calibration.py` | Platt scaling, fitted and validated out of sample |
| `sim/game_sim.py` | Complete, tested, run live |
| `projections/anchor.py` | Line anchoring and deviations; tested |
| `projections/pipeline.py` | Shared projection path used by slice *and* backtest |
| `projections/environment.py` | Complete; play-model constants refit against 2025 |
| `projections/injury_engine.py` | Complete, tested |
| `projections/explain.py` | Complete, untested |
| `backtest/engine.py` | Metrics and guards complete and tested |
| `backtest/runner.py` | Written and run: 285 games/season, three seasons |
| `ingest/odds_api.py` | Run live, including the historical endpoints |
| `ingest/nflverse.py` | Run live; several schema assumptions corrected |
| `ingest/context_loader.py` | Run live; point-in-time guard extended to play-by-play |
| `ingest/player_matching.py` | Complete, tested; team/novelty markets now filtered |
| `ingest/teams.py` | Complete, tested |
| `ingest/weather.py` | Runs in preflight; **fails inside the slice** (open) |
| `db/schema.sql` | Complete, never applied to a live database |
| `main.py` | Routes typed and documented; all raise `NotImplementedError` |
| Frontend / ML ensemble / Celery | Not started |

## Setup

```bash
cp .env.example .env          # add your ODDS_API_KEY -- never commit this
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

`pyarrow` is required and is **not** declared by nflreadpy or polars — the
Polars-to-pandas conversion at the ingest boundary needs it, and without it
every nflverse loader raises `ModuleNotFoundError`. It is pinned in
`requirements.txt` for that reason.

### Then, in this order

```bash
python scripts/preflight.py            # everything free, 0 API credits
python scripts/preflight.py --spend    # adds one props pull (costs credits)
python scripts/slice.py --event-id <id> --season <season>
```

`--season` selects the season whose *history* builds the projection, not the
season of the game. For a week 1 game, last completed season is the correct
answer; the current season has no data yet and the loader raises rather than
returning an empty frame. `nflreadpy.get_current_season()` cannot be trusted
for this — pass `--season` explicitly.

### Backtesting and research

```bash
python scripts/backtest.py --season 2025 --weeks 1-22 --save-bets bets.csv
python scripts/backtest.py --season 2025 --weeks 5 --clv       # + closing line value
python scripts/harvest_lines.py --season 2024 --out lines-2024.csv
python scripts/calibrate_probs.py --bets bets.csv --train 1-11 --test 12-22
```

`backtest.py` replays past games through the **same** pipeline the live slice
uses — imported, not reimplemented, because a backtest with its own copy of
the projection code measures a model nobody ships. It grades only wagers the
thresholds would actually place (`needs_review` and `pass` are not bets), and
it never hands the projection a timestamp at or after kickoff.

`harvest_lines.py` fetches historical lines and joins them to settled stat
lines **without simulating**: 15 minutes a season against ~2 hours for a
backtest. Any "does X predict beating the line" question should be answered
against a harvested CSV before a backtest is considered.

## What the backtest found

Four full-season runs on 2025, 285 games each:

| run | bets | ROI | win rate | Brier | log loss |
|---|---|---|---|---|---|
| baseline | 2009 | +0.35% | 52.56% | 0.2552 | 0.7041 |
| widened distributions | 1784 | -0.20% | 52.19% | 0.2559 | 0.7056 |
| availability fixed | 2034 | -0.95% | 51.77% | 0.2560 | 0.7057 |
| line-anchored | 609 | -6.29% | 47.95% | 0.2503 | 0.6936 |

Breakeven at -110 is about 52.4%. Predicting a flat 50% on everything scores
Brier 0.2500 and log loss 0.6931 — so the first three runs were **worse than a
coin flip** on the bets they selected. Anchoring removed the false confidence:
its probabilities now match the coin flip almost exactly. That is the absence
of a lie, not the presence of information.

### Calibration

Fitting Platt scaling on weeks 1-11 and scoring weeks 12-22 gives a slope of
**0.0034** — essentially zero, meaning the raw probability adds almost nothing
over the base rate. Held-out reliability, before anchoring:

```
     raw bin      n  raw pred   actual
  0%-40%   1361     0.247      0.489
  60%-70%    707     0.648      0.512
  80%-101%   398     0.933      0.550
```

Predictions spanning 25%-93% map to outcomes spanning 49%-55%.

**Watch the slope, not ROI.** It converges far faster. If a future feature is
real, the slope rises off 0.004 long before ROI can distinguish a 2% edge from
noise.

## Signals tested against the line — all null

Each was measured as "does this predict beating the line", the only question
that can produce value, on samples large enough to answer it.

| signal | sample | result |
|---|---|---|
| opponent defence, team yards allowed | 2,657 / 1,253 | +0.010 / +0.009 |
| opponent defence, by position | 1,371 WR | WR -0.010, TE +0.034, RB +0.024 |
| role change (`flag_role_changes`) | 3,069 | -0.017 |
| vacated opportunity (team-mate Out) | 14,579 pooled 2023-25 | +0.0003 |

Two produced tempting subgroups that did not replicate. Role change appeared
to work in reverse — players whose role had *fallen* beat their line 55.6%
against a 47.4% baseline — but split by half-season the effect collapsed from
+15.9pp to +4.1pp. Vacated opportunity showed +3.5pp in the 5-10% bucket while
the 10%+ bucket showed nothing, which is not how a real effect behaves.

**Test any new signal against a harvested CSV, split by season, before
building it.** Opponent adjustment would have cost half a day; it cost five
minutes.

### Closing line value does not work here either

CLV was wired in the hope of a metric that converges in hundreds of bets
rather than thousands. On 390 matched bets it reads **+1.13pp, t = +9.34** —
and it is worthless: `corr(CLV, won) = -0.026`, and the *highest* CLV quartile
has the worst win rate (40.8%) and worst ROI (-16.3%). A tightly clustered
mean with no outcome correlation is a systematic offset between the decision
snapshot and the close, not information.

Two traps worth remembering. The bet side takes the **best** price across
books, so the closing comparison must also take the best — comparing best-of-N
to median-of-N is positively biased before any information enters, and it
inflated the share of bets that appeared to beat the close from 53% to 71%.
And a minimum-price filter *raised* CLV (+1.24 to +1.57pp) while ROI collapsed
(-7.65% to -15.4%): better-priced bets performed worse, which is what a price
being good *for a reason* looks like.

## Design decisions worth knowing

**Anchored on the line, deviations argued.** Since the projection adds nothing
over the line, `projections/anchor.py` makes the line the centre — blending at
a fitted weight of 0.95 — and requires any departure to name a reason. A
`Deviation` without a reason raises; total movement is capped at 25%, because
a signal wanting to move a line further than that is far more likely a mapping
error than an insight. With no deviation applied the model agrees with the
market and declines to bet, which is correct behaviour for a model with nothing
to add.

Anchor the **median**, not the mean. A book sets its line where about half the
outcomes fall either side. Yardage distributions are right-skewed, so forcing
the *mean* onto the line drags the median well below it — anchoring mean = 60
left the median at 48 and `P(over 60) = 0.395`, so the model read unders as
60% shots on skew alone. That cost 15 points of ROI on one week before it was
caught.

**Distributions, not point estimates.** Unchanged and still right. A normal
fitted to a 45-yard receiver with a 38-yard standard deviation puts 12% of its
mass below zero; `test_normal_would_misprice_a_low_line` shows the size of the
resulting mispricing.

**Simulation, not isolated projections.** Passing yards are the sum of that
team's simulated receiving yards, so a QB stays consistent with his receivers.
`test_passing_yards_equal_team_receiving_yards` guards the identity, and it
earned its keep: a Dirichlet share draw that floored zero shares to keep the
distribution valid started handing targets to a blocking back and broke it.

**Per-book de-vigging.** Each book's price is de-vigged against that same
book's opposite side. Comparing a DraftKings over to a FanDuel-derived fair
probability smuggles the difference in their holds into the edge.

**Edges are shrunk before ranking, and large ones are quarantined.** Edges
above 18 points route to `needs_review`. The backtest vindicates this: in the
first season run the 15%+ bucket was 21% of all bets and returned -2.2%. Very
large edges are modelling errors, not opportunities.

**Look-ahead is blocked structurally — and one hole was found.** `through_week`
was applied only to the weekly frame, so pace and red-zone tendency were
measured over the *whole* season including weeks after the game. Fixed, with a
test that fails if a later week can change an earlier projection.

**Kelly is capped.** Quarter Kelly by default, hard-capped at 2% of bankroll.
The backtest uses flat staking: Kelly sizing on a miscalibrated probability
amplifies the calibration error rather than the edge.

## What running it live actually broke

Every one of these was invisible to a test suite built on synthetic frames.

**Schema drift in nflverse.** `recent_team` became `team`; depth charts key on
`gsis_id` not `player_id`, and rank as `pos_rank` not `depth_team`. Two of
three failed *silently* — a renamed optional column defaults to zero, and an
unrecognised rank column made `build_depth_ranks` return `{}`, quietly removing
the direct-backup bonus from injury redistribution.

**A fullback ranked as the starting running back.** Depth ranks are ranks
*within nflverse's own position label*, so a fullback is FB1. The projection
merges fullbacks into running backs, so FB1 read as RB1 and a blocking
specialist on 19% of snaps inherited the starting back's opportunity prior.
Ranks are now position-qualified and folded labels sort below primary ones.

**Injury statuses resurrected from months earlier.** `groupby().last()` returns
the last *non-null* value of each column, not the last row. A receiver listed
Questionable in week 3 and practising fully from week 11 came back as
"Questionable / Full Participation in Practice", and 72 of two teams' players
were marked injured. Take the last row.

**Absent players' opportunity was deleted, not redistributed.** The simulator
zeroed an inactive player's targets *after* allocation, so his share left the
game rather than passing to team-mates — a 27% haircut on every passing and
receiving projection, despite a comment claiming renormalisation happened.

**A fifth of all targets went to players who never dressed.** Rosters average
12.6 players of whom 9.2 record a stat line. Every one was treated as certain
to play, so ~3.4 players per team absorbed 20% of the targets. Availability is
now estimated from recency-weighted appearances over the last four games, and
`propagate_injuries` composes with it rather than overwriting it.

**Pace was pinned at its clamp for every team in the league.** `seconds_per_play`
took the *median* of clock gaps, but the clock stops on an incompletion and
runs after a run, so the gaps are bimodal — roughly 5s and roughly 40s — and
the median sits in the upper cluster. Every team measured 34-40s and clamped to
35. Now measured from drive time of possession over the plays it covers; MAE
against the truth fell from 7.9s to 1.6s.

**The play model's reference was on a different scale.** `expected_plays`
differenced measured pace against 27.5s while the estimator returns 29.7-35.0.
A league-average team was simulated at 53.6 plays against an actual 60.7.
Constants refit against all 32 teams: MAE 7.08 to 2.18 plays.

**Snap counts never joined.** They key on PFR ids while everything else uses
gsis; the merge silently found nothing and `snap_share` was 0.0 for every
player. Now crosswalked via `load_id_map()` at a 99.9% match rate — though it
changed no projection, because `snap_share` and `route_participation` are
computed and then never read by anything.

## Known limitations

**The projection adds nothing over the line.** Restated because it subsumes
most of what used to be listed here.

**Four features are computed and never consumed.** `snap_share`,
`route_participation`, `off_efficiency` / `def_efficiency_faced`, and
`flag_role_changes` are all calculated, stored, and read by nothing. The README
previously claimed route participation was approximated from snap counts and
that missing features lowered confidence; neither is true.

**Three requested features remain unavailable.** Route participation is
*approximated* from snap counts, not measured. True man/zone coverage splits
and offensive-line grades require PFF or SIS licensing. Given that every signal
derivable from public data has now tested null, this is the honest frontier:
the missing information is licensed, and acquiring it is a purchasing decision
rather than an engineering one.

**Books shade lines above the median outcome.** Measured across 14,579 pooled
lines from 2023-2025, the beat-the-line rate is **48.6%**, not 50%. Any signal
must clear the vig *and* that 1.4-point tilt.

**The touchdown model is partly calibrated** and was not revisited. The
residual described previously still stands.

**Weather fails inside the slice.** Preflight pulls `SEA: source=open-meteo,
wind=2mph` successfully; the slice reports "no weather forecast" for the same
stadium and falls back to neutral conditions. Open.

**Environment priors are priors.** Pace and the play model were refit from
2025 play-by-play; `plays_per_point_of_total` was not. `pace_play_sensitivity`
is weakly identified (r = -0.30 over one season) — at the previous value the
model reproduced the real spread of team play counts but was no more accurate
than predicting the league mean, so the extra spread was noise.

**Simulated distributions are too narrow, and the obvious fix failed.**
Measured against 2025, receiving yards were 1.41x too tight and rushing 1.35x,
and the ratios were stable across independent halves. A per-game lognormal
efficiency multiplier hit the target spread but produced *no* improvement in
calibration or ROI, because it adds skew as well as spread and pushed
probabilities further from 50%. Reverted. A mean-preserving deviation scaler
is the right instrument if this is retried.

## If you pick this up again

Test the information before building the machinery. `harvest_lines.py` plus a
correlation against the line answers "is there a signal here" in minutes, and
four such tests would have saved most of the work described above.

Judge changes by calibration slope, then ROI. The slope moved measurably
across changes that ROI could not distinguish from noise on 2,000 bets.

Do not trust a favourable metric that has not been checked against the outcome
it claims to predict. `corr(CLV, won)` is one line of code and it invalidated
the project's only positive-looking result — twice.
