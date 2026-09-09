# NFL Prop Engine

Player prop and touchdown projections compared against sportsbook markets, built
around one output:

> "63% chance of going over 81.5 yards while the market implies 51%"

rather than "projected for 94 yards."

Projections are probabilistic estimates with real uncertainty. No wager is
guaranteed, and historical performance does not predict future results.

## Status

Phase 1 of the build plan, plus the probability and edge math that phases 3-5
depend on. **192 unit tests pass**, verified against hand-computed values.

| Module | State |
|---|---|
| `core/odds.py` | Complete, tested |
| `core/distributions.py` | Complete, tested |
| `core/edge.py` | Complete, tested |
| `sim/game_sim.py` | Complete, tested — 10k sims in ~53ms/game |
| `projections/environment.py` | Complete, tested |
| `projections/injury_engine.py` | Complete, tested |
| `projections/explain.py` | Complete, untested |
| `backtest/engine.py` | Metrics and guards complete and tested; runner not written |
| `ingest/odds_api.py` | Written, **never run against the live API** |
| `ingest/nflverse.py` | Written, **never run against real data** |
| `db/schema.sql` | Complete, never applied to a live database |
| `ingest/player_matching.py` | Complete, tested |
| `ingest/teams.py` | Complete, tested |
| `ingest/context_loader.py` | Transformations complete and tested; `fetch_context` never run live |
| `ingest/weather.py` | Written, never run live |
| `scripts/preflight.py` | Validates every connection; run this first |
| `projections/usage_builder.py` | Complete, tested |
| `scripts/slice.py` | Runs end to end against fixtures; live path wired, never run |
| `main.py` | Routes typed and documented; all raise `NotImplementedError` |
| Frontend | Not started |
| ML ensemble | Not started |
| Celery scheduling | Not started |

## Setup

```bash
cp .env.example .env          # add your ODDS_API_KEY -- never commit this
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                        # or: python -m unittest discover -s tests -t .
```

The test suite needs only numpy, scipy and pandas. It does not touch the
network or the database. Postgres is not needed until you start storing
snapshots -- `schema.sql` is there when you are ready.

### Then, in this order

```bash
python scripts/preflight.py            # everything free, 0 API credits
python scripts/preflight.py --spend    # adds one props pull (costs credits)
python scripts/slice.py --event-id <id> --season 2026
```

`.env` is loaded automatically by both scripts (`app/env.py`, no dependency).
Variables already exported in your shell always win over the file, so a stale
checked-out `.env` cannot clobber a CI secret. `.env` is gitignored; keep it
that way, and rotate any key that has ever been pasted into a chat, ticket or
commit.

`preflight.py` checks each connection in increasing order of cost: env vars,
dependencies, the Odds API key (via `/events`, which costs nothing), the
nflverse loaders, and then the real column names against what the
transformation layer assumes. That last check is the one that earns the script
-- every transformation was tested against synthetic frames, and this is the
first thing that verifies the shape was right. It also checks that every team
code in the live data maps to a canonical code.

### Data sources

| Source | Key needed | Notes |
|---|---|---|
| The Odds API | yes, paid | Player props need a paid plan. Live event props cost markets x regions per game; historical *event* odds cost 1 credit, historical *bulk* odds cost 10 per region per market -- use the event endpoint for backtests. Historical player props go back to 2023-05-03; featured markets to mid-2020. |
| nflverse | no | Public. Via `nflreadpy`. |
| Open-Meteo | no | Hourly forecasts; domes skip the call entirely. |
| PostgreSQL | n/a | Only needed for stored snapshots and backtesting. |

## Design decisions worth knowing

**Distributions, not point estimates.** Every market is answered from a
distribution, and the family is chosen per stat. A normal fitted to a 45-yard
receiver with a 38-yard standard deviation puts 12% of its mass below zero;
`test_normal_would_misprice_a_low_line` demonstrates the size of the resulting
mispricing.

**Simulation, not isolated projections.** Passing yards are the sum of that
team's simulated receiving yards, so a QB is automatically consistent with his
receivers. Measured on a test slate: QB↔WR1 correlation +0.64, opposing backs'
carries −0.37. Touchdowns are allocated from simulated team scoring, so a back
on a 28-point team differs from one on a 16-point team with identical usage.

**Opportunity separated from efficiency.** Targets are drawn by share from
simulated team pass attempts; yards are drawn per reception. This is what makes
a projection decomposable back into *why*, and `projections/explain.py` refuses
to render a breakdown whose parts do not reconcile with the whole.

**Edges are shrunk before ranking.** A large edge on a thin market is weak
evidence of mispricing. Edges above 18 points route to `needs_review` rather
than the dashboard, because at that magnitude a bad player mapping or a stale
alternate line is more likely than a real opportunity.

**Per-book de-vigging.** Each book's price is de-vigged against that same
book's opposite side. Comparing a DraftKings over to a FanDuel-derived fair
probability smuggles the difference in their holds into the edge.

**Look-ahead bias is blocked structurally.** `PointInTimeReader` requires an
`as_of` timestamp and refuses to read any table not registered with an
observation column. `player_game_stats` and `player_props` are permanently
unreadable during a backtest.

**Kelly is capped.** Quarter Kelly by default, hard-capped at 2% of bankroll,
and `KELLY_FRACTION` above 0.5 is rejected at config load.

## Known limitations

**Three requested features are not publicly available.** Route participation is
*approximated* from snap counts and target data, not measured. True man/zone
coverage splits and offensive-line grades require PFF or SIS licensing.
`FeatureAvailability` records which feature groups are present, and projections
built without them are marked lower-confidence rather than treating a missing
feature as zero.

**Backtesting player props needs a paid Odds API plan.** Historical player
markets are available from May 2023 onward on paid tiers only.

**The touchdown model is partly calibrated, and the residual is known.** Two
structural bugs were fixed: team touchdowns were drawn as a Poisson on top of an
already-random score (double-counting variance, all of which landed on the 2+
market), and every touchdown was allocated by goal-line share as though no score
ever came from outside the five. Allocation is now sequential with a repeat
penalty fit at 0.75.

What remains: a single scalar penalty cannot fix the *shape* of the error. The
model is close to unbiased on average but still runs about five points high on
2+ for elite scorers (anytime above ~0.65) and slightly low for marginal ones.
Run `scripts/calibrate_td.py` to see the current fit.

The bigger caveat is that this was fit against a hand-entered reference curve of
typical NFL prices, not scraped data. Wire `load_market_pairs()` in that script
to real de-vigged anytime/2+ pairs from `odds_snapshots` before trusting the
number.

**Pace and red-zone tendency are now measured, not assumed.** Both were
hardcoded; `seconds_per_play` reads the game clock in neutral situations only,
and `red_zone_rush_rate` reads actual red-zone play calls. Both fall back to
league averages on thin samples and flag it in `_pace_is_approximate`.

**Environment priors are priors, not fits.** Every coefficient in
`EnvironmentPriors` is a reasonable starting value chosen to be in the right
region for modern NFL football. They are meant to be refit from play-by-play
before the model is trusted.

**Nothing has touched live data.** The ingestion layer was written without
network access. Expect to debug player-name matching against real Odds API
responses first — that is where the schedule risk is.

## Vertical slice

```bash
cd backend
python scripts/slice.py --fixture              # offline, no key, no credits
python scripts/slice.py --event-id <id>        # live, needs ODDS_API_KEY
```

Runs the whole chain for one game — odds, player resolution, usage, injury
propagation, simulation, de-vig, edge, ranking — and prints a table. Fixture
mode runs against a checked-in payload shaped like a real Odds API response, so
the wiring is proven before a key is involved.

Two numbers to watch on the first live run:

- **Player match rate.** Below ~90% means fix the resolver before trusting any
  edge on the slate.
- **Edge distribution.** A median absolute edge above ~10 points is a
  calibration problem, not a slate full of value. The slice prints a warning
  when it sees one.

### What the slice already caught

Opportunity shares were being shrunk toward zero. Because that applies the same
multiplicative factor to every player on a team, the team-level renormalisation
cancelled it exactly — so the shrinkage was a no-op and a two-game sample was
trusted as much as a twelve-game one. Shares are now shrunk toward depth-chart
priors (`OPPORTUNITY_PRIORS`), which survives renormalisation.
`test_shrinkage_toward_zero_would_be_cancelled_by_renormalisation` guards it.

### Fetch and transform are separated

`ingest/context_loader.py` splits into `fetch_*` (network, thin, untestable
offline) and `build_*` (pure DataFrame transformations, fully tested against
synthetic nflverse-shaped frames). When the live run breaks, that split answers
"data problem or logic problem" immediately, because the logic already has
passing tests.

Two failure modes the transformations handle explicitly:

- **Team codes.** `ingest/teams.py` normalises full names, nicknames, provider
  spellings (`JAC`/`JAX`, `WSH`/`WAS`, `GNB`/`GB`) and relocations
  (`OAK`→`LV`, `SD`→`LAC`, `STL`→`LAR`). Historical rows keep the old codes, so
  a backtest joining on raw codes would silently drop every pre-relocation game
  for three franchises. Unknown identifiers raise rather than resolving to
  something plausible.
- **Missing columns.** Optional columns default to zero; required ones raise. A
  game log with no target count is not a game log with zero targets, and
  treating it as one projects a healthy receiver at zero.

## Next step

Run the slice live against one game:

```bash
python scripts/slice.py --event-id <id> --season 2026
```

Expect the first failure to be a schema mismatch — a renamed nflverse column or
a depth-chart field that is a string where an int was assumed. The
transformation tests will tell you which function to look in.
