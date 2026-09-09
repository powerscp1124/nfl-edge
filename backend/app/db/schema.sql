-- NFL prop engine schema.
--
-- Design rule: historical data is never overwritten. Odds, projections,
-- injuries and depth charts are all append-only with an observation timestamp,
-- because the backtester has to be able to reconstruct exactly what was known
-- at an arbitrary point in the past. An UPDATE on any of those tables would
-- silently destroy the ability to measure closing-line value.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fallback fuzzy name matching

-- ===========================================================================
-- Reference
-- ===========================================================================
CREATE TABLE teams (
    team_id         TEXT PRIMARY KEY,        -- canonical: 'KC', 'LAR'
    full_name       TEXT NOT NULL,
    conference      TEXT NOT NULL CHECK (conference IN ('AFC', 'NFC')),
    division        TEXT NOT NULL,
    stadium         TEXT,
    is_dome         BOOLEAN NOT NULL DEFAULT FALSE,
    surface         TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    timezone        TEXT
);

-- One row per human being. Provider IDs are columns, not separate rows, so a
-- join never has to guess which "M. Thomas" it is holding.
CREATE TABLE players (
    player_id       TEXT PRIMARY KEY,        -- canonical, we use gsis_id
    gsis_id         TEXT UNIQUE,
    espn_id         TEXT,
    pfr_id          TEXT,
    sleeper_id      TEXT,
    yahoo_id        TEXT,
    pff_id          TEXT,
    odds_api_name   TEXT,                    -- name as The Odds API spells it
    full_name       TEXT NOT NULL,
    normalized_name TEXT NOT NULL,           -- lowercase, punctuation stripped
    position        TEXT,
    team_id         TEXT REFERENCES teams(team_id),
    status          TEXT,                    -- ACT / INA / IR / PUP / SUS
    height_inches   INTEGER,
    weight_lbs      INTEGER,
    birth_date      DATE,
    rookie_year     INTEGER,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX players_normalized_name_idx ON players USING gin (normalized_name gin_trgm_ops);
CREATE INDEX players_team_idx ON players (team_id, position);

-- Unresolved names from any provider. Every one of these blocks a projection
-- rather than being guessed at, and lands here for a human to map.
CREATE TABLE player_alias_queue (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    raw_name        TEXT NOT NULL,
    team_hint       TEXT,
    position_hint   TEXT,
    best_guess_id   TEXT REFERENCES players(player_id),
    match_score     DOUBLE PRECISION,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, raw_name, team_hint)
);

CREATE TABLE games (
    game_id         TEXT PRIMARY KEY,
    odds_api_event_id TEXT UNIQUE,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    season_type     TEXT NOT NULL DEFAULT 'REG',
    home_team       TEXT NOT NULL REFERENCES teams(team_id),
    away_team       TEXT NOT NULL REFERENCES teams(team_id),
    kickoff         TIMESTAMPTZ NOT NULL,
    stadium         TEXT,
    roof            TEXT,
    surface         TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (home_team <> away_team)
);
CREATE INDEX games_kickoff_idx ON games (kickoff);
CREATE INDEX games_season_week_idx ON games (season, week);

-- ===========================================================================
-- Observed state (append-only)
-- ===========================================================================
CREATE TABLE depth_charts (
    id              BIGSERIAL PRIMARY KEY,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,           -- espn / nflverse / thetwodeep
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    position        TEXT NOT NULL,
    depth_rank      INTEGER NOT NULL,
    package         TEXT,                    -- base / nickel / goal_line
    season          INTEGER,
    week            INTEGER
);
CREATE INDEX depth_charts_lookup_idx ON depth_charts (team_id, observed_at DESC);

CREATE TABLE injuries (
    id              BIGSERIAL PRIMARY KEY,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    team_id         TEXT REFERENCES teams(team_id),
    season          INTEGER,
    week            INTEGER,
    report_status   TEXT,                    -- Out / Doubtful / Questionable
    practice_status TEXT,                    -- DNP / Limited / Full
    body_part       TEXT,
    is_final_report BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX injuries_player_idx ON injuries (player_id, observed_at DESC);

CREATE TABLE weather (
    id              BIGSERIAL PRIMARY KEY,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    forecast_for    TIMESTAMPTZ NOT NULL,
    temperature_f   DOUBLE PRECISION,
    wind_mph        DOUBLE PRECISION,
    wind_bearing    DOUBLE PRECISION,
    precipitation_prob DOUBLE PRECISION,
    precipitation_type TEXT,
    humidity        DOUBLE PRECISION,
    is_dome         BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX weather_game_idx ON weather (game_id, observed_at DESC);

-- ===========================================================================
-- Statistics
-- ===========================================================================
CREATE TABLE player_game_stats (
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    opponent_id     TEXT NOT NULL REFERENCES teams(team_id),
    is_home         BOOLEAN NOT NULL,
    snaps           INTEGER,
    snap_share      DOUBLE PRECISION,
    routes_run      INTEGER,
    route_participation DOUBLE PRECISION,
    targets         INTEGER,
    receptions      INTEGER,
    receiving_yards DOUBLE PRECISION,
    air_yards       DOUBLE PRECISION,
    yac             DOUBLE PRECISION,
    target_share    DOUBLE PRECISION,
    air_yards_share DOUBLE PRECISION,
    adot            DOUBLE PRECISION,
    carries         INTEGER,
    rushing_yards   DOUBLE PRECISION,
    rush_share      DOUBLE PRECISION,
    yards_before_contact DOUBLE PRECISION,
    yards_after_contact  DOUBLE PRECISION,
    pass_attempts   INTEGER,
    completions     INTEGER,
    passing_yards   DOUBLE PRECISION,
    sacks_taken     INTEGER,
    rz_targets      INTEGER,
    rz_carries      INTEGER,
    ez_targets      INTEGER,
    inside_5_carries INTEGER,
    rush_td         INTEGER,
    rec_td          INTEGER,
    pass_td         INTEGER,
    total_td        INTEGER,
    source          TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX pgs_player_idx ON player_game_stats (player_id);

CREATE TABLE team_game_stats (
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    opponent_id     TEXT NOT NULL REFERENCES teams(team_id),
    is_home         BOOLEAN NOT NULL,
    plays           INTEGER,
    pass_attempts   INTEGER,
    rush_attempts   INTEGER,
    dropbacks       INTEGER,
    seconds_per_play DOUBLE PRECISION,
    neutral_pass_rate DOUBLE PRECISION,
    proe            DOUBLE PRECISION,
    off_epa_play    DOUBLE PRECISION,
    def_epa_play    DOUBLE PRECISION,
    rush_epa_allowed DOUBLE PRECISION,
    pass_epa_allowed DOUBLE PRECISION,
    pressure_rate   DOUBLE PRECISION,
    sack_rate       DOUBLE PRECISION,
    stuff_rate      DOUBLE PRECISION,
    explosive_pass_rate DOUBLE PRECISION,
    explosive_run_rate  DOUBLE PRECISION,
    red_zone_trips  INTEGER,
    red_zone_td_rate DOUBLE PRECISION,
    red_zone_rush_rate DOUBLE PRECISION,
    points          INTEGER,
    touchdowns      INTEGER,
    source          TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, team_id)
);

CREATE TABLE advanced_player_stats (
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    metric_group    TEXT NOT NULL,           -- ngs_receiving / ngs_passing / pff
    metrics         JSONB NOT NULL,
    source          TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, player_id, metric_group)
);

CREATE TABLE advanced_team_stats (
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    metric_group    TEXT NOT NULL,
    metrics         JSONB NOT NULL,
    source          TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, team_id, metric_group)
);

-- ===========================================================================
-- Market
-- ===========================================================================
-- Every poll appends. This table grows fast (a full slate with alternates is
-- roughly 200k rows per week) so it is partitioned by month and the analytical
-- views read from the partition, not the parent.
CREATE TABLE odds_snapshots (
    id              BIGSERIAL,
    snapshot_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id         TEXT REFERENCES games(game_id),
    odds_api_event_id TEXT NOT NULL,
    bookmaker       TEXT NOT NULL,
    market_key      TEXT NOT NULL,
    stat            TEXT NOT NULL,
    player_id       TEXT REFERENCES players(player_id),
    raw_player_name TEXT NOT NULL,
    side            TEXT NOT NULL,
    line            DOUBLE PRECISION,        -- NULL for anytime TD, not 0
    american_odds   DOUBLE PRECISION NOT NULL,
    decimal_odds    DOUBLE PRECISION NOT NULL,
    raw_prob        DOUBLE PRECISION NOT NULL,
    is_alternate    BOOLEAN NOT NULL DEFAULT FALSE,
    book_last_update TIMESTAMPTZ,
    PRIMARY KEY (id, snapshot_at)
) PARTITION BY RANGE (snapshot_at);

CREATE INDEX odds_lookup_idx ON odds_snapshots
    (odds_api_event_id, stat, player_id, snapshot_at DESC);

-- Materialised current board. Refreshed after each poll; the snapshot table
-- remains the source of truth.
CREATE TABLE player_props (
    id              BIGSERIAL PRIMARY KEY,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    consensus_line  DOUBLE PRECISION,
    median_line     DOUBLE PRECISION,
    best_over_book  TEXT,
    best_over_line  DOUBLE PRECISION,
    best_over_odds  DOUBLE PRECISION,
    best_under_book TEXT,
    best_under_line DOUBLE PRECISION,
    best_under_odds DOUBLE PRECISION,
    n_books         INTEGER NOT NULL,
    opening_line    DOUBLE PRECISION,
    closing_line    DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (game_id, player_id, stat)
);

CREATE TABLE line_movements (
    id              BIGSERIAL PRIMARY KEY,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    bookmaker       TEXT NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    line            DOUBLE PRECISION,
    over_odds       DOUBLE PRECISION,
    under_odds      DOUBLE PRECISION,
    no_vig_over     DOUBLE PRECISION,
    minutes_to_kickoff DOUBLE PRECISION
);
CREATE INDEX line_movements_idx ON line_movements
    (game_id, player_id, stat, observed_at);

-- ===========================================================================
-- Model
-- ===========================================================================
CREATE TABLE model_versions (
    model_version   TEXT PRIMARY KEY,        -- NFL-2026-WEEK-01-V3
    market          TEXT NOT NULL,
    algorithm       TEXT NOT NULL,
    training_start  DATE NOT NULL,
    training_end    DATE NOT NULL,
    feature_list    JSONB NOT NULL,
    hyperparameters JSONB NOT NULL,
    ensemble_weights JSONB,
    data_sources    JSONB NOT NULL,
    validation_metrics JSONB,
    git_sha         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE player_projections (
    id              BIGSERIAL PRIMARY KEY,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    mean            DOUBLE PRECISION NOT NULL,
    median          DOUBLE PRECISION NOT NULL,
    std_dev         DOUBLE PRECISION NOT NULL,
    p10             DOUBLE PRECISION NOT NULL,
    p25             DOUBLE PRECISION NOT NULL,
    p50             DOUBLE PRECISION NOT NULL,
    p75             DOUBLE PRECISION NOT NULL,
    p90             DOUBLE PRECISION NOT NULL,
    distribution_family TEXT NOT NULL,
    n_simulations   INTEGER,
    -- Opportunity and efficiency stored separately so a projection can always
    -- be decomposed into why it landed where it did.
    expected_opportunity DOUBLE PRECISION,
    expected_efficiency  DOUBLE PRECISION,
    feature_contributions JSONB,
    confidence      DOUBLE PRECISION,
    confidence_parts JSONB,
    data_sufficient BOOLEAN NOT NULL DEFAULT TRUE,
    insufficiency_reason TEXT
);
CREATE INDEX projections_lookup_idx ON player_projections
    (game_id, player_id, stat, generated_at DESC);

CREATE TABLE touchdown_projections (
    id              BIGSERIAL PRIMARY KEY,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    p0 DOUBLE PRECISION NOT NULL,
    p1 DOUBLE PRECISION NOT NULL,
    p2 DOUBLE PRECISION NOT NULL,
    p3 DOUBLE PRECISION NOT NULL,
    p4_plus DOUBLE PRECISION NOT NULL,
    anytime_td      DOUBLE PRECISION NOT NULL,
    two_plus_td     DOUBLE PRECISION NOT NULL,
    expected_td     DOUBLE PRECISION NOT NULL,
    projected_rz_touches DOUBLE PRECISION,
    projected_gl_touches DOUBLE PRECISION,
    goal_line_share DOUBLE PRECISION,
    ez_target_share DOUBLE PRECISION,
    team_implied_total DOUBLE PRECISION,
    projected_team_tds DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    CHECK (two_plus_td <= anytime_td)
);

CREATE TABLE model_predictions (
    id              BIGSERIAL PRIMARY KEY,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    side            TEXT NOT NULL,
    line            DOUBLE PRECISION,
    bookmaker       TEXT NOT NULL,
    american_odds   DOUBLE PRECISION NOT NULL,
    model_prob      DOUBLE PRECISION NOT NULL,
    market_prob     DOUBLE PRECISION NOT NULL,
    devig_method    TEXT NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,
    ev              DOUBLE PRECISION NOT NULL,
    fair_odds       DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    grade           TEXT NOT NULL,
    recommendation  TEXT NOT NULL,
    kelly_fraction  DOUBLE PRECISION,
    reasons         JSONB,
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    -- Populated after the game so calibration can be measured.
    actual_value    DOUBLE PRECISION,
    outcome         TEXT
);
CREATE INDEX predictions_edge_idx ON model_predictions (generated_at DESC, edge DESC);

-- ===========================================================================
-- Evaluation
-- ===========================================================================
CREATE TABLE backtests (
    backtest_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    seasons         INTEGER[] NOT NULL,
    decision_offset_minutes INTEGER NOT NULL, -- 10080 = 7 days before kickoff
    market          TEXT,
    n_bets          INTEGER NOT NULL,
    roi             DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    avg_edge        DOUBLE PRECISION,
    avg_clv         DOUBLE PRECISION,
    brier_score     DOUBLE PRECISION,
    log_loss        DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    mae             DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    config          JSONB NOT NULL
);

CREATE TABLE bets (
    bet_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    placed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id         TEXT,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    side            TEXT NOT NULL,
    line            DOUBLE PRECISION,
    bookmaker       TEXT NOT NULL,
    american_odds   DOUBLE PRECISION NOT NULL,
    model_prob      DOUBLE PRECISION NOT NULL,
    market_prob     DOUBLE PRECISION NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,
    fair_odds       DOUBLE PRECISION NOT NULL,
    stake_units     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    staking_method  TEXT NOT NULL DEFAULT 'flat',
    is_theoretical  BOOLEAN NOT NULL DEFAULT TRUE,
    result          TEXT,                    -- win / loss / push / void
    actual_value    DOUBLE PRECISION,
    profit_units    DOUBLE PRECISION,
    closing_odds    DOUBLE PRECISION,
    closing_line    DOUBLE PRECISION,
    clv             DOUBLE PRECISION,
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    settled_at      TIMESTAMPTZ
);
CREATE INDEX bets_user_idx ON bets (user_id, placed_at DESC);

-- ===========================================================================
-- Operations
-- ===========================================================================
CREATE TABLE ingestion_runs (
    run_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service         TEXT NOT NULL,           -- odds / injuries / weather / stats
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running',
    rows_written    INTEGER,
    api_credits_used INTEGER,
    error_message   TEXT,
    details         JSONB
);

CREATE TABLE data_validation_failures (
    id              BIGSERIAL PRIMARY KEY,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rule            TEXT NOT NULL,
    severity        TEXT NOT NULL,           -- warn / error / block
    entity_type     TEXT,
    entity_id       TEXT,
    message         TEXT NOT NULL,
    payload         JSONB,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE
);

-- Projection movement over time, for the change-detection panel.
CREATE TABLE projection_changes (
    id              BIGSERIAL PRIMARY KEY,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    player_id       TEXT NOT NULL REFERENCES players(player_id),
    stat            TEXT NOT NULL,
    previous_mean   DOUBLE PRECISION NOT NULL,
    new_mean        DOUBLE PRECISION NOT NULL,
    delta           DOUBLE PRECISION NOT NULL,
    trigger         TEXT NOT NULL,           -- injury / depth_chart / odds / weather
    trigger_detail  JSONB
);
