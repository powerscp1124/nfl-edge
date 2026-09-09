"""FastAPI application.

Endpoints are read-heavy and thin: every one of them reads a stored projection
or market row rather than computing on request. Projections are produced by the
Celery workers and written to ``player_projections`` with a model version; the
API never runs a simulation inline, because a request that silently recomputes
a projection would return a number that does not match the one the user saw a
minute ago and that no stored row can explain.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, get_settings

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Projections are probabilistic estimates with real uncertainty. No wager "
    "is guaranteed, and historical performance does not predict future "
    "results."
)


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class Distribution(BaseModel):
    mean: float
    median: float
    std_dev: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    family: str
    n_simulations: int | None = None


class BookLine(BaseModel):
    bookmaker: str
    line: float | None
    over_odds: float | None = None
    under_odds: float | None = None
    last_update: datetime | None = None


class PropRow(BaseModel):
    player_id: str
    player_name: str
    team: str
    opponent: str
    position: str
    is_home: bool
    game_id: str
    kickoff: datetime
    market: str
    consensus_line: float | None
    best_over: BookLine | None
    best_under: BookLine | None
    n_books: int
    projection: Distribution
    prob_over: float
    prob_under: float
    fair_over_odds: float
    fair_under_odds: float
    edge: float
    ev: float
    confidence: float
    confidence_parts: dict[str, float]
    grade: str
    recommendation: str
    reasons: list[str]
    model_version: str
    generated_at: datetime


class TouchdownRow(BaseModel):
    player_id: str
    player_name: str
    team: str
    opponent: str
    position: str
    p0: float
    p1: float
    p2: float
    p3: float
    p4_plus: float
    anytime_model: float
    anytime_market: float | None
    anytime_fair_odds: float
    anytime_best_odds: float | None
    anytime_best_book: str | None
    anytime_edge: float | None
    two_plus_model: float
    two_plus_market: float | None
    two_plus_fair_odds: float
    two_plus_best_odds: float | None
    two_plus_edge: float | None
    projected_rz_touches: float | None
    projected_gl_touches: float | None
    goal_line_share: float | None
    team_implied_total: float | None
    projected_team_tds: float | None
    confidence: float
    model_version: str


class ThresholdPoint(BaseModel):
    threshold: float
    model_prob: float
    market_prob: float | None = None
    best_odds: float | None = None
    best_book: str | None = None
    edge: float | None = None


class GameEnvironmentOut(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    spread_home: float | None
    total: float | None
    implied_home: float | None
    implied_away: float | None
    projected_plays_home: float | None
    projected_plays_away: float | None
    projected_pass_attempts_home: float | None
    projected_pass_attempts_away: float | None
    projected_rush_attempts_home: float | None
    projected_rush_attempts_away: float | None
    weather: dict | None
    injury_summary: list[dict]
    most_affected: list[dict]


class ModelPerformance(BaseModel):
    window: str
    n_bets: int
    roi: float
    win_rate: float
    units: float
    avg_clv: float
    brier: float
    log_loss: float
    calibration_error: float


class DataHealth(BaseModel):
    service: str
    last_success: datetime | None
    rows_written: int | None
    status: str
    staleness_minutes: float | None
    api_credits_remaining: int | None = None
    errors_24h: int = 0


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Starting API, model registry at %s", settings.database_url_safe)
    yield


app = FastAPI(
    title="NFL Prop Engine",
    version="0.1.0",
    description=(
        "Player prop and touchdown projections compared against sportsbook "
        "markets. " + DISCLAIMER
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Market = Literal["passing_yards", "rushing_yards", "receiving_yards"]


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc)}


# --- Games ----------------------------------------------------------------- #
@app.get("/games")
def list_games(
    season: int | None = None,
    week: int | None = None,
    settings: Settings = Depends(get_settings),
):
    """Scheduled games with market lines attached."""
    raise NotImplementedError("Wire to repositories.games.list_games")


@app.get("/games/{game_id}/environment", response_model=GameEnvironmentOut)
def game_environment(game_id: str):
    raise NotImplementedError("Wire to repositories.games.environment")


# --- Props ----------------------------------------------------------------- #
@app.get("/props", response_model=list[PropRow])
def list_props(
    market: Market | None = None,
    team: str | None = None,
    position: str | None = None,
    game_id: str | None = None,
    min_edge: float = Query(0.0, ge=-1.0, le=1.0),
    min_confidence: float = Query(0.0, ge=0.0, le=100.0),
    recommendation: str | None = None,
    sort: Literal["edge", "ev", "confidence", "rank"] = "rank",
    limit: int = Query(200, ge=1, le=1000),
):
    """The player props board.

    Defaults to the blended ranking rather than raw edge, because sorting by
    edge alone surfaces the thinnest markets first.
    """
    raise NotImplementedError("Wire to repositories.props.list_props")


@app.get("/props/{prop_id}")
def get_prop(prop_id: str):
    raise NotImplementedError("Wire to repositories.props.get_prop")


@app.get("/props/{prop_id}/curve", response_model=list[ThresholdPoint])
def prop_threshold_curve(prop_id: str):
    """Model probability at each alternate line, against the book's price."""
    raise NotImplementedError("Wire to repositories.props.threshold_curve")


@app.get("/props/{prop_id}/movement")
def prop_movement(prop_id: str):
    """Line and projection history, for the movement chart."""
    raise NotImplementedError("Wire to repositories.market.movement")


# --- Touchdowns ------------------------------------------------------------ #
@app.get("/touchdowns", response_model=list[TouchdownRow])
def list_touchdowns(
    market: Literal["anytime", "two_plus", "both"] = "both",
    sort: Literal["probability", "edge", "ev"] = "edge",
    position: str | None = None,
    game_id: str | None = None,
    min_edge: float = Query(0.0, ge=-1.0, le=1.0),
    limit: int = Query(200, ge=1, le=1000),
):
    """Touchdown board.

    Highest probability, highest edge and highest EV are three different
    orderings and the API keeps them distinct: a 70% scorer at -190 is a worse
    bet than a 45% scorer at +165, and sorting by probability would bury the
    second one.
    """
    raise NotImplementedError("Wire to repositories.touchdowns.list")


# --- Players --------------------------------------------------------------- #
@app.get("/players")
def list_players(team: str | None = None, position: str | None = None,
                 search: str | None = None):
    raise NotImplementedError("Wire to repositories.players.list_players")


@app.get("/players/{player_id}")
def player_profile(player_id: str):
    """Usage, form, matchup, projection and model history for one player."""
    raise NotImplementedError("Wire to repositories.players.profile")


# --- Model ----------------------------------------------------------------- #
@app.get("/model/projections")
def model_projections(game_id: str | None = None, market: Market | None = None):
    raise NotImplementedError("Wire to repositories.projections.list")


@app.get("/model/edges")
def model_edges(min_edge: float = 0.03, min_confidence: float = 60.0,
                limit: int = 50):
    raise NotImplementedError("Wire to repositories.projections.edges")


@app.get("/model/performance", response_model=list[ModelPerformance])
def model_performance(window: Literal["7d", "30d", "season", "all"] = "season",
                      market: str | None = None):
    raise NotImplementedError("Wire to repositories.results.performance")


@app.get("/model/calibration")
def model_calibration(market: str, season: int | None = None):
    raise NotImplementedError("Wire to repositories.results.calibration")


@app.get("/model/feature-importance")
def feature_importance(market: str, model_version: str | None = None):
    """Importances read from the trained model, never hand-authored."""
    raise NotImplementedError("Wire to repositories.models.feature_importance")


@app.get("/model/versions")
def model_versions():
    raise NotImplementedError("Wire to repositories.models.versions")


@app.post("/model/run", status_code=202)
def run_projections(game_id: str | None = None, n_sims: int = 10_000):
    """Queue a projection run. Returns a task id; results land in the tables."""
    raise NotImplementedError("Dispatch tasks.projections.run")


@app.post("/model/train", status_code=202)
def train_models(market: str, seasons: list[int]):
    raise NotImplementedError("Dispatch tasks.training.train")


@app.post("/backtest/run", status_code=202)
def run_backtest(seasons: list[int], decision_offset_minutes: int = 1440):
    raise NotImplementedError("Dispatch tasks.backtest.run")


# --- Market data ----------------------------------------------------------- #
@app.get("/odds")
def odds(game_id: str | None = None, market: str | None = None,
         bookmaker: str | None = None):
    raise NotImplementedError("Wire to repositories.market.current")


@app.get("/injuries")
def injuries(team: str | None = None, game_id: str | None = None):
    raise NotImplementedError("Wire to repositories.injuries.current")


@app.get("/weather")
def weather(game_id: str | None = None):
    raise NotImplementedError("Wire to repositories.weather.current")


# --- Bet tracking ---------------------------------------------------------- #
class BetIn(BaseModel):
    game_id: str
    player_id: str
    stat: str
    side: Literal["over", "under", "yes"]
    line: float | None = None
    bookmaker: str
    american_odds: float
    stake_units: float = Field(1.0, gt=0, le=10)
    staking_method: Literal["flat", "kelly", "custom"] = "flat"


@app.post("/bets", status_code=201)
def create_bet(bet: BetIn):
    raise NotImplementedError("Wire to repositories.bets.create")


@app.get("/bets")
def list_bets(settled: bool | None = None, market: str | None = None):
    raise NotImplementedError("Wire to repositories.bets.list")


@app.get("/bets/performance")
def bet_performance():
    raise NotImplementedError("Wire to repositories.bets.performance")


# --- Admin ----------------------------------------------------------------- #
@app.get("/admin/data-health", response_model=list[DataHealth])
def data_health():
    raise NotImplementedError("Wire to repositories.ops.health")


@app.get("/admin/validation-failures")
def validation_failures(resolved: bool = False):
    raise NotImplementedError("Wire to repositories.ops.validation_failures")


@app.post("/admin/refresh", status_code=202)
def refresh(service: Literal["odds", "injuries", "weather", "stats",
                             "depth_charts", "all"]):
    raise NotImplementedError("Dispatch tasks.ingest.refresh")
