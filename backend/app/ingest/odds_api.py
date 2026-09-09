"""The Odds API client.

Player prop markets live on the per-event endpoint, not the bulk odds endpoint,
so a full slate costs one request per game per market group. Quota discipline
matters: the credit cost of an event-odds request scales with the number of
markets and regions requested, so markets are batched and the poll interval
widens the further out from kickoff we are.

Nothing here is cached in memory. Every response is written to
``odds_snapshots`` as an immutable row, because closing-line value and line
movement analysis both depend on never overwriting an earlier observation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

# httpx is only needed to make requests. Importing it lazily keeps the
# normalisation helpers usable in environments that only parse stored payloads
# -- backtests, fixtures, and the offline slice.
if TYPE_CHECKING:  # pragma: no cover
    import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Only the markets this application projects, plus the supporting markets used
# as model inputs. Receptions, attempts and completions are ingested because
# they help validate the opportunity model, but they are never surfaced as
# primary picks.
PRIMARY_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_anytime_td",
    "player_tds_over",
]
ALTERNATE_MARKETS = [
    "player_pass_yds_alternate",
    "player_rush_yds_alternate",
    "player_reception_yds_alternate",
]
SUPPORTING_MARKETS = [
    "player_receptions",
    "player_pass_attempts",
    "player_pass_completions",
    "player_rush_attempts",
]
GAME_MARKETS = ["h2h", "spreads", "totals"]

MARKET_TO_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_yds_alternate": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_rush_yds_alternate": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_reception_yds_alternate": "receiving_yards",
    "player_anytime_td": "anytime_td",
    "player_tds_over": "multi_td",
}

DISPLAY_MARKETS = {"passing_yards", "rushing_yards", "receiving_yards",
                   "anytime_td", "multi_td"}


class OddsAPIError(RuntimeError):
    pass


class QuotaExhausted(OddsAPIError):
    pass


@dataclass
class QuotaState:
    """Mirrors the x-requests-* response headers so jobs can back off."""

    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None
    checked_at: datetime | None = None

    def update(self, headers) -> None:
        def _int(key: str) -> int | None:
            value = headers.get(key)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        self.remaining = _int("x-requests-remaining")
        self.used = _int("x-requests-used")
        self.last_cost = _int("x-requests-last")
        self.checked_at = datetime.now(timezone.utc)


@dataclass
class PropQuote:
    """One normalised price row, ready to insert into ``odds_snapshots``."""

    event_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmaker: str
    market_key: str
    stat: str
    player_name: str
    side: str              # over / under / yes
    line: float | None
    american: float
    last_update: datetime
    snapshot_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_alternate: bool = False

    @property
    def is_display_market(self) -> bool:
        return self.stat in DISPLAY_MARKETS


class OddsAPIClient:
    """Thin, quota-aware wrapper.

    The API key is read from the environment and never logged, never persisted,
    and never returned in an error message.
    """

    def __init__(
        self,
        api_key: str | None = None,
        regions: str = "us,us2",
        timeout: float = 20.0,
        client: "httpx.Client | None" = None,
    ) -> None:
        import httpx  # noqa: PLC0415

        self._httpx = httpx
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise OddsAPIError(
                "ODDS_API_KEY is not set. Add it to your environment or .env "
                "file; it must never be committed to the repository."
            )
        self.regions = regions
        self._client = client or httpx.Client(timeout=timeout)
        self.quota = QuotaState()

    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: dict[str, Any]) -> Any:
        params = {**params, "apiKey": self.api_key}
        try:
            resp = self._client.get(f"{BASE_URL}{path}", params=params)
        except self._httpx.HTTPError as exc:
            raise OddsAPIError(f"Request to {path} failed: {exc}") from exc

        self.quota.update(resp.headers)

        if resp.status_code == 401:
            raise OddsAPIError("The Odds API rejected the key (401).")
        if resp.status_code == 429:
            raise QuotaExhausted(
                f"Rate limited. Remaining credits: {self.quota.remaining}"
            )
        if resp.status_code >= 400:
            raise OddsAPIError(
                f"{resp.status_code} from {path}: {resp.text[:300]}"
            )
        return resp.json()

    # ------------------------------------------------------------------ #
    def list_events(self, days_ahead: int = 8) -> list[dict]:
        """Upcoming NFL events. Costs zero credits."""
        return self._get(f"/sports/{SPORT}/events", {"daysFrom": days_ahead})

    def game_lines(self) -> list[dict]:
        """Spreads and totals for the slate. One request for every game."""
        return self._get(
            f"/sports/{SPORT}/odds",
            {
                "regions": self.regions,
                "markets": ",".join(GAME_MARKETS),
                "oddsFormat": "american",
            },
        )

    def event_props(
        self,
        event_id: str,
        markets: Iterable[str] | None = None,
        odds_format: str = "american",
    ) -> dict:
        """Player props for one event.

        Credit cost is roughly ``len(markets) * len(regions)``, so keep the
        market list to what the model actually consumes.
        """
        markets = list(markets or PRIMARY_MARKETS)
        return self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            {
                "regions": self.regions,
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
            },
        )

    def historical_events(self, snapshot_iso: str) -> list[dict]:
        """The slate as it stood at a past timestamp.

        Needed to turn a historical game into the event id the props endpoint
        wants. Cheap for the same reason ``list_events`` is: no odds cross the
        wire, only the fixture list.
        """
        payload = self._get(
            f"/historical/sports/{SPORT}/events",
            {"date": snapshot_iso},
        )
        # The historical envelope wraps the slate in a "data" key.
        if isinstance(payload, dict):
            return payload.get("data", []) or []
        return payload or []

    def historical_event_props(
        self,
        event_id: str,
        snapshot_iso: str,
        markets: Iterable[str] | None = None,
    ) -> dict:
        """Historical odds as of a timestamp, for backtesting.

        Player markets are available from May 2023 on paid plans. The backtest
        engine calls this with the *decision* timestamp, never with kickoff, so
        a historical evaluation can only ever see prices that existed when the
        wager would have been placed.
        """
        return self._get(
            f"/historical/sports/{SPORT}/events/{event_id}/odds",
            {
                "regions": self.regions,
                "markets": ",".join(markets or PRIMARY_MARKETS),
                "oddsFormat": "american",
                "date": snapshot_iso,
            },
        )


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_event_props(payload: dict) -> list[PropQuote]:
    """Flatten one event-odds response into ``PropQuote`` rows.

    Player-market outcomes carry the player in ``description`` and the side in
    ``name``. Anytime-TD outcomes have no point value, which is why ``line`` is
    optional rather than defaulted to zero: a null line and a line of 0.0 mean
    very different things, and defaulting would silently corrupt the alt-line
    curve.
    """
    quotes: list[PropQuote] = []
    event_id = payload.get("id", "")
    commence = _parse_ts(payload.get("commence_time"))
    home = payload.get("home_team", "")
    away = payload.get("away_team", "")

    for book in payload.get("bookmakers", []):
        book_key = book.get("key", "unknown")
        for market in book.get("markets", []):
            key = market.get("key", "")
            stat = MARKET_TO_STAT.get(key)
            if stat is None:
                continue
            updated = _parse_ts(market.get("last_update"))
            for outcome in market.get("outcomes", []):
                player = (outcome.get("description") or "").strip()
                if not player:
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                quotes.append(
                    PropQuote(
                        event_id=event_id,
                        commence_time=commence,
                        home_team=home,
                        away_team=away,
                        bookmaker=book_key,
                        market_key=key,
                        stat=stat,
                        player_name=player,
                        side=str(outcome.get("name", "")).strip().lower(),
                        line=(float(outcome["point"])
                              if outcome.get("point") is not None else None),
                        american=float(price),
                        last_update=updated,
                        is_alternate=key.endswith("_alternate"),
                    )
                )
    return quotes


def poll_interval_seconds(minutes_to_kickoff: float) -> int:
    """Refresh cadence that spends credits where the line actually moves.

    Prices are close to static six days out and move constantly in the final
    hour, so a flat interval either burns quota or misses the movement that
    closing-line-value analysis depends on.
    """
    if minutes_to_kickoff <= 60:
        return 60
    if minutes_to_kickoff <= 360:
        return 300
    if minutes_to_kickoff <= 1440:
        return 900
    if minutes_to_kickoff <= 4320:
        return 3600
    return 21600


def group_by_player_market(
    quotes: Iterable[PropQuote],
) -> dict[tuple[str, str], list[PropQuote]]:
    """Bucket quotes so the edge engine can de-vig each book against itself."""
    grouped: dict[tuple[str, str], list[PropQuote]] = {}
    for q in quotes:
        grouped.setdefault((q.player_name, q.stat), []).append(q)
    return grouped
