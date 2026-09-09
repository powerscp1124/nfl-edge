"""Kickoff weather.

Open-Meteo is used because it needs no API key and publishes hourly forecasts,
which matters: a 1pm kickoff and a 4:25pm kickoff in the same stadium can have
materially different wind, and a daily summary would blur them together.

Domes short-circuit the request entirely. There is no reason to spend a call,
and more importantly a dome game must never pick up a weather adjustment from a
forecast for the parking lot outside.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .teams import is_dome

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Stadium coordinates for outdoor venues. Domes are omitted because they are
# never queried.
STADIUM_COORDS = {
    "BAL": (39.278, -76.623), "BUF": (42.774, -78.787),
    "CAR": (35.226, -80.853), "CHI": (41.863, -87.617),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.700),
    "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
    "JAX": (30.324, -81.637), "KC": (39.049, -94.484),
    "MIA": (25.958, -80.239), "NE": (42.091, -71.264),
    "NYG": (40.814, -74.074), "NYJ": (40.814, -74.074),
    "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
    "SEA": (47.595, -122.332), "SF": (37.403, -121.970),
    "TB": (27.976, -82.503), "TEN": (36.166, -86.771),
    "WAS": (38.908, -76.864),
}


def fetch_weather(home_team: str, kickoff: datetime,
                  timeout: float = 15.0) -> dict:
    """Forecast for the kickoff hour, in the shape ``WeatherState`` expects.

    Returns dome conditions without a network call for indoor venues. On any
    failure the caller gets neutral conditions plus an explicit flag, because a
    weather outage must degrade the projection's confidence rather than
    silently apply a zero-wind assumption to a windy game.
    """
    if is_dome(home_team):
        return {"temperature_f": 70.0, "wind_mph": 0.0,
                "precipitation_prob": 0.0, "is_dome": True,
                "source": "dome"}

    coords = STADIUM_COORDS.get(home_team)
    if coords is None:
        return _unavailable(f"no coordinates for {home_team}")

    try:
        import httpx  # noqa: PLC0415

        lat, lon = coords
        resp = httpx.get(OPEN_METEO_URL, timeout=timeout, params={
            "latitude": lat, "longitude": lon,
            "hourly": ("temperature_2m,wind_speed_10m,wind_direction_10m,"
                       "precipitation_probability,relative_humidity_2m"),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_days": 8,
            "timezone": "UTC",
        })
        resp.raise_for_status()
        return _nearest_hour(resp.json(), kickoff)
    except Exception as exc:  # noqa: BLE001
        log.warning("Weather fetch failed for %s: %s", home_team, exc)
        return _unavailable(str(exc))


def _unavailable(reason: str) -> dict:
    return {"temperature_f": 60.0, "wind_mph": 0.0,
            "precipitation_prob": 0.0, "is_dome": False,
            "source": "unavailable", "reason": reason}


def _nearest_hour(payload: dict, kickoff: datetime) -> dict:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return _unavailable("empty forecast")

    target = kickoff.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    best_idx, best_gap = 0, None
    for i, t in enumerate(times):
        gap = abs((datetime.fromisoformat(t) - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best_idx, best_gap = i, gap

    def at(key, default=0.0):
        series = hourly.get(key) or []
        return float(series[best_idx]) if best_idx < len(series) else default

    return {
        "temperature_f": at("temperature_2m", 60.0),
        "wind_mph": at("wind_speed_10m"),
        "wind_bearing": at("wind_direction_10m"),
        "precipitation_prob": at("precipitation_probability") / 100.0,
        "humidity": at("relative_humidity_2m"),
        "is_dome": False,
        "source": "open-meteo",
        "forecast_hour": times[best_idx],
        "hours_from_kickoff": round((best_gap or 0) / 3600.0, 1),
    }
