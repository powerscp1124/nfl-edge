"""Team identity across providers.

Team codes are the most boring possible source of bugs and one of the most
common. The Odds API returns full display names ("Minnesota Vikings"), nflverse
uses codes, ESPN uses its own abbreviations, and the codes have changed over
time: Oakland became Las Vegas, San Diego became Los Angeles, St. Louis became
Los Angeles, and there are two different Los Angeles teams whose codes both
begin with LA.

Historical data still carries the old codes, so a backtest that joins on team
code without normalising will silently drop every pre-relocation game for three
franchises. Everything normalises through ``canonical_team`` before any join.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Canonical codes are the current nflverse codes.
TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Every alternate code seen in the wild, mapped to canonical.
CODE_ALIASES = {
    # Relocations. Historical rows keep the old code.
    "OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR",
    # Provider spelling differences.
    "JAC": "JAX", "WSH": "WAS", "GNB": "GB", "KAN": "KC",
    "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
    "LVR": "LV", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU", "SL": "LAR", "PHO": "ARI", "RAI": "LV",
    "RAM": "LAR",
}

# Former display names, so historical odds payloads still resolve.
NAME_ALIASES = {
    "oakland raiders": "LV",
    "san diego chargers": "LAC",
    "st louis rams": "LAR",
    "st. louis rams": "LAR",
    "washington redskins": "WAS",
    "washington football team": "WAS",
}

# Stadium environment. Retractable roofs are treated as domes by default,
# because they are closed far more often than not in weather that would matter;
# the weather loader overrides this when a game-day roof status is available.
DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LAR", "LAC",
              "LV", "MIN", "NO"}


class UnknownTeamError(ValueError):
    """Raised rather than guessing. A wrong team code corrupts a whole game."""


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)


_NAME_TO_CODE = {_normalize(name): code for code, name in TEAMS.items()}
_NAME_TO_CODE.update({_normalize(k): v for k, v in NAME_ALIASES.items()})
# Nickname only, e.g. "vikings" -> MIN. Unambiguous for all 32.
_NICKNAME_TO_CODE = {
    _normalize(name.split()[-1]): code for code, name in TEAMS.items()
}


def canonical_team(value: str, strict: bool = True) -> str | None:
    """Resolve any provider's team identifier to a canonical code.

    Accepts codes ("MIN", "JAC", "OAK"), full names ("Minnesota Vikings"),
    and nicknames ("Vikings"). Raises on anything unrecognised rather than
    returning a plausible-looking wrong answer.
    """
    if not value:
        if strict:
            raise UnknownTeamError("Empty team identifier")
        return None

    raw = str(value).strip()
    upper = raw.upper()
    if upper in TEAMS:
        return upper
    if upper in CODE_ALIASES:
        return CODE_ALIASES[upper]

    key = _normalize(raw)
    if key in _NAME_TO_CODE:
        return _NAME_TO_CODE[key]
    if key in _NICKNAME_TO_CODE:
        return _NICKNAME_TO_CODE[key]
    # "New York Giants" style with the city dropped by a provider.
    for name_key, code in _NAME_TO_CODE.items():
        if key and (key in name_key or name_key in key):
            return code

    if strict:
        raise UnknownTeamError(
            f"Unrecognised team identifier: {value!r}. Add it to CODE_ALIASES "
            "or NAME_ALIASES rather than letting it fall through."
        )
    return None


def is_dome(team: str) -> bool:
    return canonical_team(team) in DOME_TEAMS


@dataclass(frozen=True)
class TeamPair:
    home: str
    away: str


def teams_from_event(payload: dict) -> TeamPair:
    """Extract canonical home/away codes from an Odds API event payload."""
    return TeamPair(
        home=canonical_team(payload.get("home_team", "")),
        away=canonical_team(payload.get("away_team", "")),
    )
