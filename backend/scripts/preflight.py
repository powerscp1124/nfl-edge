#!/usr/bin/env python3
"""Preflight: verify every data connection before spending credits.

Runs the checks in increasing order of cost, and stops at the first failure
that would make later checks meaningless. The order matters:

    1. Environment variables      free
    2. Python dependencies        free
    3. Odds API key               free -- /events costs zero credits
    4. nflverse loaders           free -- public data, no key
    5. Schema validation          free -- checks the real columns against
                                  what the transformation layer expects
    6. Weather                    free
    7. Odds API player props      COSTS CREDITS, opt in with --spend

Step 5 is the one that earns this script. Every transformation in
``context_loader`` was tested against synthetic frames shaped like nflverse
output; this is the first thing that checks the shape was right.

    python scripts/preflight.py            # everything free
    python scripts/preflight.py --spend    # also pull props for one game
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv  # noqa: E402

# Load .env before anything reads os.environ. Shell variables always win.
_loaded = load_dotenv()

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_ICON = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", SKIP: " skip "}

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> str:
    results.append((status, name, detail))
    print(f"[{_ICON[status]}] {name}" + (f"\n            {detail}" if detail else ""))
    return status


# --------------------------------------------------------------------------- #
# Values that are obviously an unedited placeholder rather than a credential.
# These produce a 401 from the API, which reads as "bad key" and sends people
# looking in the wrong place entirely.
def placeholder_reason(value: str) -> str | None:
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return ("wrapped in angle brackets -- the placeholder markers from the "
                "docs were not removed. Use the bare value.")
    if stripped != value:
        return "has leading or trailing whitespace"
    for token in ("your_", "_here", "changeme", "xxxx", "placeholder", "..."):
        if token in stripped.lower():
            return f"looks like a template value (contains {token!r})"
    if len(stripped) < 16:
        return f"only {len(stripped)} characters -- too short for an API key"
    if not all(c.isalnum() or c in "-_" for c in stripped):
        return "contains characters an API key would not normally have"
    return None


def check_environment() -> bool:
    required = ["ODDS_API_KEY"]
    optional = ["DATABASE_URL", "WEATHER_API_KEY"]
    ok = True
    for var in required:
        value = os.environ.get(var)
        if not value:
            record(FAIL, f"env {var}",
                   "Not set. Copy .env.example to .env and fill it in, then "
                   "export it or use a dotenv loader.")
            ok = False
        else:
            # Never print the key. Length and shape are enough to spot a
            # truncated paste or an unedited placeholder.
            reason = placeholder_reason(value)
            if reason:
                record(FAIL, f"env {var}",
                       f"present ({len(value)} chars) but {reason}")
                ok = False
            else:
                record(PASS, f"env {var}", f"present, {len(value)} chars")
    for var in optional:
        record(PASS if os.environ.get(var) else SKIP, f"env {var}",
               "" if os.environ.get(var) else "not set (not needed yet)")
    return ok


def check_dependencies() -> bool:
    ok = True
    for module, why in (("numpy", "core math"), ("scipy", "distributions"),
                        ("pandas", "transformations"), ("httpx", "HTTP")):
        try:
            __import__(module)
            record(PASS, f"import {module}")
        except ImportError:
            record(FAIL, f"import {module}", f"required for {why}")
            ok = False
    try:
        import nflreadpy  # noqa: F401
        record(PASS, "import nflreadpy")
    except ImportError:
        record(FAIL, "import nflreadpy",
               "pip install nflreadpy  (nfl_data_py is deprecated/archived)")
        ok = False
    try:
        import nfl_data_py  # noqa: F401
        record(WARN, "nfl_data_py present",
               "Deprecated and archived. This project no longer uses it; "
               "uninstall to avoid confusion.")
    except ImportError:
        pass
    return ok


def check_odds_key() -> tuple[bool, list]:
    """Costs zero credits: the /events endpoint is free."""
    try:
        from app.ingest.odds_api import OddsAPIClient
        client = OddsAPIClient()
        events = client.list_events()
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Odds API key", f"{type(exc).__name__}: {exc}")
        return False, []

    remaining = client.quota.remaining
    record(PASS, "Odds API key",
           f"{len(events)} upcoming NFL events, "
           f"{remaining if remaining is not None else 'unknown'} credits "
           f"remaining (this call cost 0)")
    if remaining is not None and remaining < 500:
        record(WARN, "Odds API quota",
               f"Only {remaining} credits left. A full slate of props at "
               "5 markets x 2 regions costs about 10 per game.")
    if events:
        e = events[0]
        record(PASS, "Odds API event shape",
               f"{e.get('away_team')} @ {e.get('home_team')} "
               f"({e.get('commence_time')}), id={e.get('id')}")
    return True, events


def check_nflverse(season: int) -> dict:
    """Free: nflverse is public data with no key."""
    frames = {}
    from app.ingest import nflverse

    for label, fn in (
        ("weekly", lambda: nflverse.load_weekly_stats([season])),
        ("play-by-play", lambda: nflverse.load_play_by_play([season])),
        ("snap counts", lambda: nflverse.load_snap_counts([season])),
        ("depth charts", lambda: nflverse.load_depth_charts([season])),
        ("injuries", lambda: nflverse.load_injuries([season])),
    ):
        try:
            df = fn()
            frames[label] = df
            record(PASS if len(df) else WARN, f"nflverse {label}",
                   f"{len(df):,} rows, {len(df.columns)} columns"
                   + ("" if len(df) else "  -- empty; is the season underway?"))
        except Exception as exc:  # noqa: BLE001
            record(FAIL, f"nflverse {label}", f"{type(exc).__name__}: {exc}")
    return frames


def check_schema(frames: dict) -> bool:
    """The check this script exists for.

    Every transformation was tested against synthetic frames. This is the first
    time the real column names are compared against what those transformations
    assume.
    """
    from app.ingest.context_loader import (DEPTH_RANK_COLUMNS,
                                           WEEKLY_COLUMNS)

    ok = True
    weekly = frames.get("weekly")
    if weekly is not None:
        required = {"player_id", "position", "week"}
        missing = required - set(weekly.columns)
        if missing:
            record(FAIL, "schema weekly required",
                   f"missing {sorted(missing)} -- normalize_weekly will raise")
            ok = False
        else:
            record(PASS, "schema weekly required")

        absent = [src for src, dest in WEEKLY_COLUMNS.items()
                  if src not in weekly.columns
                  and dest not in weekly.columns]
        if absent:
            record(WARN, "schema weekly optional",
                   f"not present, will default to zero: {absent}\n"
                   "            Check whether these were renamed rather than "
                   "removed -- a renamed column silently becomes a zero.")
        else:
            record(PASS, "schema weekly optional")

    pbp = frames.get("play-by-play")
    if pbp is not None:
        needed = {"play_type", "yardline_100", "posteam", "week",
                  "rusher_player_id", "receiver_player_id"}
        missing = needed - set(pbp.columns)
        if missing:
            record(FAIL, "schema pbp red-zone", f"missing {sorted(missing)}")
            ok = False
        else:
            record(PASS, "schema pbp red-zone")

        pace = {"game_seconds_remaining", "qtr", "score_differential", "game_id"}
        if pace - set(pbp.columns):
            record(WARN, "schema pbp pace",
                   f"missing {sorted(pace - set(pbp.columns))} -- pace falls "
                   "back to the crude play-volume estimate")
        else:
            record(PASS, "schema pbp pace")

    depth = frames.get("depth charts")
    if depth is not None and len(depth):
        rank_col = next((c for c in DEPTH_RANK_COLUMNS
                         if c in depth.columns), None)
        if rank_col is None:
            record(WARN, "schema depth rank",
                   f"no recognised rank column in {sorted(depth.columns)[:12]}"
                   "\n            build_depth_ranks returns empty; injury "
                   "redistribution loses the direct-backup bonus.")
        else:
            record(PASS, "schema depth rank", f"using '{rank_col}'")

    inj = frames.get("injuries")
    if inj is not None and len(inj):
        status_col = next((c for c in ("report_status", "game_status", "status")
                           if c in inj.columns), None)
        if status_col is None:
            record(FAIL, "schema injury status",
                   f"none of report_status/game_status/status in "
                   f"{sorted(inj.columns)[:12]}")
            ok = False
        else:
            record(PASS, "schema injury status", f"using '{status_col}'")
    return ok


def check_teams(frames: dict) -> bool:
    """Every team code in the real data must map to a canonical code."""
    from app.ingest.teams import canonical_team

    ok = True
    for label, candidates in (("weekly", ("team", "recent_team")),
                              ("play-by-play", ("posteam",))):
        df = frames.get(label)
        if df is None:
            continue
        column = next((c for c in candidates if c in df.columns), None)
        if column is None:
            continue
        codes = {c for c in df[column].dropna().unique() if c}
        unmapped = [c for c in codes if canonical_team(c, strict=False) is None]
        if unmapped:
            record(FAIL, f"team codes in {label}",
                   f"unmapped: {sorted(unmapped)}\n"
                   "            Add these to CODE_ALIASES in "
                   "app/ingest/teams.py before running anything else.")
            ok = False
        else:
            record(PASS, f"team codes in {label}", f"{len(codes)} codes mapped")
    return ok


def check_weather(events: list) -> None:
    from datetime import datetime, timezone

    from app.ingest.teams import canonical_team
    from app.ingest.weather import fetch_weather

    if not events:
        record(SKIP, "weather", "no events to test against")
        return
    try:
        home = canonical_team(events[0].get("home_team", ""), strict=False)
        if home is None:
            record(WARN, "weather", "could not resolve home team")
            return
        wx = fetch_weather(home, datetime.now(timezone.utc))
        if wx.get("source") == "unavailable":
            record(WARN, "weather",
                   f"unavailable ({wx.get('reason')}) -- projections fall back "
                   "to neutral conditions")
        else:
            record(PASS, "weather",
                   f"{home}: source={wx['source']}, "
                   f"wind={wx.get('wind_mph', 0):.0f}mph, "
                   f"dome={wx.get('is_dome')}")
    except Exception as exc:  # noqa: BLE001
        record(WARN, "weather", f"{type(exc).__name__}: {exc}")


def check_props(events: list) -> None:
    """The only check that costs credits."""
    from app.ingest.odds_api import OddsAPIClient, normalize_event_props

    client = OddsAPIClient()
    event_id = events[0]["id"]
    payload = client.event_props(event_id)
    quotes = normalize_event_props(payload)
    record(PASS if quotes else FAIL, "Odds API props",
           f"{len(quotes)} rows from "
           f"{len({q.bookmaker for q in quotes})} books, cost "
           f"{client.quota.last_cost} credits, "
           f"{client.quota.remaining} remaining")
    if quotes:
        by_stat: dict[str, int] = {}
        for q in quotes:
            by_stat[q.stat] = by_stat.get(q.stat, 0) + 1
        record(PASS, "props by market", str(by_stat))
        sample = quotes[0]
        record(PASS, "props player name format",
               f"e.g. {sample.player_name!r} -- check this matches the format "
               "in nflverse rosters, not an abbreviation")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--spend", action="store_true",
                        help="also pull player props for one game (costs credits)")
    args = parser.parse_args()

    print("\n=== 1. Environment ===")
    if _loaded:
        record(PASS, ".env file", f"loaded {len(_loaded)} variable(s)")
    else:
        record(SKIP, ".env file",
               "none found; relying on exported shell variables")
    env_ok = check_environment()

    print("\n=== 2. Dependencies ===")
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\nStopping: install the missing packages first.")
        return 1

    print("\n=== 3. Odds API (0 credits) ===")
    key_ok, events = (check_odds_key() if env_ok else (False, []))

    season = args.season
    if season is None:
        try:
            from app.ingest.nflverse import current_season_week
            season, week = current_season_week()
            record(PASS, "current season/week", f"{season}, week {week}")
        except Exception as exc:  # noqa: BLE001
            record(WARN, "current season/week",
                   f"{exc}; pass --season explicitly")
            season = 2026

    print(f"\n=== 4. nflverse, season {season} (0 credits) ===")
    frames = check_nflverse(season)

    print("\n=== 5. Schema ===")
    check_schema(frames)
    check_teams(frames)

    print("\n=== 6. Weather (0 credits) ===")
    check_weather(events)

    print("\n=== 7. Player props ===")
    if not args.spend:
        record(SKIP, "Odds API props", "pass --spend to test (costs credits)")
    elif not key_ok or not events:
        record(SKIP, "Odds API props", "no working key or no events")
    else:
        try:
            check_props(events)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "Odds API props", f"{type(exc).__name__}: {exc}")

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n=== Summary: {len(results) - len(fails) - len(warns)} ok, "
          f"{len(warns)} warnings, {len(fails)} failures ===")
    for _, name, detail in fails:
        print(f"  FAIL  {name}: {detail.splitlines()[0] if detail else ''}")
    if not fails:
        print("\nReady. Next:\n"
              "  python scripts/slice.py --event-id <id> --season "
              f"{season}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
