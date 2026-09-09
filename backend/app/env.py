"""Minimal .env loader.

The scripts read ``os.environ`` directly, so without this a .env file sitting
in the project root is silently ignored -- the key is "set" as far as the user
is concerned and missing as far as the process is concerned.

Deliberately dependency-free and deliberately non-overriding: a variable
already exported in the shell wins over the file, so a CI secret or a
per-session override is never clobbered by a stale checked-out .env.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE pairs from a .env file. Returns the names it set.

    Searches upward from the current directory if no path is given, so the
    scripts work from either the repo root or backend/.
    """
    if path is None:
        for parent in [Path.cwd(), *Path.cwd().parents][:4]:
            candidate = parent / ".env"
            if candidate.exists():
                path = candidate
                break
        else:
            return []

    path = Path(path)
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
