"""Player identity resolution.

The Odds API gives you a display name and nothing else. nflverse gives you a
gsis_id. Joining them is the single most likely place this pipeline breaks, and
a bad join is worse than a missing one: mapping "Michael Thomas" to the wrong
Michael Thomas produces a confident projection against the wrong player's line,
which is exactly the kind of error that shows up as a huge apparent edge.

So resolution is tiered and refuses to guess:

1. Exact match on normalized name within the team's roster.
2. Exact match on normalized name league-wide, if unambiguous.
3. Fuzzy match, but only above a high threshold AND only when the runner-up is
   clearly worse. An ambiguous fuzzy match is not resolved — it goes to the
   alias queue for a human.

Anything unresolved blocks a projection for that prop rather than falling
through to a name-based lookup.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Display-name variants that normalization alone will not reconcile, because
# the two forms share no common normalized string.
KNOWN_ALIASES = {
    "gabe davis": "gabriel davis",
    "josh palmer": "joshua palmer",
    "cam ward": "cameron ward",
    "hollywood brown": "marquise brown",
    "chig okonkwo": "chigoziem okonkwo",
    "deebo samuel": "deebo samuel sr",
}


def normalize_name(name: str) -> str:
    """Reduce a display name to a comparable key.

    Strips accents, punctuation and generational suffixes, and collapses the
    initial-dot forms books use ("D.K. Metcalf" and "DK Metcalf" must land on
    the same key). Suffixes are dropped rather than kept because books are
    inconsistent about them, and a father/son pair active simultaneously at the
    same position is rare enough to handle through the alias queue.
    """
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[.\u2019']", "", text)      # D.K. -> dk, O'Neal -> oneal
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"[-\s]+", " ", text).strip()

    parts = [p for p in text.split(" ") if p]
    while len(parts) > 2 and parts[-1] in SUFFIXES:
        parts.pop()
    key = " ".join(parts)
    return KNOWN_ALIASES.get(key, key)


def initial_key(normalized: str) -> str | None:
    """Collapse a normalized name to first-initial + surname.

    Books abbreviate inconsistently: "J. Jefferson", "Justin Jefferson" and
    "Jefferson" all appear across providers for one player. Matching on
    initial-plus-surname is deterministic rather than fuzzy — it either matches
    or it does not — so it belongs above the fuzzy tier, not inside it. It is
    still ambiguity-checked, because two players with the same surname and
    initial on one roster is uncommon but not impossible.
    """
    parts = normalized.split()
    if len(parts) < 2:
        return None
    return f"{parts[0][0]} {parts[-1]}"


@dataclass(frozen=True)
class RosterEntry:
    player_id: str
    full_name: str
    position: str
    team: str

    @property
    def key(self) -> str:
        return normalize_name(self.full_name)

    @property
    def initial(self) -> str | None:
        return initial_key(self.key)


@dataclass
class MatchResult:
    raw_name: str
    player_id: str | None
    matched_name: str | None
    method: str          # exact_team / exact_league / fuzzy / ambiguous / none
    score: float
    resolved: bool
    candidates: list[str]

    @property
    def needs_review(self) -> bool:
        return not self.resolved


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class PlayerResolver:
    """Resolves provider display names to canonical player IDs."""

    def __init__(
        self,
        roster: Iterable[RosterEntry],
        fuzzy_threshold: float = 0.90,
        ambiguity_margin: float = 0.04,
    ) -> None:
        self.roster = list(roster)
        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin

        self._by_team: dict[str, dict[str, list[RosterEntry]]] = {}
        self._by_key: dict[str, list[RosterEntry]] = {}
        self._by_initial: dict[str, list[RosterEntry]] = {}
        for entry in self.roster:
            self._by_key.setdefault(entry.key, []).append(entry)
            self._by_team.setdefault(entry.team, {}).setdefault(
                entry.key, []).append(entry)
            if entry.initial:
                self._by_initial.setdefault(entry.initial, []).append(entry)

    def resolve(
        self,
        raw_name: str,
        team_hint: str | None = None,
        position_hint: str | None = None,
    ) -> MatchResult:
        key = normalize_name(raw_name)

        # 1. Exact, scoped to the team we already know is playing.
        if team_hint and team_hint in self._by_team:
            hits = self._by_team[team_hint].get(key, [])
            if len(hits) == 1:
                return self._hit(raw_name, hits[0], "exact_team", 1.0)
            if len(hits) > 1:
                hits = self._filter_position(hits, position_hint)
                if len(hits) == 1:
                    return self._hit(raw_name, hits[0], "exact_team", 1.0)

        # 2. Exact league-wide, but only if unambiguous.
        hits = self._by_key.get(key, [])
        if len(hits) == 1:
            return self._hit(raw_name, hits[0], "exact_league", 1.0)
        if len(hits) > 1:
            narrowed = self._filter_position(hits, position_hint)
            if len(narrowed) == 1:
                return self._hit(raw_name, narrowed[0], "exact_league", 1.0)
            return MatchResult(
                raw_name, None, None, "ambiguous", 1.0, False,
                [f"{h.player_id}:{h.full_name} ({h.team})" for h in hits],
            )

        # 3. First-initial plus surname. Deterministic, so it sits above the
        #    fuzzy tier -- but still refuses on a tie.
        ik = initial_key(key)
        if ik:
            hits = [e for e in self._by_initial.get(ik, [])
                    if team_hint is None or e.team == team_hint]
            hits = self._filter_position(hits, position_hint) or hits
            if len(hits) == 1:
                return self._hit(raw_name, hits[0], "initial", 0.99)
            if len(hits) > 1:
                return MatchResult(
                    raw_name, None, None, "ambiguous", 0.99, False,
                    [f"{h.player_id}:{h.full_name} ({h.team})" for h in hits],
                )

        # 4. Fuzzy, last resort, within the team when we have one.
        pool = [e for e in self.roster
                if team_hint is None or e.team == team_hint]
        pool = self._filter_position(pool, position_hint) or pool
        if not pool:
            return MatchResult(raw_name, None, None, "none", 0.0, False, [])

        scored = sorted(
            ((_similarity(key, e.key), e) for e in pool),
            key=lambda t: -t[0],
        )
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < self.fuzzy_threshold:
            return MatchResult(
                raw_name, None, None, "none", best_score, False,
                [f"{e.full_name} ({s:.2f})" for s, e in scored[:3]],
            )
        if best_score - runner_up < self.ambiguity_margin:
            # Two plausible matches. Guessing here is how a projection ends up
            # attached to the wrong player, so it goes to review instead.
            return MatchResult(
                raw_name, None, None, "ambiguous", best_score, False,
                [f"{e.full_name} ({s:.2f})" for s, e in scored[:3]],
            )
        return self._hit(raw_name, best, "fuzzy", best_score)

    @staticmethod
    def _filter_position(entries: Sequence[RosterEntry],
                         position_hint: str | None) -> list[RosterEntry]:
        if not position_hint:
            return list(entries)
        return [e for e in entries if e.position == position_hint]

    @staticmethod
    def _hit(raw: str, entry: RosterEntry, method: str,
             score: float) -> MatchResult:
        return MatchResult(raw, entry.player_id, entry.full_name,
                           method, score, True, [])

    def resolve_many(
        self,
        names: Iterable[tuple[str, str | None]],
    ) -> tuple[dict[str, str], list[MatchResult]]:
        """Resolve a batch. Returns ``(resolved_map, unresolved_results)``."""
        resolved: dict[str, str] = {}
        unresolved: list[MatchResult] = []
        for raw, team in names:
            result = self.resolve(raw, team_hint=team)
            if result.resolved and result.player_id:
                resolved[raw] = result.player_id
            else:
                unresolved.append(result)
        return resolved, unresolved


# Books price things that are not players: a team defence, and the "no
# touchdown scorer" line. Fuzzy-matching those against a roster is not merely
# useless, it is actively misleading -- it offers a running back as a
# candidate for a defence, and it counts against the match rate that is meant
# to measure whether player names are being matched correctly.
NON_PLAYER_MARKET_TOKENS = ("no scorer", "no touchdown", "any other",
                            "field goal", "d/st", "defense", "defence",
                            "special teams")


def is_non_player_market(name: str) -> bool:
    """Whether a market's 'player' is not a player at all."""
    from .teams import canonical_team
    if not name or not name.strip():
        return True
    lowered = name.strip().lower()
    if any(token in lowered for token in NON_PLAYER_MARKET_TOKENS):
        return True
    return canonical_team(name, strict=False) is not None


def match_rate(resolved: dict, unresolved: Sequence) -> float:
    total = len(resolved) + len(unresolved)
    return len(resolved) / total if total else 0.0
