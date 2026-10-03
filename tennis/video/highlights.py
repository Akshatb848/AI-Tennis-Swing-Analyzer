"""
Highlight Generator — Auto-detect and compile match highlights.
Uses excitement scoring, shot speed, and event significance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class HighlightType(str, Enum):
    ACE = "ace"
    WINNER = "winner"
    RALLY = "long_rally"
    BREAK_POINT = "break_point"
    SET_POINT = "set_point"
    MATCH_POINT = "match_point"
    CHALLENGE = "challenge"
    HOT_SHOT = "hot_shot"
    MOMENTUM_SHIFT = "momentum_shift"


@dataclass
class Highlight:
    """A single highlight clip."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    highlight_type: HighlightType = HighlightType.WINNER
    start_time_ms: int = 0
    end_time_ms: int = 0
    excitement_score: float = 0.0
    description: str = ""
    point_number: int | None = None
    score_at_time: str = ""
    player_featured: str = ""
    shot_speed_mph: float | None = None
    rally_length: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time_ms - self.start_time_ms) / 1000


@dataclass
class HighlightReel:
    """A compiled highlight reel from multiple highlights."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    title: str = "Match Highlights"
    highlights: list[Highlight] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    max_highlights: int = 15

    def add_highlight(self, h: Highlight):
        self.highlights.append(h)
        self.highlights.sort(key=lambda x: x.excitement_score, reverse=True)
        if len(self.highlights) > self.max_highlights:
            self.highlights = self.highlights[:self.max_highlights]
        self.total_duration_seconds = sum(h.duration_seconds for h in self.highlights)


class HighlightGenerator:
    """
    Generate match highlights from scoring and event data.

    Scoring criteria:
    - Base excitement from point outcome type
    - Bonus for pressure situations (break/set/match point)
    - Bonus for long rallies (>8 shots)
    - Bonus for high shot speed (>100mph)
    - Bonus for challenges
    """

    # Base excitement scores by outcome type
    BASE_SCORES = {
        "ace": 0.7,
        "winner": 0.6,
        "double_fault": 0.3,
        "unforced_error": 0.2,
        "forced_error": 0.4,
        "net_winner": 0.75,
        "passing_shot": 0.8,
        "drop_shot_winner": 0.8,
        "lob_winner": 0.85,
    }

    # Pressure multipliers
    PRESSURE_BONUS = {
        "match_point": 0.5,
        "set_point": 0.35,
        "break_point": 0.25,
        "tiebreak": 0.2,
        "deuce": 0.1,
    }

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.highlights: list[Highlight] = []

    def score_point(
        self,
        point_number: int,
        outcome_type: str,
        winner_id: str,
        start_time_ms: int,
        end_time_ms: int,
        rally_length: int = 1,
        shot_speed_mph: float | None = None,
        score_at_time: str = "",
        pressure_context: list[str] = None,
        is_challenge: bool = False,
    ) -> Highlight:
        """Score a point and determine if it's highlight-worthy."""
        # Base excitement
        excitement = self.BASE_SCORES.get(outcome_type, 0.3)

        # Pressure bonus
        for ctx in (pressure_context or []):
            excitement += self.PRESSURE_BONUS.get(ctx, 0.0)

        # Long rally bonus
        if rally_length >= 8:
            excitement += min(0.3, (rally_length - 8) * 0.03)

        # Speed bonus
        if shot_speed_mph and shot_speed_mph > 100:
            excitement += min(0.2, (shot_speed_mph - 100) * 0.005)

        # Challenge bonus
        if is_challenge:
            excitement += 0.2

        excitement = min(1.0, excitement)

        # Determine highlight type
        hl_type = self._classify_highlight(outcome_type, pressure_context or [], rally_length, is_challenge)

        # Build description
        desc = self._generate_description(outcome_type, winner_id, rally_length, shot_speed_mph, pressure_context or [])

        highlight = Highlight(
            session_id=self.session_id,
            highlight_type=hl_type,
            start_time_ms=max(0, start_time_ms - 2000),  # 2s before
            end_time_ms=end_time_ms + 1500,               # 1.5s after
            excitement_score=excitement,
            description=desc,
            point_number=point_number,
            score_at_time=score_at_time,
            player_featured=winner_id,
            shot_speed_mph=shot_speed_mph,
            rally_length=rally_length,
        )

        self.highlights.append(highlight)
        return highlight

    def generate_reel(self, max_highlights: int = 15, min_excitement: float = 0.5) -> HighlightReel:
        """Generate a highlight reel from collected highlights."""
        reel = HighlightReel(session_id=self.session_id, max_highlights=max_highlights)
        qualified = [h for h in self.highlights if h.excitement_score >= min_excitement]
        qualified.sort(key=lambda h: h.excitement_score, reverse=True)
        for h in qualified[:max_highlights]:
            reel.add_highlight(h)
        return reel

    def generate_condensed_match(self, all_points: list[dict]) -> list[Highlight]:
        """Generate a condensed match (every point, trimmed to action)."""
        condensed = []
        for pt in all_points:
            h = Highlight(
                session_id=self.session_id,
                start_time_ms=pt.get("start_ms", 0),
                end_time_ms=pt.get("end_ms", 0),
                point_number=pt.get("number", 0),
                score_at_time=pt.get("score", ""),
                excitement_score=0.5,
            )
            condensed.append(h)
        return condensed

    def _classify_highlight(self, outcome: str, pressure: list[str], rally_len: int, challenge: bool) -> HighlightType:
        if "match_point" in pressure:
            return HighlightType.MATCH_POINT
        if "set_point" in pressure:
            return HighlightType.SET_POINT
        if "break_point" in pressure:
            return HighlightType.BREAK_POINT
        if challenge:
            return HighlightType.CHALLENGE
        if rally_len >= 12:
            return HighlightType.RALLY
        if outcome == "ace":
            return HighlightType.ACE
        if outcome in ("passing_shot", "drop_shot_winner", "lob_winner"):
            return HighlightType.HOT_SHOT
        return HighlightType.WINNER

    def _generate_description(self, outcome: str, winner: str, rally_len: int, speed: float | None, pressure: list[str]) -> str:
        parts = []
        if pressure:
            parts.append(f"On {'/'.join(pressure)}:")
        parts.append(f"{winner}")
        if outcome == "ace":
            parts.append("serves an ace" + (f" at {speed:.0f}mph" if speed else ""))
        elif rally_len >= 8:
            parts.append(f"wins a {rally_len}-shot rally with a {outcome.replace('_', ' ')}")
        else:
            parts.append(f"hits a {outcome.replace('_', ' ')}" + (f" at {speed:.0f}mph" if speed else ""))
        return " ".join(parts)
