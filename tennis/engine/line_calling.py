"""
Line Calling Module — Automated line call system with challenge support.
Mirrors SwingVision: detect bounce, determine in/out, store for replay.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from tennis.engine.event_processor import CourtGeometry, Point2D


class CallVerdict(str, Enum):
    IN = "in"
    OUT = "out"
    INCONCLUSIVE = "inconclusive"


class CallConfidence(str, Enum):
    HIGH = "high"          # >= 0.85 — auto-call
    MODERATE = "moderate"  # 0.60–0.85 — displayed but flagged
    LOW = "low"            # < 0.60 — inconclusive


class ChallengeStatus(str, Enum):
    NONE = "none"
    PENDING = "pending"
    OVERTURNED = "overturned"
    UPHELD = "upheld"


@dataclass
class LineCall:
    """A single line call with full replay context."""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp_ms: int = 0
    frame_number: int = 0

    # Ball position
    court_x: float = 0.0
    court_y: float = 0.0

    # Verdict
    verdict: CallVerdict = CallVerdict.IN
    confidence: float = 1.0
    confidence_level: CallConfidence = CallConfidence.HIGH
    distance_from_line_cm: float = 0.0
    closest_line: str = ""

    # Context
    is_serve: bool = False
    serve_side: str = "deuce"
    point_number: Optional[int] = None
    score_at_time: str = ""

    # Replay window
    replay_start_frame: int = 0
    replay_end_frame: int = 0
    replay_start_ms: int = 0
    replay_end_ms: int = 0

    # Challenge
    challenge_status: ChallengeStatus = ChallengeStatus.NONE
    challenged_by: Optional[str] = None
    original_verdict: Optional[CallVerdict] = None

    @property
    def is_challengeable(self) -> bool:
        return self.challenge_status == ChallengeStatus.NONE

    def to_display(self) -> dict:
        """Return display-ready dict for dashboard/overlay."""
        return {
            "verdict": self.verdict.value.upper(),
            "confidence": f"{self.confidence * 100:.0f}%",
            "distance": f"{self.distance_from_line_cm:.1f}cm",
            "line": self.closest_line,
            "challengeable": self.is_challengeable,
        }


@dataclass
class LineCallHistory:
    """Session-level line call log with summary statistics."""
    session_id: str = ""
    calls: list[LineCall] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def calls_in(self) -> int:
        return sum(1 for c in self.calls if c.verdict == CallVerdict.IN)

    @property
    def calls_out(self) -> int:
        return sum(1 for c in self.calls if c.verdict == CallVerdict.OUT)

    @property
    def challenges_made(self) -> int:
        return sum(1 for c in self.calls if c.challenge_status != ChallengeStatus.NONE)

    @property
    def challenges_successful(self) -> int:
        return sum(1 for c in self.calls if c.challenge_status == ChallengeStatus.OVERTURNED)

    @property
    def average_confidence(self) -> float:
        if not self.calls:
            return 0.0
        return sum(c.confidence for c in self.calls) / len(self.calls)

    def get_summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "in": self.calls_in,
            "out": self.calls_out,
            "average_confidence": round(self.average_confidence * 100, 1),
            "challenges_made": self.challenges_made,
            "challenges_successful": self.challenges_successful,
        }

    def get_calls_for_player(self, player_id: str) -> list[LineCall]:
        return [c for c in self.calls if c.challenged_by == player_id]


class LineCallingSystem:
    """
    Automated line calling with confidence thresholds.

    Confidence levels:
    - >= 0.85: auto-call, high confidence
    - 0.60–0.85: displayed but flagged for review
    - < 0.60: marked inconclusive
    """

    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_MODERATE = 0.60

    def __init__(self, session_id: str = "", is_doubles: bool = False, fps: float = 30.0):
        self.session_id = session_id
        self.fps = fps
        self.court = CourtGeometry(is_doubles=is_doubles)
        self.history = LineCallHistory(session_id=session_id)
        self._point_counter = 0

    def process_bounce(
        self,
        court_x: float,
        court_y: float,
        frame_number: int,
        is_serve: bool = False,
        serve_side: str = "deuce",
        score_at_time: str = "",
    ) -> LineCall:
        """Process a ball bounce and generate a line call."""
        timestamp_ms = int(frame_number / self.fps * 1000)

        # Use court geometry for in/out determination
        position = Point2D(x=court_x, y=court_y)
        is_in, distance_cm, closest_line = self.court.is_ball_in(
            position, is_serve=is_serve, serve_side=serve_side
        )

        # Determine confidence from distance
        confidence = self._calculate_confidence(distance_cm)
        confidence_level = self._classify_confidence(confidence)

        # Determine verdict
        if confidence < self.CONFIDENCE_MODERATE:
            verdict = CallVerdict.INCONCLUSIVE
        elif is_in:
            verdict = CallVerdict.IN
        else:
            verdict = CallVerdict.OUT

        # Replay window: ±2 seconds
        replay_frames = int(self.fps * 2)

        call = LineCall(
            session_id=self.session_id,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            court_x=court_x,
            court_y=court_y,
            verdict=verdict,
            confidence=confidence,
            confidence_level=confidence_level,
            distance_from_line_cm=distance_cm,
            closest_line=closest_line,
            is_serve=is_serve,
            serve_side=serve_side,
            point_number=self._point_counter,
            score_at_time=score_at_time,
            replay_start_frame=max(0, frame_number - replay_frames),
            replay_end_frame=frame_number + replay_frames,
            replay_start_ms=max(0, timestamp_ms - 2000),
            replay_end_ms=timestamp_ms + 2000,
        )

        self.history.calls.append(call)
        return call

    def challenge_call(self, call_id: str, challenger_id: str) -> Optional[LineCall]:
        """
        Challenge a line call. Re-evaluates with stricter threshold.
        Returns updated call or None if not found.
        """
        for call in self.history.calls:
            if call.call_id == call_id and call.is_challengeable:
                call.challenge_status = ChallengeStatus.PENDING
                call.challenged_by = challenger_id
                call.original_verdict = call.verdict

                # Re-evaluate with raw data (in production: use frame-level analysis)
                # For now, close calls (< 3cm) may be overturned
                if call.distance_from_line_cm < 3.0 and call.confidence < 0.90:
                    # Overturn: flip verdict
                    call.verdict = CallVerdict.IN if call.original_verdict == CallVerdict.OUT else CallVerdict.OUT
                    call.challenge_status = ChallengeStatus.OVERTURNED
                else:
                    call.challenge_status = ChallengeStatus.UPHELD

                return call
        return None

    def get_call(self, call_id: str) -> Optional[LineCall]:
        for call in self.history.calls:
            if call.call_id == call_id:
                return call
        return None

    def set_point_number(self, point_number: int):
        self._point_counter = point_number

    def _calculate_confidence(self, distance_cm: float) -> float:
        """Map distance from line to confidence score."""
        if distance_cm > 30:
            return 0.99
        elif distance_cm > 15:
            return 0.95
        elif distance_cm > 8:
            return 0.88
        elif distance_cm > 4:
            return 0.75
        elif distance_cm > 2:
            return 0.65
        else:
            return 0.55

    def _classify_confidence(self, confidence: float) -> CallConfidence:
        if confidence >= self.CONFIDENCE_HIGH:
            return CallConfidence.HIGH
        elif confidence >= self.CONFIDENCE_MODERATE:
            return CallConfidence.MODERATE
        else:
            return CallConfidence.LOW
