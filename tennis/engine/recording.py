"""
Recording Engine — Continuous match recording with auto-segmentation.
Mirrors SwingVision: start recording, auto-detect everything, stop when done.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from tennis.engine.event_processor import EventProcessor
from tennis.engine.scoring import ScoringEngine
from tennis.models.match import MatchConfig, MatchMode


class MatchType(str, Enum):
    PRACTICE_RALLY = "practice_rally"
    SINGLES = "singles"
    DOUBLES = "doubles"


class Environment(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    AUTO = "auto"


class RecordingState(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    CALIBRATING = "calibrating"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class MatchSetupConfig:
    """
    Step-by-step match setup. All fields have defaults
    so a first-time user can proceed immediately.
    """
    match_type: MatchType = MatchType.SINGLES
    environment: Environment = Environment.OUTDOOR
    player_count: int = 2
    player_names: list[str] = field(default_factory=lambda: ["Player 1", "Player 2"])
    player_handedness: list[Handedness] = field(default_factory=lambda: [Handedness.AUTO, Handedness.AUTO])
    court_surface: str = "hard"

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        expected = 4 if self.match_type == MatchType.DOUBLES else 2
        if self.match_type != MatchType.PRACTICE_RALLY and len(self.player_names) != expected:
            errors.append(f"Expected {expected} players for {self.match_type.value}")
        if self.match_type == MatchType.DOUBLES:
            self.player_count = 4
        return errors


@dataclass
class SegmentedPoint:
    """A single point auto-segmented from continuous recording."""
    point_number: int
    start_frame: int
    end_frame: int
    start_time_ms: int
    end_time_ms: int
    rally_length: int = 0
    winner_id: Optional[str] = None
    outcome: str = ""
    line_calls: list[dict] = field(default_factory=list)
    score_after: str = ""


@dataclass
class SegmentedGame:
    """A game auto-segmented from points."""
    game_number: int
    set_number: int
    points: list[SegmentedPoint] = field(default_factory=list)
    winner_id: Optional[str] = None
    score_after: str = ""


@dataclass
class SegmentedSet:
    """A set auto-segmented from games."""
    set_number: int
    games: list[SegmentedGame] = field(default_factory=list)
    winner_id: Optional[str] = None
    final_score: str = ""


class RecordingSession:
    """
    Continuous recording session manager.

    Usage:
        session = RecordingSession()
        session.setup(config)
        session.start_recording()
        # ... frames flow in via process_frame()
        session.stop_recording()
        summary = session.get_summary()
    """

    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.state: RecordingState = RecordingState.IDLE
        self.setup_config: Optional[MatchSetupConfig] = None
        self.started_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        self.frame_count: int = 0
        self.fps: float = 30.0

        # Sub-engines
        self._event_processor: Optional[EventProcessor] = None
        self._scoring_engine: Optional[ScoringEngine] = None

        # Auto-segmentation state
        self._current_rally_start_frame: int = 0
        self._current_rally_start_ms: int = 0
        self._rally_active: bool = False
        self._rally_shot_count: int = 0

        # Results
        self.points: list[SegmentedPoint] = []
        self.games: list[SegmentedGame] = []
        self.sets: list[SegmentedSet] = []
        self.line_calls: list[dict] = []

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self, config: Optional[MatchSetupConfig] = None) -> dict:
        """
        Configure match. Returns setup summary.
        If no config provided, uses defaults (singles, outdoor, auto-detect).
        """
        if config is None:
            config = MatchSetupConfig()

        errors = config.validate()
        if errors:
            return {"status": "error", "errors": errors}

        self.setup_config = config
        self.state = RecordingState.SETUP

        is_doubles = config.match_type == MatchType.DOUBLES
        self._event_processor = EventProcessor(is_doubles=is_doubles)

        if config.match_type != MatchType.PRACTICE_RALLY:
            match_config = MatchConfig(
                match_mode=MatchMode.DOUBLES if is_doubles else MatchMode.SINGLES,
            )
            self._scoring_engine = ScoringEngine(match_config)
            player_ids = [f"p{i}" for i in range(config.player_count)]
            self._scoring_engine.start_match(
                player_ids[0], player_ids[1],
                config.player_names[0], config.player_names[1],
            )

        self.state = RecordingState.CALIBRATING
        return {
            "status": "ready",
            "session_id": self.id,
            "match_type": config.match_type.value,
            "players": config.player_names,
            "environment": config.environment.value,
        }

    # ── Recording Control ────────────────────────────────────────────────────

    def start_recording(self) -> dict:
        """Begin recording. Auto-processing starts immediately."""
        if self.state not in (RecordingState.CALIBRATING, RecordingState.SETUP):
            return {"status": "error", "message": "Setup required before recording"}

        self.state = RecordingState.RECORDING
        self.started_at = datetime.utcnow()
        self.frame_count = 0
        return {"status": "recording", "session_id": self.id}

    def stop_recording(self) -> dict:
        """Stop recording and finalize segmentation."""
        if self.state != RecordingState.RECORDING:
            return {"status": "error", "message": "Not currently recording"}

        self.state = RecordingState.PROCESSING
        self.stopped_at = datetime.utcnow()

        # Close any open rally
        if self._rally_active:
            self._close_rally(self.frame_count, self._frame_to_ms(self.frame_count))

        # Build game/set segmentation from points
        self._build_game_set_segments()

        self.state = RecordingState.COMPLETE
        return {
            "status": "complete",
            "session_id": self.id,
            "duration_seconds": self._get_duration_seconds(),
            "total_frames": self.frame_count,
            "points_detected": len(self.points),
            "games_detected": len(self.games),
            "sets_detected": len(self.sets),
            "line_calls_total": len(self.line_calls),
        }

    # ── Frame Processing ─────────────────────────────────────────────────────

    def process_frame(self, ball_event=None, player_events=None) -> Optional[dict]:
        """
        Process a single frame during recording.
        Returns any events generated (line calls, point outcomes).
        """
        if self.state != RecordingState.RECORDING:
            return None

        self.frame_count += 1
        timestamp_ms = self._frame_to_ms(self.frame_count)
        results = {}

        # Process ball event through event processor
        if ball_event and self._event_processor:
            event_results = self._event_processor.process_ball_event(
                ball_event, player_events
            )

            # Line call detected
            if event_results.get("line_call"):
                lc = event_results["line_call"]
                call_record = {
                    "call_id": str(uuid.uuid4()),
                    "frame": self.frame_count,
                    "timestamp_ms": timestamp_ms,
                    "verdict": "in" if lc.is_in else "out",
                    "confidence": lc.confidence,
                    "distance_cm": lc.distance_from_line_cm,
                    "closest_line": lc.closest_line,
                    "court_x": lc.bounce_position.x if lc.bounce_position else 0,
                    "court_y": lc.bounce_position.y if lc.bounce_position else 0,
                    "replay_start_frame": max(0, self.frame_count - int(self.fps * 2)),
                    "replay_end_frame": self.frame_count + int(self.fps * 2),
                }
                self.line_calls.append(call_record)
                results["line_call"] = call_record

            # Rally tracking
            if event_results.get("rally_event"):
                rally = event_results["rally_event"]
                if rally.is_complete:
                    self._close_rally(
                        self.frame_count, timestamp_ms,
                        rally_length=rally.rally_length,
                        winner_id=rally.winner_id,
                        outcome=rally.outcome_type.value if rally.outcome_type else "",
                    )
                    results["point_ended"] = True

            # Auto-detect rally start from ball hits
            if ball_event.event_type and "hit" in str(ball_event.event_type).lower():
                if not self._rally_active:
                    self._start_rally(self.frame_count, timestamp_ms)
                self._rally_shot_count += 1

        return results if results else None

    # ── Auto-Segmentation ────────────────────────────────────────────────────

    def _start_rally(self, frame: int, timestamp_ms: int):
        self._rally_active = True
        self._current_rally_start_frame = frame
        self._current_rally_start_ms = timestamp_ms
        self._rally_shot_count = 0

    def _close_rally(self, end_frame: int, end_ms: int, **kwargs):
        point = SegmentedPoint(
            point_number=len(self.points) + 1,
            start_frame=self._current_rally_start_frame,
            end_frame=end_frame,
            start_time_ms=self._current_rally_start_ms,
            end_time_ms=end_ms,
            rally_length=kwargs.get("rally_length", self._rally_shot_count),
            winner_id=kwargs.get("winner_id"),
            outcome=kwargs.get("outcome", ""),
            line_calls=[lc for lc in self.line_calls
                        if self._current_rally_start_frame <= lc["frame"] <= end_frame],
        )

        # Update scoring engine
        if self._scoring_engine and point.winner_id:
            self._scoring_engine.score_point(point.winner_id)
            state = self._scoring_engine.get_match_state()
            point.score_after = self._scoring_engine.get_score_display()

        self.points.append(point)
        self._rally_active = False
        self._rally_shot_count = 0

    def _build_game_set_segments(self):
        """Build game/set structure from scored points."""
        if not self._scoring_engine:
            return

        state = self._scoring_engine.get_match_state()
        # Build games from point groupings
        current_game_points: list[SegmentedPoint] = []
        game_num = 0
        set_num = 1

        for pt in self.points:
            current_game_points.append(pt)
            # Simple heuristic: game boundary when score resets
            if "0-0" in pt.score_after or len(current_game_points) >= 12:
                game_num += 1
                game = SegmentedGame(
                    game_number=game_num,
                    set_number=set_num,
                    points=list(current_game_points),
                    score_after=pt.score_after,
                )
                self.games.append(game)
                current_game_points = []

        # Close any remaining points as a game
        if current_game_points:
            game_num += 1
            self.games.append(SegmentedGame(
                game_number=game_num,
                set_number=set_num,
                points=current_game_points,
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _frame_to_ms(self, frame: int) -> int:
        return int(frame / self.fps * 1000)

    def _get_duration_seconds(self) -> float:
        if self.started_at and self.stopped_at:
            return (self.stopped_at - self.started_at).total_seconds()
        return self.frame_count / self.fps

    def get_summary(self) -> dict:
        """Get complete session summary for dashboard consumption."""
        return {
            "session_id": self.id,
            "state": self.state.value,
            "match_type": self.setup_config.match_type.value if self.setup_config else "unknown",
            "players": self.setup_config.player_names if self.setup_config else [],
            "duration_seconds": self._get_duration_seconds(),
            "total_frames": self.frame_count,
            "points": [
                {
                    "number": p.point_number,
                    "start_ms": p.start_time_ms,
                    "end_ms": p.end_time_ms,
                    "rally_length": p.rally_length,
                    "winner": p.winner_id,
                    "outcome": p.outcome,
                    "score_after": p.score_after,
                    "line_calls": len(p.line_calls),
                }
                for p in self.points
            ],
            "line_calls": self.line_calls,
            "games": len(self.games),
            "sets": len(self.sets),
        }
