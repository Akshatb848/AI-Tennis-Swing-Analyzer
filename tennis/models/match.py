"""
Match data models — Full tennis match state representation.
Supports points, games, sets, tiebreaks, and complete match lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────

class CourtSurface(str, Enum):
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"
    ARTIFICIAL = "artificial"


class MatchFormat(str, Enum):
    BEST_OF_1 = "best_of_1"
    BEST_OF_3 = "best_of_3"
    BEST_OF_5 = "best_of_5"
    TIEBREAK_ONLY = "tiebreak_only"
    FAST4 = "fast4"


class MatchMode(str, Enum):
    SINGLES = "singles"
    DOUBLES = "doubles"


class MatchStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    SUSPENDED = "suspended"


class PointScore(str, Enum):
    ZERO = "0"
    FIFTEEN = "15"
    THIRTY = "30"
    FORTY = "40"
    ADVANTAGE = "AD"


class PointOutcomeType(str, Enum):
    WINNER = "winner"
    UNFORCED_ERROR = "unforced_error"
    FORCED_ERROR = "forced_error"
    ACE = "ace"
    DOUBLE_FAULT = "double_fault"
    LET = "let"
    NET = "net"
    SERVICE_WINNER = "service_winner"


class ShotType(str, Enum):
    SERVE = "serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    VOLLEY_FH = "volley_forehand"
    VOLLEY_BH = "volley_backhand"
    OVERHEAD = "overhead"
    DROP_SHOT = "drop_shot"
    LOB = "lob"
    SLICE_FH = "slice_forehand"
    SLICE_BH = "slice_backhand"
    RETURN_FH = "return_forehand"
    RETURN_BH = "return_backhand"


class CourtZone(str, Enum):
    """Court zone for shot placement analysis."""
    DEUCE_WIDE = "deuce_wide"
    DEUCE_BODY = "deuce_body"
    DEUCE_T = "deuce_t"
    AD_WIDE = "ad_wide"
    AD_BODY = "ad_body"
    AD_T = "ad_t"
    NET_LEFT = "net_left"
    NET_CENTER = "net_center"
    NET_RIGHT = "net_right"
    BASELINE_LEFT = "baseline_left"
    BASELINE_CENTER = "baseline_center"
    BASELINE_RIGHT = "baseline_right"
    NO_MANS_LAND_LEFT = "no_mans_land_left"
    NO_MANS_LAND_CENTER = "no_mans_land_center"
    NO_MANS_LAND_RIGHT = "no_mans_land_right"


# ── Core Models ──────────────────────────────────────────────────────────────

class MatchConfig(BaseModel):
    """Configuration for a tennis match."""
    match_format: MatchFormat = MatchFormat.BEST_OF_3
    match_mode: MatchMode = MatchMode.SINGLES
    surface: CourtSurface = CourtSurface.HARD
    tiebreak_at: int = Field(default=6, ge=1, le=12, description="Games at which tiebreak begins")
    final_set_tiebreak: bool = Field(default=True, description="Use tiebreak in final set (vs advantage)")
    final_set_tiebreak_to: int = Field(default=7, description="Points to win final set tiebreak")
    no_ad_scoring: bool = Field(default=False, description="No-ad (sudden death deuce) scoring")
    serve_clock_seconds: int = Field(default=25, ge=10, le=40)


class ShotDetail(BaseModel):
    """Detail of a single shot within a point."""
    shot_type: ShotType
    player_id: str
    speed_mph: float | None = None
    speed_kph: float | None = None
    spin_rpm: float | None = None
    placement_zone: CourtZone | None = None
    depth_meters: float | None = None
    net_clearance_cm: float | None = None
    timestamp_ms: int
    frame_number: int
    court_position_x: float | None = None
    court_position_y: float | None = None


class PointOutcome(BaseModel):
    """Outcome of a single point."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    point_number: int
    set_number: int
    game_number: int
    server_id: str
    winner_id: str
    outcome_type: PointOutcomeType
    last_shot: ShotDetail | None = None
    shot_sequence: list[ShotDetail] = Field(default_factory=list)
    rally_length: int = 0
    score_before: dict = Field(default_factory=dict)
    score_after: dict = Field(default_factory=dict)
    timestamp_start_ms: int = 0
    timestamp_end_ms: int = 0
    duration_seconds: float = 0.0
    is_break_point: bool = False
    is_set_point: bool = False
    is_match_point: bool = False


class GameState(BaseModel):
    """State of a single game."""
    game_number: int
    server_id: str
    points_player1: PointScore = PointScore.ZERO
    points_player2: PointScore = PointScore.ZERO
    is_deuce: bool = False
    advantage_player_id: str | None = None
    is_tiebreak: bool = False
    tiebreak_points_player1: int = 0
    tiebreak_points_player2: int = 0
    is_complete: bool = False
    winner_id: str | None = None


class SetState(BaseModel):
    """State of a set."""
    set_number: int
    games_player1: int = 0
    games_player2: int = 0
    is_tiebreak: bool = False
    current_game: GameState | None = None
    is_complete: bool = False
    winner_id: str | None = None
    games: list[GameState] = Field(default_factory=list)


class MatchState(BaseModel):
    """Complete match state."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: MatchConfig = Field(default_factory=MatchConfig)
    player1_id: str = "player1"
    player2_id: str = "player2"
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"
    status: MatchStatus = MatchStatus.NOT_STARTED
    sets_player1: int = 0
    sets_player2: int = 0
    current_set: SetState | None = None
    sets: list[SetState] = Field(default_factory=list)
    points_timeline: list[PointOutcome] = Field(default_factory=list)
    total_points_played: int = 0
    server_id: str | None = None
    first_server_id: str | None = None
    winner_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_minutes: float = 0.0

    @property
    def sets_to_win(self) -> int:
        fmt = self.config.match_format
        if fmt == MatchFormat.BEST_OF_5:
            return 3
        elif fmt == MatchFormat.BEST_OF_3:
            return 2
        elif fmt == MatchFormat.BEST_OF_1:
            return 1
        return 1

    @property
    def score_display(self) -> str:
        """Human-readable score string."""
        parts = []
        for s in self.sets:
            parts.append(f"{s.games_player1}-{s.games_player2}")
        if self.current_set and not self.current_set.is_complete:
            cs = self.current_set
            cg = cs.current_game
            set_score = f"{cs.games_player1}-{cs.games_player2}"
            if cg and not cg.is_complete:
                if cg.is_tiebreak:
                    game_score = f"({cg.tiebreak_points_player1}-{cg.tiebreak_points_player2})"
                else:
                    game_score = f"({cg.points_player1.value}-{cg.points_player2.value})"
                set_score += f" {game_score}"
            parts.append(set_score)
        return " | ".join(parts) if parts else "0-0"


class MatchSummary(BaseModel):
    """Lightweight match summary for listings."""
    id: str
    player1_name: str
    player2_name: str
    status: MatchStatus
    score_display: str
    surface: CourtSurface
    match_format: MatchFormat
    winner_name: str | None = None
    duration_minutes: float = 0.0
    total_points: int = 0
    started_at: datetime | None = None
