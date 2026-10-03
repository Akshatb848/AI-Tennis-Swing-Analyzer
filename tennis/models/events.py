"""
Event data models — Ball events, player events, rallies, and line calls.
These represent raw CV detections converted into tennis-meaningful events.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    BALL_BOUNCE = "ball_bounce"
    BALL_HIT = "ball_hit"
    BALL_OUT = "ball_out"
    BALL_NET = "ball_net"
    BALL_LET = "ball_let"
    RALLY_START = "rally_start"
    RALLY_END = "rally_end"
    POINT_START = "point_start"
    POINT_END = "point_end"
    SERVE = "serve"
    FAULT = "fault"
    PLAYER_POSITION = "player_position"
    POSE_SNAPSHOT = "pose_snapshot"
    CHALLENGE_REQUESTED = "challenge_requested"
    CHALLENGE_RESOLVED = "challenge_resolved"


class LineCallVerdict(str, Enum):
    IN = "in"
    OUT = "out"
    LET = "let"
    NET = "net"
    UNKNOWN = "unknown"


class ChallengeStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    OVERTURNED = "overturned"
    EXPIRED = "expired"


class BounceConfidence(str, Enum):
    HIGH = "high"         # >90% confidence
    MEDIUM = "medium"     # 70-90%
    LOW = "low"           # 50-70%
    UNCERTAIN = "uncertain"  # <50%


# ── Position & Geometry ──────────────────────────────────────────────────────

class Point2D(BaseModel):
    """2D point in court coordinates (meters from court center)."""
    x: float
    y: float


class Point3D(BaseModel):
    """3D point for ball position (meters)."""
    x: float
    y: float
    z: float = 0.0


class BoundingBox(BaseModel):
    """Bounding box in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0


class PoseKeypoint(BaseModel):
    """Single pose keypoint."""
    name: str
    x: float
    y: float
    confidence: float = 0.0


class PlayerPose(BaseModel):
    """Full body pose with 17 keypoints (COCO format)."""
    keypoints: list[PoseKeypoint] = Field(default_factory=list)
    overall_confidence: float = 0.0

    # Named accessors for key joints
    @property
    def left_shoulder(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "left_shoulder"), None)

    @property
    def right_shoulder(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "right_shoulder"), None)

    @property
    def left_elbow(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "left_elbow"), None)

    @property
    def right_elbow(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "right_elbow"), None)

    @property
    def left_wrist(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "left_wrist"), None)

    @property
    def right_wrist(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "right_wrist"), None)

    @property
    def left_hip(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "left_hip"), None)

    @property
    def right_hip(self) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == "right_hip"), None)


# ── Event Models ─────────────────────────────────────────────────────────────

class BallEvent(BaseModel):
    """A ball detection/tracking event from CV pipeline."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp_ms: int
    frame_number: int
    session_id: str

    # ── Ball state ───────────────────────────────────────
    position_image: BoundingBox | None = None
    position_court: Point2D | None = None
    position_3d: Point3D | None = None
    velocity_mph: float | None = None
    velocity_kph: float | None = None
    spin_proxy_rpm: float | None = None
    trajectory_angle_deg: float | None = None

    # ── Detection quality ────────────────────────────────
    detection_confidence: float = 0.0
    is_occluded: bool = False
    is_interpolated: bool = False
    tracker_id: int | None = None

    # ── Line call (for bounces) ──────────────────────────
    line_call: LineCallVerdict | None = None
    line_call_confidence: float = 0.0
    distance_from_line_cm: float | None = None
    uncertainty_radius_cm: float | None = None


class PlayerEvent(BaseModel):
    """A player detection/pose event from CV pipeline."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.PLAYER_POSITION
    timestamp_ms: int
    frame_number: int
    session_id: str
    player_id: str

    # ── Player state ─────────────────────────────────────
    position_image: BoundingBox | None = None
    position_court: Point2D | None = None
    pose: PlayerPose | None = None
    court_zone: str | None = None
    velocity_mps: float | None = None
    facing_direction_deg: float | None = None

    # ── Detection quality ────────────────────────────────
    detection_confidence: float = 0.0
    tracker_id: int | None = None
    is_serving: bool = False


class RallyEvent(BaseModel):
    """A complete rally from serve to point conclusion."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    match_id: str | None = None
    point_number: int

    # ── Timeline ─────────────────────────────────────────
    start_frame: int
    end_frame: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_seconds: float = 0.0

    # ── Rally content ────────────────────────────────────
    server_id: str
    winner_id: str | None = None
    rally_length: int = 0
    shots: list[dict] = Field(default_factory=list, description="Ordered shot events")
    ball_events: list[str] = Field(
        default_factory=list, description="Ball event IDs in this rally"
    )

    # ── Outcome ──────────────────────────────────────────
    outcome_type: str | None = None
    last_shot_type: str | None = None
    last_shot_player_id: str | None = None

    # ── Analytics ────────────────────────────────────────
    max_shot_speed_mph: float = 0.0
    avg_shot_speed_mph: float = 0.0
    excitement_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="ML-scored excitement level for highlight ranking"
    )


class LineCallEvent(BaseModel):
    """A line call decision, potentially subject to challenge."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    match_id: str | None = None
    rally_id: str
    ball_event_id: str

    # ── Decision ─────────────────────────────────────────
    verdict: LineCallVerdict
    confidence: float = 0.0
    bounce_position_court: Point2D
    closest_line: str = ""
    distance_from_line_cm: float = 0.0
    uncertainty_radius_cm: float = 0.0

    # ── Challenge ────────────────────────────────────────
    is_challenged: bool = False
    challenge_status: ChallengeStatus = ChallengeStatus.PENDING
    challenged_by_player_id: str | None = None
    challenge_timestamp_ms: int | None = None
    replay_frame_start: int | None = None
    replay_frame_end: int | None = None
    original_verdict: LineCallVerdict | None = None
    final_verdict: LineCallVerdict | None = None

    timestamp_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EventBatch(BaseModel):
    """Batch of events for bulk ingestion from device."""
    session_id: str
    device_id: str
    batch_number: int = 0
    ball_events: list[BallEvent] = Field(default_factory=list)
    player_events: list[PlayerEvent] = Field(default_factory=list)
    timestamp_start_ms: int = 0
    timestamp_end_ms: int = 0
    frame_count: int = 0
