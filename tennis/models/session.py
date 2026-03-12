"""
Session data models — Capture sessions, calibration, timeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from tennis.models.match import CourtSurface, MatchConfig


class SessionMode(str, Enum):
    MATCH = "match"
    PRACTICE = "practice"
    DRILL = "drill"
    WARMUP = "warmup"


class SessionStatus(str, Enum):
    SETUP = "setup"
    CALIBRATING = "calibrating"
    RECORDING = "recording"
    PAUSED = "paused"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CalibrationState(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    CALIBRATED = "calibrated"
    RECALIBRATING = "recalibrating"
    FAILED = "failed"


class DeviceInfo(BaseModel):
    """Device hardware information."""
    device_id: str
    device_model: str = ""           # e.g. "iPhone 15 Pro"
    os_version: str = ""             # e.g. "iOS 18.2"
    chip: str = ""                   # e.g. "A17 Pro"
    neural_engine_cores: int = 16
    available_memory_gb: float = 0.0
    battery_level: float = 1.0
    thermal_state: str = "nominal"   # nominal | fair | serious | critical


class CameraCalibration(BaseModel):
    """Camera intrinsic/extrinsic calibration parameters."""
    # Intrinsics
    focal_length_x: float = 0.0
    focal_length_y: float = 0.0
    principal_point_x: float = 0.0
    principal_point_y: float = 0.0
    distortion_coefficients: list[float] = Field(default_factory=lambda: [0.0] * 5)

    # Extrinsics (camera → court)
    rotation_matrix: list[list[float]] = Field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    translation_vector: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )

    # Homography (image → court plane)
    homography_matrix: list[list[float]] = Field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    reprojection_error: float = 0.0
    calibration_confidence: float = 0.0
    last_updated_frame: int = 0


class CaptureSession(BaseModel):
    """A recording session — the top-level container for all data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mode: SessionMode = SessionMode.MATCH
    status: SessionStatus = SessionStatus.SETUP

    # ── Match Setup ─────────────────────────────────────
    match_type: str = "singles"          # practice_rally, singles, doubles
    environment: str = "outdoor"          # indoor, outdoor
    player_handedness: list[str] = Field(
        default_factory=lambda: ["auto", "auto"],
        description="Per-player handedness: left, right, or auto"
    )

    # ── Players ──────────────────────────────────────────
    player_ids: list[str] = Field(default_factory=list)
    player_names: list[str] = Field(default_factory=list)

    # ── Court & Match ────────────────────────────────────
    court_surface: CourtSurface = CourtSurface.HARD
    match_config: Optional[MatchConfig] = None
    match_id: Optional[str] = None
    venue_name: Optional[str] = None
    court_number: Optional[str] = None

    # ── Device & Calibration ─────────────────────────────
    device: Optional[DeviceInfo] = None
    calibration: Optional[CameraCalibration] = None
    calibration_state: CalibrationState = CalibrationState.NOT_STARTED

    # ── Video ────────────────────────────────────────────
    video_url: Optional[str] = None
    video_local_path: Optional[str] = None
    video_duration_seconds: float = 0.0
    total_frames: int = 0
    fps: float = 30.0
    resolution_width: int = 1920
    resolution_height: int = 1080

    # ── Processing ───────────────────────────────────────
    processing_mode: str = "on_device"  # on_device | cloud | hybrid
    processing_progress: float = 0.0
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None

    # ── Metadata ─────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # ── Subscription ─────────────────────────────────────
    user_id: Optional[str] = None
    subscription_tier: str = "free"


class SessionTimeline(BaseModel):
    """Timestamp-indexed event stream for a session."""
    session_id: str
    events: list[dict] = Field(
        default_factory=list,
        description="Time-ordered event dicts with type, timestamp, and payload"
    )
    duration_ms: int = 0
    event_count: int = 0

    def add_event(self, event_type: str, timestamp_ms: int, payload: dict) -> None:
        self.events.append({
            "type": event_type,
            "timestamp_ms": timestamp_ms,
            "payload": payload
        })
        self.event_count = len(self.events)
        if timestamp_ms > self.duration_ms:
            self.duration_ms = timestamp_ms


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list[CaptureSession]
    total: int
    page: int = 1
    page_size: int = 20
