"""
Coaching data models — Swing analysis, feedback, weekly goals.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CoachingPriority(str, Enum):
    CRITICAL = "critical"     # Injury risk or major flaw
    HIGH = "high"             # Significant improvement opportunity
    MEDIUM = "medium"         # Notable but not urgent
    LOW = "low"               # Fine-tuning


class StanceType(str, Enum):
    OPEN = "open"
    SEMI_OPEN = "semi_open"
    CLOSED = "closed"
    NEUTRAL = "neutral"


class KineticChainPhase(str, Enum):
    PREPARATION = "preparation"
    BACKSWING = "backswing"
    FORWARD_SWING = "forward_swing"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


class FlawCategory(str, Enum):
    GRIP = "grip"
    FOOTWORK = "footwork"
    RACKET_PATH = "racket_path"
    BODY_ROTATION = "body_rotation"
    CONTACT_POINT = "contact_point"
    FOLLOW_THROUGH = "follow_through"
    BALANCE = "balance"
    TIMING = "timing"
    POSITIONING = "positioning"
    SPLIT_STEP = "split_step"
    SERVE_TOSS = "serve_toss"
    READY_POSITION = "ready_position"


class SwingAnalysis(BaseModel):
    """Analysis of a single swing/shot from pose data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    player_id: str
    point_number: Optional[int] = None
    shot_number: Optional[int] = None

    # ── Shot identification ──────────────────────────────
    shot_type: str = ""           # forehand, backhand, serve, etc.
    stance: StanceType = StanceType.NEUTRAL
    grip_type: Optional[str] = None  # eastern, semi_western, western, continental

    # ── Pose sequence ────────────────────────────────────
    frame_start: int = 0
    frame_end: int = 0
    pose_sequence_length: int = 0
    pose_keypoints_summary: dict = Field(
        default_factory=dict,
        description="Summary stats of keypoint positions across the swing"
    )

    # ── Kinetic chain analysis ───────────────────────────
    kinetic_chain_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score (0-1) for each phase of the kinetic chain"
    )
    overall_kinetic_chain_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall kinetic chain efficiency"
    )

    # ── Biomechanics ─────────────────────────────────────
    hip_shoulder_separation_deg: Optional[float] = None
    trunk_rotation_deg: Optional[float] = None
    elbow_angle_at_contact_deg: Optional[float] = None
    wrist_lag_frames: Optional[int] = None
    racket_head_speed_proxy: Optional[float] = None

    # ── Detected issues ──────────────────────────────────
    detected_flaws: list[DetectedFlaw] = Field(default_factory=list)

    timestamp_ms: int = 0
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class DetectedFlaw(BaseModel):
    """A specific biomechanical flaw detected in a swing."""
    category: FlawCategory
    description: str
    severity: CoachingPriority = CoachingPriority.MEDIUM
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_frame: Optional[int] = None
    suggestion: str = ""
    # How many times this flaw has appeared in recent sessions
    recurrence_count: int = 0
    is_repeatable: bool = False


class CoachingFeedback(BaseModel):
    """AI-generated coaching feedback for a session."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    player_id: str
    match_id: Optional[str] = None

    # ── The One Correction ───────────────────────────────
    primary_correction: str = Field(
        default="",
        description="The single most impactful thing to fix right now"
    )
    primary_correction_priority: CoachingPriority = CoachingPriority.HIGH
    primary_correction_category: Optional[FlawCategory] = None

    # ── Full analysis ────────────────────────────────────
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    tactical_insights: list[str] = Field(default_factory=list)
    drill_suggestions: list[DrillSuggestion] = Field(default_factory=list)

    # ── Session summary ──────────────────────────────────
    session_narrative: str = Field(
        default="",
        description="Natural language summary of the session"
    )
    performance_rating: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Overall session performance score"
    )
    improvement_vs_previous: Optional[float] = Field(
        default=None,
        description="Percentage improvement vs last session"
    )

    # ── Comparison ───────────────────────────────────────
    comparison_to_reference: Optional[str] = Field(
        default=None,
        description="How player compares to skill-appropriate reference"
    )

    # ── Metadata ─────────────────────────────────────────
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0"
    confidence: float = 0.0


class DrillSuggestion(BaseModel):
    """A suggested drill to address a specific weakness."""
    name: str
    description: str
    target_flaw: FlawCategory
    duration_minutes: int = 15
    difficulty: str = "intermediate"  # beginner | intermediate | advanced
    equipment_needed: list[str] = Field(default_factory=list)
    video_reference_url: Optional[str] = None


class WeeklyGoal(BaseModel):
    """Adaptive weekly goal for progressive improvement."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    player_id: str
    week_start: datetime
    week_end: datetime

    # ── Goals ────────────────────────────────────────────
    primary_goal: str
    secondary_goals: list[str] = Field(default_factory=list)
    target_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric name → target value"
    )

    # ── Progress ─────────────────────────────────────────
    sessions_target: int = 3
    sessions_completed: int = 0
    progress_pct: float = 0.0
    is_achieved: bool = False
    actual_metrics: dict[str, float] = Field(default_factory=dict)

    # ── Context ──────────────────────────────────────────
    based_on_sessions: list[str] = Field(
        default_factory=list, description="Session IDs that informed this goal"
    )
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PlayerProgressReport(BaseModel):
    """Long-term progress report across multiple sessions."""
    player_id: str
    player_name: str
    report_period_start: datetime
    report_period_end: datetime

    # ── Trend data ───────────────────────────────────────
    sessions_in_period: int = 0
    skill_trajectory: list[dict] = Field(
        default_factory=list,
        description="Date → skill metrics snapshots"
    )
    recurring_flaws: list[DetectedFlaw] = Field(default_factory=list)
    resolved_flaws: list[str] = Field(default_factory=list)
    goals_achieved: int = 0
    goals_total: int = 0

    # ── AI narrative ─────────────────────────────────────
    progress_summary: str = ""
    next_milestone: str = ""
    estimated_ntrp_change: Optional[float] = None


class CoachingInsight(BaseModel):
    """Evidence-based coaching insight from continuous analysis.

    Each insight includes:
    - What was observed (the issue)
    - Supporting evidence from match data
    - Suggested improvement action
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    player_id: str = ""

    # ── The observation ──────────────────────────────────
    observed_issue: str = Field(
        default="",
        description="Clear, specific description of the observed issue"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Data points supporting the observation"
    )
    suggested_improvement: str = Field(
        default="",
        description="Actionable improvement recommendation"
    )

    # ── Metadata ─────────────────────────────────────────
    category: str = ""  # positioning, shot_selection, movement, consistency, tactics
    occurrences: int = 1
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    first_seen_point: int = 0
    last_seen_point: int = 0
    detected_at: datetime = Field(default_factory=datetime.utcnow)
