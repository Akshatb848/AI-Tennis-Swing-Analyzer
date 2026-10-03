"""
Player data models — Profiles, style embeddings, and session stats.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from tennis.timeutil import utcnow


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    AMBIDEXTROUS = "ambidextrous"


class PlayStyle(str, Enum):
    AGGRESSIVE_BASELINER = "aggressive_baseliner"
    DEFENSIVE_BASELINER = "defensive_baseliner"
    SERVE_AND_VOLLEY = "serve_and_volley"
    ALL_COURT = "all_court"
    COUNTERPUNCHER = "counterpuncher"
    BIG_SERVER = "big_server"
    UNKNOWN = "unknown"


class SkillLevel(str, Enum):
    BEGINNER = "beginner"           # NTRP 1.0–2.5
    INTERMEDIATE = "intermediate"   # NTRP 3.0–3.5
    ADVANCED = "advanced"           # NTRP 4.0–4.5
    EXPERT = "expert"               # NTRP 5.0–5.5
    PROFESSIONAL = "professional"   # NTRP 6.0–7.0


class PlayerProfile(BaseModel):
    """Full player profile with biographical and play-style data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str | None = None
    avatar_url: str | None = None
    handedness: Handedness = Handedness.RIGHT
    backhand_type: str = Field(default="two_handed", description="one_handed | two_handed")
    play_style: PlayStyle = PlayStyle.UNKNOWN
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    ntrp_rating: float | None = Field(default=None, ge=1.0, le=7.0)
    height_cm: float | None = None
    weight_kg: float | None = None
    age: int | None = None
    years_playing: int | None = None

    # ── Historical aggregates ────────────────────────────
    total_matches: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    avg_first_serve_pct: float = 0.0
    avg_winners_per_match: float = 0.0
    avg_unforced_errors_per_match: float = 0.0

    # ── Metadata ─────────────────────────────────────────
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PlayerStyleEmbedding(BaseModel):
    """
    64-dimensional vector capturing a player's play patterns.
    Built from pose sequences, shot distributions, movement patterns, 
    and tactical tendencies across multiple sessions.
    """
    player_id: str
    embedding: list[float] = Field(
        default_factory=lambda: [0.0] * 64,
        min_length=64,
        max_length=64,
        description="64-dim style vector"
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Named embedding component scores"
    )
    # Component breakdown
    aggression_score: float = Field(default=0.5, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    net_approach_tendency: float = Field(default=0.5, ge=0.0, le=1.0)
    serve_power_index: float = Field(default=0.5, ge=0.0, le=1.0)
    court_coverage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    shot_variety_score: float = Field(default=0.5, ge=0.0, le=1.0)
    pressure_performance: float = Field(default=0.5, ge=0.0, le=1.0)
    endurance_index: float = Field(default=0.5, ge=0.0, le=1.0)
    updated_at: datetime = Field(default_factory=utcnow)
    sessions_analyzed: int = 0


class PlayerSessionStats(BaseModel):
    """Per-session aggregated statistics for a player."""
    player_id: str
    session_id: str
    match_id: str | None = None

    # ── Serve stats ──────────────────────────────────────
    total_serve_points: int = 0
    first_serves_in: int = 0
    first_serve_pct: float = 0.0
    second_serves_in: int = 0
    second_serve_pct: float = 0.0
    aces: int = 0
    double_faults: int = 0
    avg_first_serve_speed_mph: float = 0.0
    avg_second_serve_speed_mph: float = 0.0
    max_serve_speed_mph: float = 0.0
    first_serve_win_pct: float = 0.0
    second_serve_win_pct: float = 0.0

    # ── Return stats ─────────────────────────────────────
    total_return_points: int = 0
    return_points_won: int = 0
    return_win_pct: float = 0.0

    # ── Rally stats ──────────────────────────────────────
    total_points: int = 0
    points_won: int = 0
    winners: int = 0
    unforced_errors: int = 0
    forced_errors: int = 0
    winner_to_ue_ratio: float = 0.0
    avg_rally_length: float = 0.0
    longest_rally: int = 0

    # ── Shot stats ───────────────────────────────────────
    forehand_winners: int = 0
    backhand_winners: int = 0
    forehand_errors: int = 0
    backhand_errors: int = 0
    net_points_won: int = 0
    net_points_total: int = 0
    net_approach_success_pct: float = 0.0

    # ── Movement stats ───────────────────────────────────
    total_distance_meters: float = 0.0
    avg_recovery_time_ms: float = 0.0
    court_coverage_pct: float = 0.0

    # ── Break point stats ────────────────────────────────
    break_points_won: int = 0
    break_points_total: int = 0
    break_point_conversion_pct: float = 0.0
    break_points_saved: int = 0
    break_points_faced: int = 0
    break_point_save_pct: float = 0.0

    # ── Computed at end of session ───────────────────────
    computed_at: datetime = Field(default_factory=utcnow)


class PlayerComparison(BaseModel):
    """Side-by-side player comparison for analytics."""
    player1: PlayerSessionStats
    player2: PlayerSessionStats
    player1_name: str
    player2_name: str
    highlight_stats: list[str] = Field(
        default_factory=list,
        description="Stats where there's a significant difference"
    )
