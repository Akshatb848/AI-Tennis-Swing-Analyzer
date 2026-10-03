"""
Coaching routes — AI coaching feedback, swing analysis, weekly goals.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tennis.engine.coaching_engine import CoachingEngine
from tennis.models.coaching import CoachingFeedback, SwingAnalysis, WeeklyGoal
from tennis.models.player import PlayerProfile, PlayerSessionStats, PlayerStyleEmbedding

router = APIRouter()

_coaching = CoachingEngine()
_feedback_store: dict[str, list[CoachingFeedback]] = {}
_goals_store: dict[str, WeeklyGoal] = {}
_embeddings: dict[str, PlayerStyleEmbedding] = {}


@router.post("/analyze-swing", response_model=SwingAnalysis)
async def analyze_swing(
    session_id: str,
    player_id: str,
    shot_type: str = "forehand",
    pose_sequence: list[dict] = [],
):
    """Analyze a single swing from pose data."""
    return _coaching.analyze_swing(session_id, player_id, pose_sequence, shot_type)


@router.post("/feedback")
async def generate_feedback(
    player_name: str = "Player",
    player_id: str = "p1",
    session_id: str = "s1",
    points_won: int = 20,
    total_points: int = 40,
    first_serve_pct: float = 60.0,
    winners: int = 10,
    unforced_errors: int = 8,
    aces: int = 2,
    double_faults: int = 1,
):
    """Generate AI coaching feedback for a session."""
    player = PlayerProfile(id=player_id, name=player_name)
    stats = PlayerSessionStats(
        player_id=player_id, session_id=session_id,
        total_points=total_points, points_won=points_won,
        first_serve_pct=first_serve_pct, winners=winners,
        unforced_errors=unforced_errors, aces=aces,
        double_faults=double_faults,
    )
    embedding = _embeddings.get(player_id)
    swings = [_coaching.analyze_swing(session_id, player_id, [{"mock": True}], "forehand")]
    feedback = _coaching.generate_feedback(player, stats, swings, style_embedding=embedding)

    if player_id not in _feedback_store:
        _feedback_store[player_id] = []
    _feedback_store[player_id].append(feedback)
    return feedback.model_dump()


@router.post("/embedding/{player_id}")
async def build_embedding(player_id: str, player_name: str = "Player"):
    """Build/update player style embedding."""
    player = PlayerProfile(id=player_id, name=player_name)
    # In production, aggregate from real session stats
    mock_stats = [PlayerSessionStats(
        player_id=player_id, session_id="mock",
        first_serve_pct=62, winners=12, unforced_errors=8,
        aces=3, net_points_won=4, net_points_total=8,
    )]
    emb = _coaching.build_style_embedding(player, mock_stats)
    _embeddings[player_id] = emb
    return emb.model_dump()


@router.get("/embedding/{player_id}")
async def get_embedding(player_id: str):
    """Get player style embedding."""
    if player_id not in _embeddings:
        raise HTTPException(404, "Embedding not found")
    return _embeddings[player_id].model_dump()


@router.post("/weekly-goal/{player_id}")
async def generate_weekly_goal(player_id: str, player_name: str = "Player"):
    """Generate adaptive weekly goal."""
    player = PlayerProfile(id=player_id, name=player_name)
    recent = _feedback_store.get(player_id, [])
    current = _goals_store.get(player_id)
    goal = _coaching.generate_weekly_goal(player, recent, current)
    _goals_store[player_id] = goal
    return goal.model_dump()


@router.get("/weekly-goal/{player_id}")
async def get_weekly_goal(player_id: str):
    """Get current weekly goal."""
    if player_id not in _goals_store:
        raise HTTPException(404, "No goal set")
    return _goals_store[player_id].model_dump()
