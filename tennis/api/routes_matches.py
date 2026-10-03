"""
Match routes — Match state, scoring, and timeline.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tennis.engine.scoring import ScoringEngine
from tennis.models.match import (
    MatchConfig,
    MatchState,
    PointOutcomeType,
)

router = APIRouter()

# ── In-memory match engines ──────────────────────────────────────────────────
_engines: dict[str, ScoringEngine] = {}


@router.post("/", response_model=MatchState, status_code=201)
async def create_match(
    player1_name: str = "Player 1",
    player2_name: str = "Player 2",
    match_format: str = "best_of_3",
    surface: str = "hard",
    no_ad: bool = False,
):
    """Create and start a new match."""
    config = MatchConfig(
        match_format=match_format,
        surface=surface,
        no_ad_scoring=no_ad,
    )
    engine = ScoringEngine(config)
    state = engine.start_match(
        player1_id="p1", player2_id="p2",
        player1_name=player1_name, player2_name=player2_name,
    )
    _engines[state.id] = engine
    return state


@router.get("/{match_id}", response_model=MatchState)
async def get_match(match_id: str):
    """Get full match state."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Match not found")
    return engine.get_match_state()


@router.post("/{match_id}/score")
async def score_point(
    match_id: str,
    winner: str,
    outcome_type: PointOutcomeType = PointOutcomeType.WINNER,
):
    """Score a point in the match."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Match not found")
    try:
        pt = engine.score_point(winner, outcome_type)
        return {
            "point": pt.model_dump(),
            "score_display": engine.get_score_display(),
            "match_over": engine.is_match_over(),
            "winner": engine.get_winner(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{match_id}/undo")
async def undo_point(match_id: str):
    """Undo the last scored point (for challenge corrections)."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Match not found")
    state = engine.undo_last_point()
    if not state:
        raise HTTPException(status_code=400, detail="No points to undo")
    return {"score_display": engine.get_score_display(), "state": state.model_dump()}


@router.get("/{match_id}/score")
async def get_score(match_id: str):
    """Get current score display."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Match not found")
    return {
        "score_display": engine.get_score_display(),
        "server": engine.get_server(),
        "match_over": engine.is_match_over(),
    }


@router.get("/{match_id}/timeline")
async def get_timeline(match_id: str):
    """Get the points timeline for a match."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Match not found")
    state = engine.get_match_state()
    return {
        "total_points": state.total_points_played,
        "timeline": [p.model_dump() for p in state.points_timeline],
    }
