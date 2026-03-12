"""
Analytics routes — Advanced match analytics endpoints.

Provides:
- Shot distribution and success rates
- Rally analytics
- Momentum analysis
- Fatigue indicators
- Improvement areas
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException

from tennis.engine.stats_calculator import StatsCalculator
from tennis.engine.coaching_engine import ContinuousCoach

router = APIRouter()

# Shared references
from tennis.api.routes_matches import _engines

_stats_calc = StatsCalculator()
_coaching_sessions: dict[str, ContinuousCoach] = {}


@router.get("/match/{match_id}/shots")
async def get_shot_distribution(match_id: str, player_id: str = ""):
    """Shot type distribution and success rates."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    pid = player_id or state.player1_id
    return _stats_calc.generate_shot_distribution(state.points_timeline, pid)


@router.get("/match/{match_id}/rallies")
async def get_rally_analytics(match_id: str, player_id: str = ""):
    """Rally length analytics with distribution and win rates."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    pid = player_id or state.player1_id
    return _stats_calc.generate_rally_analytics(state.points_timeline, pid)


@router.get("/match/{match_id}/momentum")
async def get_momentum_analysis(match_id: str, player_id: str = ""):
    """Momentum analysis with rolling win percentage."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    pid = player_id or state.player1_id
    return _stats_calc.generate_momentum_analysis(state.points_timeline, pid)


@router.get("/match/{match_id}/fatigue/{player_id}")
async def get_fatigue_indicators(match_id: str, player_id: str):
    """Fatigue indicators by match phase."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    return _stats_calc.generate_fatigue_indicators(state.points_timeline, player_id)


@router.get("/match/{match_id}/improvements/{player_id}")
async def get_improvement_areas(match_id: str, player_id: str):
    """Identify areas for improvement."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    return {"player_id": player_id, "areas": _stats_calc.generate_improvement_areas(state.points_timeline, player_id)}


@router.get("/match/{match_id}/movement/{player_id}")
async def get_movement_analytics(match_id: str, player_id: str):
    """Movement analytics — court coverage and positioning."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    heatmap = _stats_calc.generate_placement_heatmap(state.points_timeline, player_id)
    speeds = _stats_calc.generate_speed_distribution(state.points_timeline, player_id)
    return {
        "player_id": player_id,
        "placement_heatmap": heatmap,
        "speed_distribution": speeds,
        "total_shots": sum(heatmap.values()),
    }
