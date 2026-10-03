"""
Stats routes — Player statistics, match analytics, and leaderboards.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tennis.engine.stats_calculator import StatsCalculator

router = APIRouter()

# Reference to match engines (shared with routes_matches)
from tennis.api.routes_matches import _engines

_stats_calc = StatsCalculator()


@router.get("/match/{match_id}")
async def get_match_stats(match_id: str):
    """Get comprehensive stats for a match."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    comparison = _stats_calc.compute_match_comparison(engine.get_match_state())
    return comparison.model_dump()


@router.get("/match/{match_id}/player/{player_id}")
async def get_player_match_stats(match_id: str, player_id: str):
    """Get stats for a specific player in a match."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    stats = _stats_calc.compute_player_stats(player_id, engine.get_match_state())
    return stats.model_dump()


@router.get("/match/{match_id}/heatmap/{player_id}")
async def get_placement_heatmap(match_id: str, player_id: str):
    """Get shot placement heatmap for a player."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    heatmap = _stats_calc.generate_placement_heatmap(state.points_timeline, player_id)
    return {"player_id": player_id, "heatmap": heatmap}


@router.get("/match/{match_id}/speeds/{player_id}")
async def get_speed_distribution(match_id: str, player_id: str):
    """Get shot speed distribution for a player."""
    engine = _engines.get(match_id)
    if not engine:
        raise HTTPException(404, "Match not found")
    state = engine.get_match_state()
    speeds = _stats_calc.generate_speed_distribution(state.points_timeline, player_id)
    return {"player_id": player_id, "speeds": speeds, "count": len(speeds)}
