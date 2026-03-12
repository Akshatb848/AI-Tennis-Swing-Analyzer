"""
Replay routes — Match replay intelligence endpoints.

Provides:
- Frame-indexed overlay timeline
- Point navigation
- Rally navigation
- Shot navigation
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Optional

from tennis.engine.review_engine import ReviewEngine

router = APIRouter()

# Shared references
from tennis.api.routes_matches import _engines

_review_engine = ReviewEngine()


@router.get("/{session_id}/timeline")
async def get_replay_timeline(session_id: str, start_frame: int = 0, end_frame: int = -1):
    """Full replay timeline with overlay data for video playback."""
    # Get session data from engines
    engine = _engines.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    state = engine.get_match_state()
    timeline = []

    for i, pt in enumerate(state.points_timeline):
        point_entry = {
            "point_number": pt.point_number,
            "start_frame": pt.timestamp_start_ms,
            "end_frame": pt.timestamp_end_ms,
            "score_before": pt.score_before,
            "score_after": pt.score_after,
            "outcome_type": pt.outcome_type.value,
            "winner_id": pt.winner_id,
            "rally_length": pt.rally_length,
            "shots": [
                {
                    "shot_type": s.shot_type.value,
                    "player_id": s.player_id,
                    "speed_mph": s.speed_mph,
                    "frame_number": s.frame_number,
                    "timestamp_ms": s.timestamp_ms,
                    "placement_zone": s.placement_zone.value if s.placement_zone else None,
                }
                for s in pt.shot_sequence
            ],
        }
        timeline.append(point_entry)

    return {"session_id": session_id, "points": timeline, "total_points": len(timeline)}


@router.get("/{session_id}/points")
async def get_point_navigation(session_id: str):
    """Point navigation index — start/end frames for each point."""
    engine = _engines.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    state = engine.get_match_state()
    nav = []
    for pt in state.points_timeline:
        nav.append({
            "point_number": pt.point_number,
            "start_ms": pt.timestamp_start_ms,
            "end_ms": pt.timestamp_end_ms,
            "duration_s": pt.duration_seconds,
            "score_before": pt.score_before,
            "outcome_type": pt.outcome_type.value,
            "winner_id": pt.winner_id,
        })
    return {"session_id": session_id, "points": nav}


@router.get("/{session_id}/rallies")
async def get_rally_navigation(session_id: str):
    """Rally navigation index — start/end frames for each rally."""
    engine = _engines.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    state = engine.get_match_state()
    rallies = []
    for pt in state.points_timeline:
        rallies.append({
            "point_number": pt.point_number,
            "start_ms": pt.timestamp_start_ms,
            "end_ms": pt.timestamp_end_ms,
            "rally_length": pt.rally_length,
            "outcome_type": pt.outcome_type.value,
        })
    return {"session_id": session_id, "rallies": rallies}


@router.get("/{session_id}/shots")
async def get_shot_navigation(session_id: str):
    """Shot navigation index — frame number for each shot event."""
    engine = _engines.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    state = engine.get_match_state()
    shots = []
    for pt in state.points_timeline:
        for s in pt.shot_sequence:
            shots.append({
                "point_number": pt.point_number,
                "shot_type": s.shot_type.value,
                "player_id": s.player_id,
                "speed_mph": s.speed_mph,
                "frame_number": s.frame_number,
                "timestamp_ms": s.timestamp_ms,
                "placement_zone": s.placement_zone.value if s.placement_zone else None,
            })
    return {"session_id": session_id, "shots": shots, "total_shots": len(shots)}
