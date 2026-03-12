"""
Line call routes — Automated line call history and challenge system.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException

from tennis.engine.line_calling import LineCallingSystem

router = APIRouter()

# Active line calling systems per session
_systems: dict[str, LineCallingSystem] = {}


def get_or_create_system(session_id: str, is_doubles: bool = False) -> LineCallingSystem:
    if session_id not in _systems:
        _systems[session_id] = LineCallingSystem(session_id=session_id, is_doubles=is_doubles)
    return _systems[session_id]


@router.get("/{session_id}")
async def get_line_calls(session_id: str):
    """Get all line calls for a session."""
    system = _systems.get(session_id)
    if not system:
        return {"session_id": session_id, "calls": [], "summary": {"total_calls": 0}}

    return {
        "session_id": session_id,
        "calls": [c.to_display() for c in system.history.calls],
        "summary": system.history.get_summary(),
    }


@router.get("/{session_id}/{call_id}/replay")
async def get_replay(session_id: str, call_id: str):
    """Get frame window for replay of a specific line call."""
    system = _systems.get(session_id)
    if not system:
        raise HTTPException(status_code=404, detail="Session not found")

    call = system.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Line call not found")

    return {
        "call_id": call.call_id,
        "verdict": call.verdict.value,
        "confidence": call.confidence,
        "distance_cm": call.distance_from_line_cm,
        "replay_start_frame": call.replay_start_frame,
        "replay_end_frame": call.replay_end_frame,
        "replay_start_ms": call.replay_start_ms,
        "replay_end_ms": call.replay_end_ms,
        "court_x": call.court_x,
        "court_y": call.court_y,
    }


@router.post("/{call_id}/challenge")
async def challenge_call(call_id: str, session_id: str, challenger_id: str):
    """Submit a challenge for a line call."""
    system = _systems.get(session_id)
    if not system:
        raise HTTPException(status_code=404, detail="Session not found")

    result = system.challenge_call(call_id, challenger_id)
    if not result:
        raise HTTPException(status_code=404, detail="Call not found or already challenged")

    return {
        "call_id": result.call_id,
        "original_verdict": result.original_verdict.value if result.original_verdict else None,
        "new_verdict": result.verdict.value,
        "challenge_status": result.challenge_status.value,
        "overturned": result.challenge_status.value == "overturned",
    }
