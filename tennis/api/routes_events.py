"""
Event routes — Ingest CV events and query event streams.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from tennis.engine.event_processor import EventProcessor
from tennis.models.events import (
    BallEvent,
    ChallengeStatus,
    EventBatch,
    LineCallEvent,
    PlayerEvent,
)

router = APIRouter()

# ── In-memory stores ────────────────────────────────────────────────────────
_ball_events: dict[str, list[BallEvent]] = {}
_player_events: dict[str, list[PlayerEvent]] = {}
_line_calls: dict[str, list[LineCallEvent]] = {}
_processors: dict[str, EventProcessor] = {}


def _get_processor(session_id: str) -> EventProcessor:
    if session_id not in _processors:
        _processors[session_id] = EventProcessor()
    return _processors[session_id]


@router.post("/batch", status_code=201)
async def ingest_event_batch(batch: EventBatch):
    """Ingest a batch of CV events from device."""
    sid = batch.session_id
    processor = _get_processor(sid)

    if sid not in _ball_events:
        _ball_events[sid] = []
    if sid not in _player_events:
        _player_events[sid] = []
    if sid not in _line_calls:
        _line_calls[sid] = []

    results = []
    for be in batch.ball_events:
        _ball_events[sid].append(be)
        result = processor.process_ball_event(be, batch.player_events[:2])
        if result:
            if "line_call" in result:
                _line_calls[sid].append(result["line_call"])
            results.append(result)

    for pe in batch.player_events:
        _player_events[sid].append(pe)

    return {
        "session_id": sid,
        "ball_events_ingested": len(batch.ball_events),
        "player_events_ingested": len(batch.player_events),
        "tennis_events_generated": len(results),
        "results": results,
    }


@router.get("/{session_id}/ball")
async def get_ball_events(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get ball events for a session."""
    events = _ball_events.get(session_id, [])
    return {
        "total": len(events),
        "events": [e.model_dump() for e in events[offset:offset + limit]],
    }


@router.get("/{session_id}/line-calls")
async def get_line_calls(session_id: str):
    """Get all line call decisions for a session."""
    calls = _line_calls.get(session_id, [])
    return {
        "total": len(calls),
        "line_calls": [lc.model_dump() for lc in calls],
    }


@router.post("/{session_id}/challenge/{line_call_id}")
async def challenge_line_call(
    session_id: str,
    line_call_id: str,
    player_id: str,
):
    """Challenge a line call decision."""
    calls = _line_calls.get(session_id, [])
    lc = next((c for c in calls if c.id == line_call_id), None)
    if not lc:
        raise HTTPException(404, "Line call not found")

    lc.is_challenged = True
    lc.challenged_by_player_id = player_id
    lc.original_verdict = lc.verdict
    # In production, re-analyze frames around the bounce
    # For now, keep the AI decision
    lc.final_verdict = lc.verdict
    lc.challenge_status = ChallengeStatus.CONFIRMED

    return {
        "challenge_result": "confirmed" if lc.verdict == lc.original_verdict else "overturned",
        "original": lc.original_verdict,
        "final": lc.final_verdict,
        "confidence": lc.confidence,
        "distance_from_line_cm": lc.distance_from_line_cm,
    }
