"""
Session routes — CRUD operations for capture sessions.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query

from tennis.models.match import MatchConfig
from tennis.models.session import (
    CaptureSession,
    SessionListResponse,
    SessionMode,
    SessionStatus,
)
from tennis.timeutil import utcnow

router = APIRouter()

# ── In-memory store (swap for DB in production) ──────────────────────────────
_sessions: dict[str, CaptureSession] = {}


@router.post("/", response_model=CaptureSession, status_code=201)
async def create_session(
    mode: SessionMode = SessionMode.MATCH,
    player_names: list[str] = ["Player 1", "Player 2"],  # noqa: B006 - FastAPI deep-copies parameter defaults per request; default is published in the OpenAPI schema
    court_surface: str = "hard",
    venue_name: str | None = None,
):
    """Create a new capture session."""
    session = CaptureSession(
        mode=mode,
        player_names=player_names,
        player_ids=[str(uuid.uuid4()) for _ in player_names],
        court_surface=court_surface,
        venue_name=venue_name,
        match_config=MatchConfig() if mode == SessionMode.MATCH else None,
    )
    _sessions[session.id] = session
    return session


@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    mode: SessionMode | None = None,
    status: SessionStatus | None = None,
):
    """List all sessions with filtering and pagination."""
    items = list(_sessions.values())
    if mode:
        items = [s for s in items if s.mode == mode]
    if status:
        items = [s for s in items if s.status == status]
    items.sort(key=lambda s: s.created_at, reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    items = items[start:start + page_size]
    return SessionListResponse(sessions=items, total=total, page=page, page_size=page_size)


@router.get("/{session_id}", response_model=CaptureSession)
async def get_session(session_id: str):
    """Get a session by ID."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}/status")
async def update_session_status(session_id: str, status: SessionStatus):
    """Update session status."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.status = status
    session.updated_at = utcnow()
    if status == SessionStatus.RECORDING:
        session.processing_started_at = utcnow()
    elif status == SessionStatus.COMPLETED:
        session.processing_completed_at = utcnow()
    return {"id": session_id, "status": status}


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
