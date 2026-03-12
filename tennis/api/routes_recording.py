"""
Recording routes — Guided match setup, recording control, and video ingest.
"""

from __future__ import annotations

import os
import uuid
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks

from tennis.engine.recording import (
    RecordingSession, MatchSetupConfig, MatchType, Environment, Handedness,
)
from tennis.engine.live_pipeline import LivePipeline, SessionResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Active recording sessions
_recordings: dict[str, RecordingSession] = {}

# Active live pipelines (one per session)
_pipelines: dict[str, LivePipeline] = {}

# Completed session results
_session_results: dict[str, SessionResult] = {}

# Upload directory
UPLOAD_DIR = "./uploads/tennisiq"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/setup")
async def setup_recording(
    match_type: str = "singles",
    environment: str = "outdoor",
    player_names: Optional[list[str]] = None,
    court_surface: str = "hard",
):
    """
    Step 1: Configure match before recording.
    All fields are optional — defaults produce a ready-to-record session.
    """
    config = MatchSetupConfig(
        match_type=MatchType(match_type),
        environment=Environment(environment),
        player_names=player_names or ["Player 1", "Player 2"],
        court_surface=court_surface,
    )
    if match_type == "doubles":
        config.player_count = 4
        if not player_names:
            config.player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
        config.player_handedness = [Handedness.AUTO] * 4

    session = RecordingSession()
    result = session.setup(config)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("errors", []))

    _recordings[session.id] = session
    return result


@router.post("/{session_id}/start")
async def start_recording(session_id: str):
    """Step 2: Begin recording. Processing starts automatically."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    result = session.start_recording()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/{session_id}/stop")
async def stop_recording(session_id: str):
    """Step 3: Stop recording. Segmentation finalized automatically."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    result = session.stop_recording()

    # Stop pipeline if running
    pipeline = _pipelines.get(session_id)
    if pipeline and pipeline.is_running:
        session_result = pipeline.stop()
        _session_results[session_id] = session_result

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.get("/{session_id}/status")
async def recording_status(session_id: str):
    """Check recording status and progress."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    response = {
        "session_id": session.id,
        "state": session.state.value,
        "frames_processed": session.frame_count,
        "points_detected": len(session.points),
        "line_calls": len(session.line_calls),
        "duration_seconds": session._get_duration_seconds(),
    }

    # Add pipeline status if active
    pipeline = _pipelines.get(session_id)
    if pipeline:
        response["pipeline"] = pipeline.get_live_state()

    return response


@router.get("/{session_id}/summary")
async def recording_summary(session_id: str):
    """Get complete session summary after recording stops."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    summary = session.get_summary()

    # Merge pipeline results if available
    pipeline_result = _session_results.get(session_id)
    if pipeline_result:
        summary["pipeline_stats"] = {
            "total_frames": pipeline_result.total_frames,
            "avg_latency_ms": pipeline_result.avg_latency_ms,
            "errors": pipeline_result.errors,
        }

    return summary


@router.post("/{session_id}/ingest-video")
async def ingest_video(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video file and trigger full pipeline processing.

    The video is processed asynchronously — check status via GET /status.
    """
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    # Save upload
    file_ext = os.path.splitext(file.filename or "video.mp4")[1]
    video_path = os.path.join(UPLOAD_DIR, f"{session_id}{file_ext}")

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    logger.info("Video uploaded: %s (%.1f MB)", video_path, len(content) / 1024 / 1024)

    # Create and start pipeline
    pipeline = LivePipeline(target_fps=15.0)  # Process at 15fps for efficiency
    _pipelines[session_id] = pipeline

    # Process in background
    background_tasks.add_task(_process_video, session_id, video_path, pipeline)

    return {
        "session_id": session_id,
        "video_path": video_path,
        "status": "processing",
        "message": "Video uploaded and processing started. Check /status for progress.",
    }


async def _process_video(session_id: str, video_path: str, pipeline: LivePipeline):
    """Background task to process an uploaded video."""
    try:
        result = await pipeline.process_video(video_path)
        _session_results[session_id] = result

        # Update recording session with pipeline data
        session = _recordings.get(session_id)
        if session and pipeline.recording:
            # Transfer events from pipeline recording to session
            session.points = pipeline.recording.points
            session.line_calls = pipeline.recording.line_calls

        logger.info(
            "Video processing complete for session %s: %d frames, %d points",
            session_id, result.total_frames, result.points_detected,
        )
    except Exception as e:
        logger.error("Video processing failed for session %s: %s", session_id, e)
        _session_results[session_id] = SessionResult(
            session_id=session_id,
            errors=[str(e)],
        )


@router.get("/{session_id}/live")
async def live_state(session_id: str):
    """Get real-time match state (for live dashboards and overlays)."""
    pipeline = _pipelines.get(session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="No active pipeline for this session")

    return pipeline.get_live_state()
