"""
Analyze Route — Clean job-based video analysis API.

Endpoints:
  POST /api/v1/analyze          Start a new analysis job
  GET  /api/v1/analyze/{id}     Poll job status + progress
  GET  /api/v1/analyze/{id}/results  Get full results when complete
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException

from tennis.video.fast_analyzer import (
    STAGES,
    AnalysisJob,
    JobState,
    create_job,
    get_job,
    run_analysis,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyze"])

# Upload dir where routes_upload.py stores files
UPLOAD_DIR = os.environ.get("VIDEO_UPLOAD_DIR", "./uploads/tennisiq")


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  — start analysis
# ─────────────────────────────────────────────────────────────────────────────

@router.post("")
async def start_analysis(
    background_tasks: BackgroundTasks,
    session_id: str   = Form(...),
    upload_id:  str   = Form(...),
    match_type: str   = Form("singles"),
    filename:   str   = Form("video.mp4"),
):
    """
    Start a video analysis job.

    The video must already be uploaded via POST /upload.
    Accepts: session_id, upload_id, match_type, filename.
    Returns immediately with job_id for polling.
    """
    # Locate the uploaded file
    video_path = _find_video(upload_id, filename)
    if not video_path:
        # If we can't find it by upload_id, try any mp4 in upload dir with that name
        video_path = _find_video_by_name(filename)

    if not video_path:
        raise HTTPException(
            status_code=404,
            detail=f"Uploaded video not found for upload_id={upload_id}. "
                   "Ensure the file was uploaded via POST /api/v1/upload first.",
        )

    job = create_job(session_id=session_id, video_path=video_path, match_type=match_type)
    job.upload_id = upload_id  # stored so results endpoint can build video stream URL
    background_tasks.add_task(_run_job, job)

    logger.info("Started analysis job %s for session %s (%s)",
                job.job_id, session_id, video_path)

    return {
        "job_id":     job.job_id,
        "session_id": session_id,
        "upload_id":  upload_id,
        "state":      job.state.value,
        "message":    "Analysis started",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /analyze/{job_id}  — poll progress
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}")
async def get_analysis_status(job_id: str):
    """
    Poll the progress of an analysis job.

    Returns stage name, progress 0-100, and live stats.
    State is one of: queued | running | complete | failed
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    stage_idx  = job.stage_index
    stage_name = STAGES[stage_idx][1] if stage_idx < len(STAGES) else "Processing"

    # Partial live stats visible while running
    motion = job.results.get("motion", {})
    ball   = job.results.get("ball", {})

    return {
        "job_id":           job.job_id,
        "session_id":       job.session_id,
        "state":            job.state.value,
        "stage_index":      stage_idx,
        "stage_name":       stage_name,
        "stages":           [{"id": k, "label": v} for k, v in STAGES],
        "progress":         round(job.progress, 1),
        "error":            job.error,

        # Live metadata
        "duration_seconds": job.duration_seconds,
        "fps":              job.fps,
        "frames_total":     job.total_frames,
        "frames_processed": int(job.total_frames * job.progress / 100),

        # Live stats (populated as stages complete)
        "players_detected": motion.get("player_count_estimate", 0),
        "ball_detections":  ball.get("detections", 0),
        "points_detected":  job.results.get("stats", {}).get("total_points", 0),
        "line_calls":       0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /analyze/{job_id}/results  — full results
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/results")
async def get_analysis_results(job_id: str):
    """
    Return the complete analysis results.
    Only available when state == 'complete'.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.state != JobState.COMPLETE:
        raise HTTPException(status_code=409, detail="Analysis not yet complete")

    return {
        "job_id":     job.job_id,
        "session_id": job.session_id,
        "match_type": job.match_type,
        "upload_id":  getattr(job, "upload_id", ""),
        "video": {
            "duration_seconds": job.duration_seconds,
            "fps":              job.fps,
            "total_frames":     job.total_frames,
            "width":            job.width,
            "height":           job.height,
        },
        "results": job.results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /analyze/{job_id}/coaching  — coaching report
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/coaching")
async def get_coaching_report(job_id: str):
    """Return the AI coaching report. Available once analysis is complete."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.state != JobState.COMPLETE:
        raise HTTPException(status_code=409, detail="Analysis not yet complete")

    coaching = job.results.get("coaching_report", {})
    if not coaching:
        raise HTTPException(status_code=404, detail="Coaching report not generated")

    return {
        "job_id": job.job_id,
        "match_type": job.match_type,
        "coaching_report": coaching,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /analyze/{job_id}/overlays  — frame overlay descriptors
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/overlays")
async def get_frame_overlays(job_id: str):
    """
    Return timed frame overlay descriptors for the video player.

    Each overlay contains: timestamp, duration, label, color, coaching_text,
    and a list of canvas drawing instructions (type, x, y, etc.).
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.state != JobState.COMPLETE:
        raise HTTPException(status_code=409, detail="Analysis not yet complete")

    overlays = job.results.get("frame_overlays", [])
    return {
        "job_id": job.job_id,
        "total": len(overlays),
        "overlays": overlays,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /analyze/{job_id}/voice-script  — timed narration events
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/voice-script")
async def get_voice_script(job_id: str):
    """
    Return timed narration events for the AI voice coach.

    Each event: {timestamp, text, event_type}
    The frontend triggers speech synthesis at each timestamp.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.state != JobState.COMPLETE:
        raise HTTPException(status_code=409, detail="Analysis not yet complete")

    voice = job.results.get("voice_script", {})
    return {
        "job_id": job.job_id,
        "total_cues": voice.get("total_cues", 0),
        "script": voice.get("script", []),
        "intro_text": voice.get("intro_text", ""),
        "summary_text": voice.get("summary_text", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

async def _run_job(job: AnalysisJob):
    try:
        await run_analysis(job)
    except Exception as exc:
        job.state   = JobState.FAILED
        job.error   = str(exc)
        logger.exception("Background job %s failed", job.job_id)


def _find_video(upload_id: str, filename: str) -> Optional[str]:
    """Try to locate the video by upload_id prefix or exact filename."""
    if not os.path.isdir(UPLOAD_DIR):
        return None

    # Look for files starting with upload_id or named filename
    for f in os.listdir(UPLOAD_DIR):
        full = os.path.join(UPLOAD_DIR, f)
        if f.startswith(upload_id) or f == filename:
            if os.path.isfile(full):
                return full
    return None


def _find_video_by_name(filename: str) -> Optional[str]:
    """Last-resort: find any recently uploaded video file."""
    if not os.path.isdir(UPLOAD_DIR):
        return None
    entries = []
    for f in os.listdir(UPLOAD_DIR):
        full = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(full) and f.lower().endswith((".mp4", ".mov", ".mkv", ".m4v")):
            entries.append((os.path.getmtime(full), full))
    if entries:
        entries.sort(reverse=True)
        return entries[0][1]
    return None
