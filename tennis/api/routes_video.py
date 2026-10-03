"""
Video routes — Upload, processing, highlights, and overlay management.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

_videos: dict[str, dict] = {}


@router.post("/upload")
async def upload_video(
    session_id: str,
    file: UploadFile = File(...),  # noqa: B008 - idiomatic FastAPI parameter declaration
):
    """Upload a match video for processing."""
    video_id = str(uuid.uuid4())
    _videos[video_id] = {
        "id": video_id,
        "session_id": session_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "uploaded",
        "processing_progress": 0.0,
        "highlights": [],
        "overlays_applied": False,
    }
    return {"video_id": video_id, "status": "uploaded", "message": "Video queued for processing"}


@router.get("/{video_id}")
async def get_video_status(video_id: str):
    """Get video processing status."""
    video = _videos.get(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return video


@router.post("/{video_id}/process")
async def start_processing(video_id: str):
    """Start video processing pipeline."""
    video = _videos.get(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    video["status"] = "processing"
    return {"video_id": video_id, "status": "processing"}


@router.get("/{video_id}/highlights")
async def get_highlights(video_id: str, max_count: int = 10):
    """Get auto-generated highlights for a video."""
    video = _videos.get(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return {
        "video_id": video_id,
        "highlights": video.get("highlights", []),
        "total": len(video.get("highlights", [])),
    }


@router.get("/{video_id}/exports")
async def get_export_options(video_id: str):
    """Get available export options for a video."""
    return {
        "video_id": video_id,
        "exports": [
            {"type": "full_match", "status": "available"},
            {"type": "condensed_match", "status": "available"},
            {"type": "highlight_reel", "status": "available"},
            {"type": "individual_points", "status": "available"},
        ],
    }
