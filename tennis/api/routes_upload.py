"""
Upload routes — Chunked video file upload endpoint + byte-range video streaming.
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, Response

from tennis.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(settings.VIDEO_UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track multi-chunk upload state
_uploads: dict[str, dict] = {}

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
}


# ─────────────────────────────────────────────────────────────────────────────
# POST /  — upload (single or chunked)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/")
async def upload_video(
    file: UploadFile = File(...),
    chunk_index: int = Form(0),
    total_chunks: int = Form(1),
    upload_id: Optional[str] = Form(None),
):
    """
    Upload a video file (single-shot or chunked).

    For single uploads: send the full file with total_chunks=1.
    For chunked uploads: send each chunk with an upload_id, chunk_index, total_chunks.
    On the final chunk the assembled file path is returned.
    """
    if not upload_id:
        upload_id = str(uuid.uuid4())

    # Validate extension
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lower()
    allowed = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format {ext}. Allowed: {allowed}")

    # Read chunk data
    chunk_data = await file.read()

    # Store chunk temporarily
    chunk_dir = UPLOAD_DIR / "chunks" / upload_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_index:05d}"
    chunk_path.write_bytes(chunk_data)

    # Track metadata
    if upload_id not in _uploads:
        _uploads[upload_id] = {
            "upload_id": upload_id,
            "filename": filename,
            "ext": ext,
            "total_chunks": total_chunks,
            "received_chunks": set(),
            "status": "uploading",
            "final_path": None,
            "size_bytes": 0,
        }

    record = _uploads[upload_id]
    record["received_chunks"].add(chunk_index)
    logger.info("Chunk %d/%d received for upload %s", chunk_index + 1, total_chunks, upload_id)

    # If all chunks received, assemble
    if len(record["received_chunks"]) == total_chunks:
        final_path = UPLOAD_DIR / f"{upload_id}{ext}"
        with open(final_path, "wb") as out:
            for i in range(total_chunks):
                cp = chunk_dir / f"chunk_{i:05d}"
                out.write(cp.read_bytes())
                cp.unlink()

        # Cleanup chunk dir
        try:
            chunk_dir.rmdir()
        except OSError:
            pass

        size = final_path.stat().st_size
        record["status"] = "complete"
        record["final_path"] = str(final_path)
        record["size_bytes"] = size

        logger.info("Upload complete: %s (%.1f MB)", final_path, size / 1024 / 1024)
        return {
            "upload_id": upload_id,
            "filename": filename,
            "size_bytes": size,
            "path": str(final_path),
            "status": "complete",
        }

    return {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "received": len(record["received_chunks"]),
        "total": total_chunks,
        "status": "uploading",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /{upload_id}  — upload status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get the status of an ongoing or completed upload."""
    record = _uploads.get(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {
        "upload_id": upload_id,
        "filename": record["filename"],
        "status": record["status"],
        "received_chunks": len(record["received_chunks"]),
        "total_chunks": record["total_chunks"],
        "size_bytes": record["size_bytes"],
        "path": record["final_path"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /{upload_id}/stream  — serve video with byte-range support
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{upload_id}/stream")
async def stream_video(upload_id: str, request: Request):
    """
    Stream an uploaded video file to the browser with byte-range (HTTP 206) support.

    This endpoint is required for HTML5 <video> scrubbing to work correctly.
    The frontend should use this URL as the <video src="...">.

    Also searches the upload directory for files starting with upload_id
    so it works even if the server restarted and lost in-memory state.
    """
    # Try in-memory lookup first
    record = _uploads.get(upload_id)
    file_path: Optional[Path] = None

    if record and record.get("final_path"):
        file_path = Path(record["final_path"])

    # Fallback: scan upload directory (survives server restart)
    if not file_path or not file_path.exists():
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and f.stem == upload_id:
                    file_path = f
                    # Restore in-memory record if missing
                    if upload_id not in _uploads:
                        ext = f.suffix.lower()
                        _uploads[upload_id] = {
                            "upload_id": upload_id,
                            "filename": f.name,
                            "ext": ext,
                            "total_chunks": 1,
                            "received_chunks": {0},
                            "status": "complete",
                            "final_path": str(f),
                            "size_bytes": f.stat().st_size,
                        }
                    break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file for upload_id={upload_id} not found")

    ext = file_path.suffix.lower()
    content_type = CONTENT_TYPES.get(ext, "video/mp4")
    file_size = file_path.stat().st_size

    # Parse Range header for byte-range streaming
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        range_val = range_header.replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_file(path: Path, offset: int, chunk_size: int):
            with open(path, "rb") as f:
                f.seek(offset)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": content_type,
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(
            iter_file(file_path, start, length),
            status_code=206,
            headers=headers,
        )

    # Full file response (no Range header)
    def iter_full(path: Path):
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_full(file_path),
        status_code=200,
        headers={
            "Content-Length": str(file_size),
            "Content-Type": content_type,
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )
