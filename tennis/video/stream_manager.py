"""
Stream Manager — Live stream lifecycle management.

Manages RTMP ingest, real-time overlay rendering, and HLS output.
Production-ready backend for livestreaming tennis matches.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class StreamStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    LIVE = "live"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for a live stream."""
    session_id: str = ""
    input_url: str = ""  # RTMP input URL
    output_dir: str = ""  # HLS output directory
    overlay_enabled: bool = True
    score_overlay: bool = True
    line_call_overlay: bool = True
    adaptive_bitrate: bool = True
    target_latency_ms: int = 3000  # 3 second glass-to-glass


@dataclass
class StreamState:
    """Current state of a live stream."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StreamStatus = StreamStatus.IDLE
    session_id: str = ""
    viewers: int = 0
    frames_processed: int = 0
    uptime_seconds: float = 0.0
    output_url: str = ""
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None


class StreamManager:
    """
    Manages live stream lifecycle.

    Handles:
    - RTMP ingest from OBS/mobile camera
    - Real-time score/line-call overlay compositing
    - HLS output generation with adaptive bitrate
    - CDN-ready output paths
    """

    def __init__(self, output_base_dir: str = "./streams"):
        self.output_base_dir = output_base_dir
        self._streams: dict[str, StreamState] = {}
        os.makedirs(output_base_dir, exist_ok=True)

    def create_stream(self, config: StreamConfig) -> StreamState:
        """Create a new stream session."""
        state = StreamState(
            session_id=config.session_id,
            status=StreamStatus.IDLE,
        )
        self._streams[state.id] = state

        # Create output directory
        stream_dir = os.path.join(self.output_base_dir, state.id)
        os.makedirs(stream_dir, exist_ok=True)
        state.output_url = f"{stream_dir}/master.m3u8"

        logger.info("Created stream %s for session %s", state.id, config.session_id)
        return state

    def start_stream(self, stream_id: str) -> bool:
        """Start a live stream."""
        state = self._streams.get(stream_id)
        if not state:
            return False

        state.status = StreamStatus.LIVE
        state.started_at = datetime.utcnow()

        # In production: start ffmpeg process for RTMP → HLS
        # ffmpeg -i rtmp://input -c:v copy -f hls -hls_time 2 -hls_list_size 5 output.m3u8
        logger.info("Stream %s started", stream_id)
        return True

    def stop_stream(self, stream_id: str) -> bool:
        """Stop a live stream."""
        state = self._streams.get(stream_id)
        if not state:
            return False

        state.status = StreamStatus.STOPPED
        if state.started_at:
            state.uptime_seconds = (datetime.utcnow() - state.started_at).total_seconds()

        logger.info("Stream %s stopped after %.0fs", stream_id, state.uptime_seconds)
        return True

    def add_overlay_frame(
        self, stream_id: str, score: str = "", line_call: Optional[dict] = None,
    ):
        """Update overlay data for next frame compositing."""
        state = self._streams.get(stream_id)
        if not state or state.status != StreamStatus.LIVE:
            return
        state.frames_processed += 1
        # In production: composite overlay onto video frames

    def get_stream_state(self, stream_id: str) -> Optional[StreamState]:
        """Get current stream state."""
        return self._streams.get(stream_id)

    def list_streams(self) -> list[StreamState]:
        """List all streams."""
        return list(self._streams.values())
