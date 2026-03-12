"""
Video Capture — Real-time frame capture from files, streams, and cameras.

Supports:
- Local video files (.mp4, .mov, .m4v)
- RTSP/RTMP streams (livestream-ready)
- Camera device indices (mobile/webcam)
- Frame-accurate timestamping
- Configurable FPS with frame skipping for thermal management
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available. Video capture will not function.")


class CaptureSourceType(str, Enum):
    FILE = "file"
    CAMERA = "camera"
    RTSP = "rtsp"
    RTMP = "rtmp"


@dataclass
class VideoMetadata:
    """Metadata extracted from video source."""
    width: int = 0
    height: int = 0
    fps: float = 30.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    codec: str = ""
    source_type: CaptureSourceType = CaptureSourceType.FILE
    source_path: str = ""


@dataclass
class CapturedFrame:
    """A single captured frame with metadata."""
    frame: np.ndarray  # BGR numpy array (H, W, 3)
    frame_number: int
    timestamp_ms: int  # Frame timestamp in ms from start
    capture_time_ns: int  # Monotonic clock at capture
    width: int = 0
    height: int = 0
    is_keyframe: bool = False


class VideoCapture:
    """
    Real-time frame capture from video files, streams, or cameras.

    Usage:
        capture = VideoCapture("match.mp4", target_fps=30)
        async for frame in capture.frames():
            process(frame)
        capture.release()
    """

    def __init__(
        self,
        source: Union[str, int],
        target_fps: float = 30.0,
        max_buffer_frames: int = 120,
        resize_width: Optional[int] = None,
        resize_height: Optional[int] = None,
    ):
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV is required for video capture. Install opencv-python-headless.")

        self.source = source
        self.target_fps = target_fps
        self.max_buffer_frames = max_buffer_frames
        self.resize_width = resize_width
        self.resize_height = resize_height

        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._frame_count: int = 0
        self._start_time_ns: int = 0
        self._is_open: bool = False
        self._source_type = self._detect_source_type(source)

    def open(self) -> VideoMetadata:
        """Open the video source and extract metadata."""
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")

        self._is_open = True
        self._frame_count = 0
        self._start_time_ns = time.monotonic_ns()
        self._metadata = self._extract_metadata()
        logger.info(
            "Opened video: %dx%d @ %.1f fps, %.1f sec, %d frames",
            self._metadata.width, self._metadata.height,
            self._metadata.fps, self._metadata.duration_seconds,
            self._metadata.total_frames,
        )
        return self._metadata

    def get_metadata(self) -> VideoMetadata:
        """Get video metadata. Must call open() first."""
        if self._metadata is None:
            raise RuntimeError("Call open() before get_metadata()")
        return self._metadata

    async def frames(self) -> AsyncIterator[CapturedFrame]:
        """
        Async iterator yielding frames at target_fps.
        For files: reads all frames, skipping if source FPS > target FPS.
        For streams: reads in real-time.
        """
        if not self._is_open:
            self.open()

        source_fps = self._metadata.fps if self._metadata else 30.0
        frame_skip = max(1, int(source_fps / self.target_fps)) if source_fps > self.target_fps else 1
        frame_interval_s = 1.0 / self.target_fps
        source_idx = 0

        while self._is_open and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                break

            source_idx += 1

            # Skip frames to match target FPS
            if source_idx % frame_skip != 0:
                continue

            self._frame_count += 1
            capture_time = time.monotonic_ns()

            # Resize if requested
            if self.resize_width and self.resize_height:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            h, w = frame.shape[:2]
            timestamp_ms = int((self._frame_count / self.target_fps) * 1000)

            captured = CapturedFrame(
                frame=frame,
                frame_number=self._frame_count,
                timestamp_ms=timestamp_ms,
                capture_time_ns=capture_time,
                width=w,
                height=h,
            )

            yield captured

            # For live sources, pace to real-time
            if self._source_type in (CaptureSourceType.CAMERA, CaptureSourceType.RTSP, CaptureSourceType.RTMP):
                await asyncio.sleep(frame_interval_s)
            else:
                # For files, yield control but don't sleep (process as fast as possible)
                await asyncio.sleep(0)

        logger.info("Capture complete: %d frames processed", self._frame_count)

    def read_frame(self) -> Optional[CapturedFrame]:
        """Read a single frame synchronously. Returns None at end of video."""
        if not self._is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_count += 1
        capture_time = time.monotonic_ns()

        if self.resize_width and self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        h, w = frame.shape[:2]
        timestamp_ms = int((self._frame_count / self.target_fps) * 1000)

        return CapturedFrame(
            frame=frame,
            frame_number=self._frame_count,
            timestamp_ms=timestamp_ms,
            capture_time_ns=capture_time,
            width=w,
            height=h,
        )

    def seek_frame(self, frame_number: int) -> bool:
        """Seek to a specific frame (file sources only)."""
        if self._cap is None or self._source_type != CaptureSourceType.FILE:
            return False
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True

    def seek_ms(self, timestamp_ms: int) -> bool:
        """Seek to a specific timestamp in milliseconds."""
        if self._cap is None or self._source_type != CaptureSourceType.FILE:
            return False
        self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        return True

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self._is_open

    def release(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Video capture released after %d frames", self._frame_count)

    def __del__(self):
        self.release()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _extract_metadata(self) -> VideoMetadata:
        cap = self._cap
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
            source_type=self._source_type,
            source_path=str(self.source),
        )

    @staticmethod
    def _detect_source_type(source: Union[str, int]) -> CaptureSourceType:
        if isinstance(source, int):
            return CaptureSourceType.CAMERA
        s = str(source).lower()
        if s.startswith("rtsp://"):
            return CaptureSourceType.RTSP
        if s.startswith("rtmp://"):
            return CaptureSourceType.RTMP
        return CaptureSourceType.FILE
