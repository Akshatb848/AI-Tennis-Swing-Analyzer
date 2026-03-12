"""
Frame Buffer — Thread-safe ring buffer for replay windows and post-match analysis.

Provides:
- Ring buffer for recent frames (line call replays need ±2s)
- Deterministic frame numbering
- Memory-efficient: drops oldest frames when full
- Session-level frame event storage
"""

from __future__ import annotations

import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    """Lightweight frame event record (no pixel data)."""
    frame_number: int
    timestamp_ms: int
    ball_detected: bool = False
    ball_confidence: float = 0.0
    player_count: int = 0
    court_calibrated: bool = False
    processing_latency_ms: float = 0.0


class FrameBuffer:
    """
    Thread-safe ring buffer for video frames.

    Keeps the last `max_frames` in memory for replay access.
    Stores lightweight FrameEvents for the full session timeline.
    """

    def __init__(self, max_frames: int = 120, fps: float = 30.0):
        self.max_frames = max_frames
        self.fps = fps
        self._frames: deque[tuple[int, np.ndarray]] = deque(maxlen=max_frames)
        self._events: list[FrameEvent] = []
        self._lock = threading.Lock()
        self._total_frames = 0

    def add_frame(self, frame_number: int, frame: np.ndarray, event: Optional[FrameEvent] = None):
        """Add a frame to the buffer."""
        with self._lock:
            self._frames.append((frame_number, frame))
            self._total_frames += 1
            if event:
                self._events.append(event)

    def add_event(self, event: FrameEvent):
        """Add a frame event (no pixel data)."""
        with self._lock:
            self._events.append(event)

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame by number. Returns None if not in buffer."""
        with self._lock:
            for fn, frame in self._frames:
                if fn == frame_number:
                    return frame.copy()
        return None

    def get_frame_range(self, start_frame: int, end_frame: int) -> list[tuple[int, np.ndarray]]:
        """Get frames in a range. Returns available frames (may be partial)."""
        with self._lock:
            return [
                (fn, frame.copy())
                for fn, frame in self._frames
                if start_frame <= fn <= end_frame
            ]

    def get_replay_window(self, center_frame: int, window_seconds: float = 2.0) -> list[tuple[int, np.ndarray]]:
        """Get frames around a center frame for replay (e.g., line call review)."""
        half_window = int(window_seconds * self.fps)
        return self.get_frame_range(
            center_frame - half_window,
            center_frame + half_window,
        )

    def get_latest_frame(self) -> Optional[tuple[int, np.ndarray]]:
        """Get the most recent frame."""
        with self._lock:
            if self._frames:
                fn, frame = self._frames[-1]
                return fn, frame.copy()
        return None

    def get_events(self, start_frame: int = 0, end_frame: int = -1) -> list[FrameEvent]:
        """Get frame events in a range."""
        with self._lock:
            if end_frame < 0:
                return [e for e in self._events if e.frame_number >= start_frame]
            return [
                e for e in self._events
                if start_frame <= e.frame_number <= end_frame
            ]

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def buffered_frames(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def buffer_duration_seconds(self) -> float:
        """How many seconds of video are currently in the buffer."""
        with self._lock:
            if not self._frames:
                return 0.0
            first_fn = self._frames[0][0]
            last_fn = self._frames[-1][0]
            return (last_fn - first_fn) / self.fps

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._frames.clear()
            self._events.clear()
            self._total_frames = 0

    def get_session_summary(self) -> dict:
        """Get summary statistics for the buffered session."""
        with self._lock:
            if not self._events:
                return {
                    "total_frames": self._total_frames,
                    "buffered_frames": len(self._frames),
                    "events_recorded": 0,
                }

            ball_detections = sum(1 for e in self._events if e.ball_detected)
            avg_latency = (
                sum(e.processing_latency_ms for e in self._events) / len(self._events)
                if self._events else 0
            )

            return {
                "total_frames": self._total_frames,
                "buffered_frames": len(self._frames),
                "events_recorded": len(self._events),
                "ball_detection_rate": ball_detections / len(self._events) if self._events else 0,
                "avg_processing_latency_ms": round(avg_latency, 2),
                "buffer_duration_seconds": self.buffer_duration_seconds,
            }
