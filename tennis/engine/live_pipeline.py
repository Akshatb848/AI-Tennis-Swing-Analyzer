"""
Live Pipeline — End-to-end real-time processing orchestrator.

Binds together:
- VideoCapture (frame ingest)
- FrameAnalyzer / InferencePipeline (ML inference + tracking)
- RecordingSession (rally/point/scoring state)
- FrameBuffer (replay access)

Usage:
    pipeline = LivePipeline(config)
    result = await pipeline.process_video("match.mp4")
    # or frame-by-frame:
    pipeline.start(config)
    for each frame:
        result = pipeline.process_frame(frame_data)
    summary = pipeline.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tennis.engine.recording import RecordingSession, MatchSetupConfig, RecordingState
from tennis.ml.inference_pipeline import InferencePipeline, FrameResult
from tennis.video.capture import VideoCapture, CapturedFrame, VideoMetadata
from tennis.video.frame_buffer import FrameBuffer, FrameEvent

logger = logging.getLogger(__name__)


@dataclass
class FrameProcessingResult:
    """Result of processing a single frame through the live pipeline."""
    frame_number: int = 0
    timestamp_ms: int = 0
    ball_detected: bool = False
    ball_confidence: float = 0.0
    player_count: int = 0
    court_calibrated: bool = False
    processing_latency_ms: float = 0.0
    rally_active: bool = False
    current_score: str = ""
    line_call: Optional[dict] = None


@dataclass
class SessionResult:
    """Full result of processing a video/session."""
    session_id: str = ""
    total_frames: int = 0
    duration_seconds: float = 0.0
    points_detected: int = 0
    line_calls: int = 0
    avg_latency_ms: float = 0.0
    video_metadata: Optional[VideoMetadata] = None
    summary: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class LivePipeline:
    """
    End-to-end live processing: video → ML → events → scoring.

    Orchestrates the full analysis pipeline for real-time or file-based
    tennis match processing.
    """

    def __init__(
        self,
        config: Optional[MatchSetupConfig] = None,
        target_fps: float = 30.0,
        buffer_seconds: float = 4.0,
        models_dir: str = "./models",
    ):
        self.config = config
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        self.models_dir = models_dir

        # Components (created on start)
        self._inference: Optional[InferencePipeline] = None
        self._recording: Optional[RecordingSession] = None
        self._capture: Optional[VideoCapture] = None
        self._buffer: Optional[FrameBuffer] = None

        self._is_running = False
        self._total_frames = 0
        self._total_latency_ms = 0.0
        self._errors: list[str] = []

    def start(self, config: Optional[MatchSetupConfig] = None):
        """Initialize all pipeline components for a new session."""
        if config:
            self.config = config

        # Create inference pipeline
        self._inference = InferencePipeline(
            session_id="live",
            fps=self.target_fps,
            models_dir=self.models_dir,
        )
        self._inference.initialize()

        # Create recording session
        self._recording = RecordingSession()
        if self.config:
            self._recording.setup(self.config)

        # Create frame buffer
        buffer_frames = int(self.buffer_seconds * self.target_fps)
        self._buffer = FrameBuffer(
            max_frames=buffer_frames,
            fps=self.target_fps,
        )

        self._is_running = True
        self._total_frames = 0
        self._total_latency_ms = 0.0
        self._errors = []

        logger.info("LivePipeline started with config: %s", self.config)

    async def process_video(self, source: str) -> SessionResult:
        """
        Process an entire video file through the pipeline.

        Args:
            source: Path to video file

        Returns:
            Complete session result with analysis summary
        """
        self.start()

        # Open video
        self._capture = VideoCapture(
            source=source,
            target_fps=self.target_fps,
        )
        metadata = self._capture.open()

        # Start recording
        if self._recording and self._recording.state == RecordingState.READY:
            self._recording.start_recording()

        logger.info(
            "Processing video: %s (%dx%d @ %.1f fps, %.1fs)",
            source, metadata.width, metadata.height,
            metadata.fps, metadata.duration_seconds,
        )

        # Process all frames
        async for captured_frame in self._capture.frames():
            try:
                self.process_frame(captured_frame.frame, captured_frame.timestamp_ms)
            except Exception as e:
                self._errors.append(f"Frame {captured_frame.frame_number}: {str(e)}")
                logger.error("Error processing frame %d: %s",
                             captured_frame.frame_number, e)

        # Finalize
        self._capture.release()
        return self.stop(metadata)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int = 0,
    ) -> FrameProcessingResult:
        """
        Process a single raw frame through the full pipeline.

        This is the core method for both real-time and file-based processing.
        """
        if not self._is_running or self._inference is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

        start_time = time.perf_counter()

        # 1) Run ML inference
        frame_result: FrameResult = self._inference.process_raw_frame(
            frame, timestamp_ms=timestamp_ms,
        )

        # 2) Feed into recording session
        if self._recording and self._recording.state == RecordingState.RECORDING:
            self._recording.process_frame(
                ball_event=frame_result.ball_event,
                player_events=frame_result.player_events,
            )

        # 3) Store in buffer
        self._total_frames += 1
        latency = (time.perf_counter() - start_time) * 1000
        self._total_latency_ms += latency

        if self._buffer:
            event = FrameEvent(
                frame_number=self._total_frames,
                timestamp_ms=timestamp_ms,
                ball_detected=frame_result.ball_event is not None,
                ball_confidence=(
                    frame_result.ball_event.detection_confidence
                    if frame_result.ball_event else 0.0
                ),
                player_count=len(frame_result.player_events),
                court_calibrated=frame_result.court_calibrated,
                processing_latency_ms=latency,
            )
            self._buffer.add_frame(self._total_frames, frame, event)

        # Build result
        result = FrameProcessingResult(
            frame_number=self._total_frames,
            timestamp_ms=timestamp_ms,
            ball_detected=frame_result.ball_event is not None,
            ball_confidence=(
                frame_result.ball_event.detection_confidence
                if frame_result.ball_event else 0.0
            ),
            player_count=len(frame_result.player_events),
            court_calibrated=frame_result.court_calibrated,
            processing_latency_ms=latency,
        )

        return result

    def stop(self, metadata: Optional[VideoMetadata] = None) -> SessionResult:
        """Stop the pipeline and return session results."""
        self._is_running = False

        # Stop recording
        summary = {}
        if self._recording:
            if self._recording.state == RecordingState.RECORDING:
                self._recording.stop_recording()
            summary = self._recording.get_summary()

        # Build session result
        avg_latency = (
            self._total_latency_ms / self._total_frames
            if self._total_frames > 0 else 0.0
        )

        result = SessionResult(
            session_id=self._recording.id if self._recording else "",
            total_frames=self._total_frames,
            duration_seconds=(
                self._total_frames / self.target_fps
                if self.target_fps > 0 else 0.0
            ),
            points_detected=len(self._recording.points) if self._recording else 0,
            line_calls=len(self._recording.line_calls) if self._recording else 0,
            avg_latency_ms=round(avg_latency, 2),
            video_metadata=metadata,
            summary=summary,
            errors=self._errors,
        )

        logger.info(
            "Pipeline stopped: %d frames, %d points, %.1fms avg latency",
            result.total_frames, result.points_detected, result.avg_latency_ms,
        )

        return result

    def get_live_state(self) -> dict:
        """Get current live state for dashboard / overlay consumption."""
        state = {
            "is_running": self._is_running,
            "total_frames": self._total_frames,
            "duration_seconds": (
                self._total_frames / self.target_fps
                if self.target_fps > 0 else 0.0
            ),
        }

        if self._inference:
            state["inference_stats"] = self._inference.get_stats()

        if self._recording:
            state["recording_state"] = self._recording.state.value
            state["points_detected"] = len(self._recording.points)
            state["line_calls"] = len(self._recording.line_calls)

        if self._buffer:
            state["buffer"] = self._buffer.get_session_summary()

        return state

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def frame_count(self) -> int:
        return self._total_frames

    @property
    def recording(self) -> Optional[RecordingSession]:
        return self._recording

    @property
    def buffer(self) -> Optional[FrameBuffer]:
        return self._buffer
