"""
Inference Pipeline — Orchestrated per-frame ML inference.
Coordinates BallNet, PlayerNet, CourtNet, and ShotNet in production.
Supports both pre-computed detections and raw frame analysis.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tennis.ml.ball_tracker import BallTracker
from tennis.ml.player_detector import PlayerDetector
from tennis.ml.court_detector import CourtDetector
from tennis.ml.frame_analyzer import FrameAnalyzer, FrameDetections
from tennis.models.events import BallEvent, BoundingBox, PlayerEvent


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_number: int
    ball_event: Optional[BallEvent] = None
    player_events: list[PlayerEvent] = field(default_factory=list)
    court_calibrated: bool = False
    latency_ms: float = 0.0


class InferencePipeline:
    """
    Orchestrated per-frame ML inference pipeline.
    
    In production (iOS), each model runs on CoreML/ANE.
    This Python version provides the same interface for cloud processing.
    """

    def __init__(self, session_id: str = "", fps: float = 30.0, models_dir: str = "./models"):
        self.session_id = session_id
        self.fps = fps
        self.ball_tracker = BallTracker(fps=fps)
        self.player_detector = PlayerDetector(max_players=4)
        self.court_detector = CourtDetector()
        self.frame_analyzer: Optional[FrameAnalyzer] = None
        self.models_dir = models_dir
        self.frame_count = 0
        self.is_initialized = False
        self._total_latency_ms = 0.0

    def initialize(self):
        """Load models and prepare for inference."""
        self.frame_analyzer = FrameAnalyzer(models_dir=self.models_dir)
        self.frame_analyzer.warmup()
        self.is_initialized = True

    def process_frame(
        self,
        ball_detection: Optional[BoundingBox] = None,
        player_detections: Optional[list[BoundingBox]] = None,
        player_keypoints: Optional[list[list[tuple[float, float, float]]]] = None,
        court_keypoints: Optional[list[tuple[float, float, float]]] = None,
    ) -> FrameResult:
        """
        Process a single frame through all models.
        
        In production, raw frame pixels are passed and models run inference.
        Here we accept pre-computed detections for the pipeline orchestration.
        """
        self.frame_count += 1
        result = FrameResult(frame_number=self.frame_count)

        # 1. Court detection (runs every N frames for recalibration)
        if court_keypoints:
            self.court_detector.process_frame(court_keypoints, self.frame_count)
        result.court_calibrated = self.court_detector.is_calibrated

        # 2. Ball tracking
        if ball_detection is not None:
            ball_event = self.ball_tracker.process_frame(
                ball_detection, self.frame_count, self.session_id
            )
            if ball_event and self.court_detector.is_calibrated and ball_event.position_image:
                cx = (ball_event.position_image.x1 + ball_event.position_image.x2) / 2
                cy = (ball_event.position_image.y1 + ball_event.position_image.y2) / 2
                ball_event.position_court = self.court_detector.image_to_court(cx, cy)
            result.ball_event = ball_event

        # 3. Player detection + pose
        if player_detections:
            player_events = self.player_detector.process_frame(
                player_detections,
                keypoints_list=player_keypoints,
                frame_number=self.frame_count,
                session_id=self.session_id,
                fps=self.fps,
            )
            for pe in player_events:
                if self.court_detector.is_calibrated and pe.position_image:
                    cx = (pe.position_image.x1 + pe.position_image.x2) / 2
                    cy = pe.position_image.y2  # feet position
                    pe.position_court = self.court_detector.image_to_court(cx, cy)
            result.player_events = player_events

        return result

    def process_raw_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int = 0,
    ) -> FrameResult:
        """
        Process a raw video frame through the full pipeline.

        Runs FrameAnalyzer to extract detections, then passes them
        through the existing tracking and scoring logic.

        Args:
            frame: BGR numpy array (H, W, 3)
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            FrameResult with all tracking results
        """
        if not self.is_initialized:
            self.initialize()

        start = time.perf_counter()

        # Run ML inference / heuristic detection
        detections: FrameDetections = self.frame_analyzer.analyze_frame(
            frame,
            frame_number=self.frame_count + 1,
            timestamp_ms=timestamp_ms,
        )

        # Feed detections into existing pipeline
        result = self.process_frame(
            ball_detection=detections.ball_bbox,
            player_detections=detections.player_bboxes if detections.player_bboxes else None,
            court_keypoints=detections.court_keypoints if detections.court_keypoints else None,
        )

        result.latency_ms = (time.perf_counter() - start) * 1000
        self._total_latency_ms += result.latency_ms

        return result

    def reset(self):
        self.ball_tracker.reset()
        self.player_detector.reset()
        self.court_detector.reset()
        self.frame_count = 0
        self._total_latency_ms = 0.0

    def get_stats(self) -> dict:
        avg_latency = (
            self._total_latency_ms / self.frame_count
            if self.frame_count > 0 else 0.0
        )
        return {
            "frames_processed": self.frame_count,
            "active_players": self.player_detector.get_active_players(),
            "court_calibrated": self.court_detector.is_calibrated,
            "ball_tracking": self.ball_tracker.kalman.is_tracking,
            "avg_latency_ms": round(avg_latency, 2),
        }
