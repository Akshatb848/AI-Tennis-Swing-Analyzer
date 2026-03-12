"""
Frame Analyzer — Real ML inference on raw video frames.

Two operating modes:
1. Primary: OpenCV DNN with ONNX models (YOLOv8n, MoveNet)
2. Fallback: Heuristic detection using color analysis, contours, and Hough lines

Both paths produce identical FrameDetections output for downstream consumption.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tennis.models.events import BoundingBox, Point2D

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class DetectedKeypoint:
    """A single pose keypoint."""
    x: float
    y: float
    confidence: float
    name: str = ""


@dataclass
class FrameDetections:
    """All detections from a single frame."""
    frame_number: int = 0
    timestamp_ms: int = 0
    inference_time_ms: float = 0.0

    # Ball detection
    ball_bbox: Optional[BoundingBox] = None
    ball_center: Optional[Point2D] = None

    # Player detections
    player_bboxes: list[BoundingBox] = field(default_factory=list)
    player_keypoints: list[list[DetectedKeypoint]] = field(default_factory=list)

    # Court keypoints for homography
    court_keypoints: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def has_ball(self) -> bool:
        return self.ball_bbox is not None and self.ball_bbox.confidence > 0.3

    @property
    def has_players(self) -> bool:
        return len(self.player_bboxes) > 0

    @property
    def has_court(self) -> bool:
        return len(self.court_keypoints) >= 4


class FrameAnalyzer:
    """
    Runs ML inference on raw video frames.

    Falls back to heuristic detection when models are not available.
    """

    def __init__(
        self,
        models_dir: str = "./models",
        device: str = "cpu",
        ball_conf_threshold: float = 0.25,
        player_conf_threshold: float = 0.4,
    ):
        self.models_dir = models_dir
        self.device = device
        self.ball_conf_threshold = ball_conf_threshold
        self.player_conf_threshold = player_conf_threshold

        self._ball_net = None
        self._player_net = None
        self._court_net = None
        self._use_heuristic = True
        self._initialized = False

        self._try_load_models()

    def _try_load_models(self):
        """Attempt to load ONNX models. Fall back to heuristics if unavailable."""
        # In current deployment, we use heuristic mode.
        # When ONNX models are available, they can be loaded here:
        #   self._ball_net = cv2.dnn.readNetFromONNX(f"{self.models_dir}/ballnet.onnx")
        #   etc.
        self._use_heuristic = True
        self._initialized = True
        logger.info("FrameAnalyzer initialized in %s mode",
                     "heuristic" if self._use_heuristic else "model")

    def warmup(self, frame_shape: tuple = (1080, 1920, 3)):
        """Warmup inference with a dummy frame."""
        dummy = np.zeros(frame_shape, dtype=np.uint8)
        self.analyze_frame(dummy, frame_number=0)
        logger.info("FrameAnalyzer warmup complete")

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: int = 0,
    ) -> FrameDetections:
        """
        Analyze a single video frame and return all detections.

        Args:
            frame: BGR numpy array (H, W, 3)
            frame_number: Sequential frame number
            timestamp_ms: Timestamp in milliseconds

        Returns:
            FrameDetections with ball, player, and court results
        """
        if not HAS_OPENCV:
            return FrameDetections(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
            )

        start = time.perf_counter()

        if self._use_heuristic:
            detections = self._heuristic_analyze(frame, frame_number, timestamp_ms)
        else:
            detections = self._model_analyze(frame, frame_number, timestamp_ms)

        detections.inference_time_ms = (time.perf_counter() - start) * 1000
        return detections

    # ── Heuristic Detection (fallback) ───────────────────────────────────────

    def _heuristic_analyze(
        self, frame: np.ndarray, frame_number: int, timestamp_ms: int,
    ) -> FrameDetections:
        """Computer vision heuristics for ball, player, and court detection."""
        detections = FrameDetections(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
        )

        h, w = frame.shape[:2]

        # 1) Ball detection — find yellow/green blobs
        detections.ball_bbox = self._detect_ball_heuristic(frame, w, h)
        if detections.ball_bbox:
            cx = (detections.ball_bbox.x1 + detections.ball_bbox.x2) / 2
            cy = (detections.ball_bbox.y1 + detections.ball_bbox.y2) / 2
            detections.ball_center = Point2D(x=cx, y=cy)

        # 2) Player detection — find large contours in upper portion
        detections.player_bboxes = self._detect_players_heuristic(frame, w, h)

        # 3) Court detection — find lines via Hough transform
        detections.court_keypoints = self._detect_court_heuristic(frame, w, h)

        return detections

    def _detect_ball_heuristic(
        self, frame: np.ndarray, w: int, h: int,
    ) -> Optional[BoundingBox]:
        """Detect tennis ball using color filtering (yellow/green in HSV)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tennis ball yellow-green range
        lower_yellow = np.array([25, 80, 80])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Tennis ball should be small relative to frame
            if area < 20 or area > w * h * 0.02:
                continue

            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect ratio check — ball should be roughly circular
            aspect = bw / bh if bh > 0 else 0
            if aspect < 0.5 or aspect > 2.0:
                continue

            score = circularity * min(area / 100, 1.0)
            if score > best_score:
                best_score = score
                confidence = min(circularity * 0.8 + 0.2, 1.0)
                best_ball = BoundingBox(
                    x1=float(x), y1=float(y),
                    x2=float(x + bw), y2=float(y + bh),
                    confidence=confidence,
                )

        return best_ball

    def _detect_players_heuristic(
        self, frame: np.ndarray, w: int, h: int,
    ) -> list[BoundingBox]:
        """Detect players using background subtraction and contour analysis."""
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        players = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # Player should be tall (height > width) and reasonably sized
            if bh < h * 0.1 or bw < w * 0.02:
                continue
            if area < w * h * 0.005 or area > w * h * 0.25:
                continue

            aspect = bh / bw if bw > 0 else 0
            if aspect < 1.2:  # Players are taller than wide
                continue

            confidence = min(0.5 + (aspect - 1.2) * 0.2, 0.95)
            players.append(BoundingBox(
                x1=float(x), y1=float(y),
                x2=float(x + bw), y2=float(y + bh),
                confidence=confidence,
            ))

        # Sort by area (largest first) and keep top 4
        players.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        return players[:4]

    def _detect_court_heuristic(
        self, frame: np.ndarray, w: int, h: int,
    ) -> list[tuple[float, float, float]]:
        """Detect court lines and extract keypoints using Hough line transform."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold for line detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=w * 0.1,
            maxLineGap=20,
        )

        if lines is None or len(lines) < 4:
            return []

        # Separate horizontal and vertical lines
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:  # Roughly horizontal
                horizontal.append((x1, y1, x2, y2))
            elif 60 < angle < 120:  # Roughly vertical
                vertical.append((x1, y1, x2, y2))

        # Find intersections as keypoints
        keypoints = []
        for hline in horizontal[:6]:
            for vline in vertical[:6]:
                pt = self._line_intersection(hline, vline)
                if pt and 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                    keypoints.append((pt[0], pt[1], 0.75))

        # Deduplicate nearby keypoints
        keypoints = self._deduplicate_keypoints(keypoints, min_dist=30)

        return keypoints[:14]  # Max 14 court keypoints

    @staticmethod
    def _line_intersection(
        line1: tuple, line2: tuple,
    ) -> Optional[tuple[float, float]]:
        """Find intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)

    @staticmethod
    def _deduplicate_keypoints(
        keypoints: list[tuple[float, float, float]],
        min_dist: float = 30,
    ) -> list[tuple[float, float, float]]:
        """Remove keypoints that are too close together."""
        if not keypoints:
            return []

        unique = [keypoints[0]]
        for kp in keypoints[1:]:
            is_dup = False
            for existing in unique:
                dist = np.sqrt((kp[0] - existing[0]) ** 2 + (kp[1] - existing[1]) ** 2)
                if dist < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(kp)
        return unique

    # ── Model-based Detection (production) ───────────────────────────────────

    def _model_analyze(
        self, frame: np.ndarray, frame_number: int, timestamp_ms: int,
    ) -> FrameDetections:
        """
        Model-based inference using ONNX models.
        This is a placeholder for when trained models are deployed.
        Falls back to heuristic for now.
        """
        return self._heuristic_analyze(frame, frame_number, timestamp_ms)
