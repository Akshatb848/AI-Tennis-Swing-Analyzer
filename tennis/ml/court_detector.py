"""
Court Detector — Court line detection and homography estimation.
Wraps CourtNet model with RANSAC homography and continuous recalibration.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

import numpy as np

from tennis.models.events import Point2D

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# 14 court keypoints in normalized court coordinates (0-1)
COURT_KEYPOINT_NAMES = [
    "bl_left", "bl_right", "bl_far_left", "bl_far_right",
    "sl_near_left", "sl_near_right", "sl_far_left", "sl_far_right",
    "sv_near_left", "sv_near_right", "sv_far_left", "sv_far_right",
    "center_near", "center_far",
]


class HomographyMatrix:
    """3x3 homography matrix for image → court coordinate transform."""
    def __init__(self, matrix: Optional[list[list[float]]] = None):
        self.matrix = matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.is_valid = matrix is not None
        self.reprojection_error = 0.0

    def transform_point(self, x: float, y: float) -> Point2D:
        """Transform image point to court coordinates."""
        m = self.matrix
        denom = m[2][0] * x + m[2][1] * y + m[2][2]
        if abs(denom) < 1e-8:
            return Point2D(x=0, y=0)
        cx = (m[0][0] * x + m[0][1] * y + m[0][2]) / denom
        cy = (m[1][0] * x + m[1][1] * y + m[1][2]) / denom
        return Point2D(x=cx, y=cy)


class CourtDetector:
    """Court line detection + homography estimation with EMA smoothing."""

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.homography = HomographyMatrix()
        self.keypoints: list[tuple[float, float]] = []
        self._smoothed_keypoints: list[tuple[float, float]] = []
        self.calibration_confidence = 0.0
        self.frame_count = 0

    def process_frame(
        self, detected_keypoints: list[tuple[float, float, float]], frame_number: int = 0,
    ) -> HomographyMatrix:
        """Process frame with detected court keypoints. Returns updated homography."""
        self.frame_count = frame_number
        valid = [(x, y, c) for x, y, c in detected_keypoints if c > 0.5]
        if len(valid) < 4:
            return self.homography

        raw = [(x, y) for x, y, _ in valid]
        # EMA smooth keypoints
        if not self._smoothed_keypoints:
            self._smoothed_keypoints = raw
        else:
            smoothed = []
            for i, (nx, ny) in enumerate(raw):
                if i < len(self._smoothed_keypoints):
                    ox, oy = self._smoothed_keypoints[i]
                    sx = self.ema_alpha * nx + (1 - self.ema_alpha) * ox
                    sy = self.ema_alpha * ny + (1 - self.ema_alpha) * oy
                    smoothed.append((sx, sy))
                else:
                    smoothed.append((nx, ny))
            self._smoothed_keypoints = smoothed

        self.keypoints = self._smoothed_keypoints
        self.homography = self._estimate_homography(self._smoothed_keypoints)
        self.calibration_confidence = sum(c for _, _, c in valid) / len(valid)
        return self.homography

    def _estimate_homography(self, image_pts: list[tuple[float, float]]) -> HomographyMatrix:
        """Estimate homography from image keypoints to court coords.

        Uses cv2.findHomography with RANSAC when OpenCV is available,
        falls back to simplified affine approximation otherwise.
        """
        if len(image_pts) < 4:
            return self.homography

        # Standard court corners in meters (singles court)
        # Origin at center, width = 8.23m, length = 23.77m
        court_pts_ref = [
            (-4.115, -11.885), (4.115, -11.885),
            (-4.115, 11.885), (4.115, 11.885),
        ]
        n_pts = min(len(image_pts), len(court_pts_ref))

        if HAS_OPENCV and n_pts >= 4:
            # Use real RANSAC homography
            src = np.array(image_pts[:n_pts], dtype=np.float64)
            dst = np.array(court_pts_ref[:n_pts], dtype=np.float64)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                hm = HomographyMatrix(H.tolist())
                hm.is_valid = True
                # Calculate reprojection error
                if mask is not None:
                    inliers = mask.ravel().astype(bool)
                    if inliers.any():
                        src_in = src[inliers]
                        dst_in = dst[inliers]
                        projected = cv2.perspectiveTransform(
                            src_in.reshape(-1, 1, 2), H,
                        ).reshape(-1, 2)
                        hm.reprojection_error = float(
                            np.mean(np.linalg.norm(projected - dst_in, axis=1))
                        )
                logger.debug(
                    "Homography estimated via RANSAC, reproj error=%.3f",
                    hm.reprojection_error,
                )
                return hm

        # Fallback: simplified affine approximation from first 4 points
        ix, iy = image_pts[0]
        cx, cy = court_pts_ref[0]
        sx = (court_pts_ref[1][0] - court_pts_ref[0][0]) / max(image_pts[1][0] - image_pts[0][0], 1)
        sy = (court_pts_ref[2][1] - court_pts_ref[0][1]) / max(image_pts[2][1] - image_pts[0][1], 1)
        h = [[sx, 0, cx - sx * ix], [0, sy, cy - sy * iy], [0, 0, 1]]

        hm = HomographyMatrix(h)
        hm.is_valid = True
        return hm

    def image_to_court(self, x: float, y: float) -> Point2D:
        """Convert image coordinates to court coordinates."""
        return self.homography.transform_point(x, y)

    @property
    def is_calibrated(self) -> bool:
        return self.homography.is_valid and self.calibration_confidence > 0.6

    def reset(self):
        self.homography = HomographyMatrix()
        self.keypoints.clear()
        self._smoothed_keypoints.clear()
        self.calibration_confidence = 0.0
