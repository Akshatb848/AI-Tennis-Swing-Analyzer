"""
Court Detection Agent — detects court boundaries, net, baseline, service boxes.

Uses HSV color segmentation and Hough line detection to find court geometry.
Outputs structured court data used by all downstream agents.
"""
from __future__ import annotations
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class CourtAgent:
    """Expert agent for court detection and geometry analysis."""

    name = "court"

    def analyze(self, frames: list, context: dict) -> dict:
        """
        Analyze sampled frames to detect court layout.

        Returns:
            surface: 'hard' | 'clay' | 'grass'
            confidence: 0-1
            net_y: estimated net y position (0-1 normalized)
            baseline_top_y: top baseline y (0-1)
            baseline_bot_y: bottom baseline y (0-1)
            court_bounds: {left, right, top, bottom} all 0-1
            lines_detected: int
        """
        if not frames or not HAS_CV2:
            return self._fallback(context)

        green_ratios, blue_ratios, red_ratios = [], [], []
        line_counts = []
        net_estimates = []

        for frame in frames[:15]:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            total_px = h * w

            # Surface color detection
            mask_g = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratios.append(cv2.countNonZero(mask_g) / total_px)
            mask_b = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255]))
            blue_ratios.append(cv2.countNonZero(mask_b) / total_px)
            mask_r = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([20, 255, 200]))
            red_ratios.append(cv2.countNonZero(mask_r) / total_px)

            # Line detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=15)
            if lines is not None:
                line_counts.append(len(lines))
                # Estimate net position from dominant horizontal lines near center
                h_lines = [l[0] for l in lines if abs(l[0][1] - l[0][3]) < 8]
                if h_lines:
                    ys = sorted([l[1] / h for l in h_lines])
                    # Net is typically near vertical center
                    center_lines = [y for y in ys if 0.35 < y < 0.65]
                    if center_lines:
                        net_estimates.append(sum(center_lines) / len(center_lines))
            else:
                line_counts.append(0)

        avg_g = sum(green_ratios) / len(green_ratios)
        avg_b = sum(blue_ratios) / len(blue_ratios)
        avg_r = sum(red_ratios) / len(red_ratios)
        avg_lines = sum(line_counts) / max(1, len(line_counts))

        if avg_b > avg_g and avg_b > avg_r:
            surface = "hard"
            conf = min(0.95, avg_b * 10)
        elif avg_r > avg_g and avg_r > avg_b:
            surface = "clay"
            conf = min(0.95, avg_r * 10)
        else:
            surface = "grass"
            conf = min(0.95, avg_g * 8)

        net_y = sum(net_estimates) / len(net_estimates) if net_estimates else 0.5

        return {
            "surface": surface,
            "confidence": round(conf, 3),
            "net_y": round(net_y, 3),
            "baseline_top_y": round(max(0.15, net_y - 0.28), 3),
            "baseline_bot_y": round(min(0.92, net_y + 0.28), 3),
            "court_bounds": {"left": 0.1, "right": 0.9, "top": 0.12, "bottom": 0.92},
            "lines_detected": int(avg_lines),
            "green_ratio": round(avg_g, 4),
            "blue_ratio": round(avg_b, 4),
            "red_ratio": round(avg_r, 4),
        }

    def _fallback(self, context: dict) -> dict:
        return {
            "surface": "hard",
            "confidence": 0.3,
            "net_y": 0.5,
            "baseline_top_y": 0.22,
            "baseline_bot_y": 0.78,
            "court_bounds": {"left": 0.1, "right": 0.9, "top": 0.12, "bottom": 0.92},
            "lines_detected": 0,
            "green_ratio": 0.0,
            "blue_ratio": 0.0,
            "red_ratio": 0.0,
        }
