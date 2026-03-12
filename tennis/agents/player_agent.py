"""
Player Tracking Agent — detects players, tracks their positions, posture, and movement patterns.

Uses contour analysis and background subtraction to find players,
then analyzes their movement across the court.
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


class PlayerAgent:
    """Expert agent for player detection and movement tracking."""

    name = "player"

    def analyze(self, frames: list, context: dict) -> dict:
        """
        Track players across frames and analyze movement patterns.

        Returns:
            player_count: int
            player_tracks: list of per-player position sequences
            court_coverage: dict per player (% of court zones visited)
            movement_intensity: float (0-1)
            avg_rally_speed: float (sprint speed estimate km/h)
            positioning: list of {frame_idx, players: [{x,y}]}
        """
        match_type = context.get("match_type", "singles")
        expected = 4 if match_type == "doubles" else 2

        if not frames or not HAS_CV2:
            return self._fallback(expected, context)

        # Use MOG2 background subtractor for motion-based player detection
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=25, detectShadows=False
        )

        positioning = []
        all_player_positions = []  # (frame_idx, [(x,y)])

        for fidx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            fg_mask = bg_sub.apply(frame)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # Find contours (players are large moving blobs)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = (h * w) * 0.003   # At least 0.3% of frame
            max_area = (h * w) * 0.20    # At most 20%
            player_blobs = sorted(
                [c for c in contours if min_area < cv2.contourArea(c) < max_area],
                key=cv2.contourArea,
                reverse=True
            )[:expected]

            players_in_frame = []
            for c in player_blobs:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] / w
                    cy = M["m01"] / M["m00"] / h
                    x1, y1, bw, bh = cv2.boundingRect(c)
                    players_in_frame.append({
                        "x": round(cx, 3),
                        "y": round(cy, 3),
                        "bbox": [round(x1/w, 3), round(y1/h, 3),
                                 round((x1+bw)/w, 3), round((y1+bh)/h, 3)],
                    })

            if players_in_frame:
                positioning.append({"frame_idx": fidx, "players": players_in_frame})
                all_player_positions.append((fidx, players_in_frame))

        # Estimate player count
        if all_player_positions:
            avg_players = sum(len(p) for _, p in all_player_positions) / len(all_player_positions)
            player_count = min(4, max(1, round(avg_players)))
        else:
            player_count = expected

        # Court coverage: divide court into a 3x3 grid and check which zones each player visits
        zone_counts = [{} for _ in range(player_count)]
        fps = context.get("fps", 30.0)
        speeds = []

        for i, (_, plist) in enumerate(all_player_positions):
            for pi, pos in enumerate(plist[:player_count]):
                if pi >= player_count:
                    break
                zx = int(pos["x"] * 3)
                zy = int(pos["y"] * 3)
                zone_key = f"{min(zx,2)},{min(zy,2)}"
                zone_counts[pi][zone_key] = zone_counts[pi].get(zone_key, 0) + 1

        # Speed estimation from position changes
        if len(all_player_positions) > 1:
            for i in range(1, len(all_player_positions)):
                f1, p1_list = all_player_positions[i - 1]
                f2, p2_list = all_player_positions[i]
                fdiff = f2 - f1
                if fdiff <= 0 or fdiff > 5:
                    continue
                for pi in range(min(len(p1_list), len(p2_list))):
                    dx = p2_list[pi]["x"] - p1_list[pi]["x"]
                    dy = p2_list[pi]["y"] - p1_list[pi]["y"]
                    dist = math.sqrt(dx**2 + dy**2)
                    # Court is ~23.77m wide, player x covers ~80% = ~19m across width
                    m_dist = dist * 19.0
                    t = fdiff / fps
                    if t > 0:
                        speed_ms = m_dist / t
                        speed_kmh = speed_ms * 3.6
                        if 0.5 < speed_kmh < 35:  # realistic player speed
                            speeds.append(speed_kmh)

        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 5.5
        total_zones = 9
        coverage = [
            round(len(z) / total_zones, 2) for z in zone_counts
        ]

        return {
            "player_count": player_count,
            "positioning": positioning[:100],
            "court_coverage": coverage,
            "movement_intensity": round(min(1.0, avg_speed / 15.0), 3),
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": round(max(speeds) if speeds else avg_speed * 2.5, 1),
            "frames_analyzed": len(frames),
        }

    def _fallback(self, expected: int, context: dict) -> dict:
        return {
            "player_count": expected,
            "positioning": [],
            "court_coverage": [0.5] * expected,
            "movement_intensity": 0.4,
            "avg_speed_kmh": 5.5,
            "max_speed_kmh": 12.0,
            "frames_analyzed": 0,
        }
