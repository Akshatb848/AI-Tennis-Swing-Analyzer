"""
Shot Classifier — Classifies tennis shot types from ball trajectory,
player pose, and court position data.

Production-grade classification with confidence calibration.
"""

from __future__ import annotations
import math
from typing import Optional
from tennis.models.events import PlayerPose, Point2D
from tennis.models.match import ShotType


class ShotClassifier:
    """Classifies shot types from trajectory, pose, and position analysis.

    Classification priority:
    1. Serve (first shot, or explicitly flagged)
    2. Return (2nd shot in rally)
    3. Smash (high ball + downward trajectory + near net)
    4. Volley (player near net, no bounce)
    5. Lob (high net clearance)
    6. Drop shot (low speed + steep angle)
    7. Slice (negative trajectory angle + moderate speed)
    8. Forehand / Backhand (default, side determined by pose or position)
    """

    VOLLEY_MAX_DIST = 4.0       # meters from net for volley detection
    LOB_MIN_CLEARANCE = 200.0   # cm net clearance for lob
    DROP_MAX_SPEED = 30.0       # mph max for drop shot
    SMASH_MIN_HEIGHT = 3.0      # meters ball height for smash
    SMASH_MAX_NET_DIST = 5.0    # meters from net for smash
    SLICE_MAX_ANGLE = -10.0     # degrees for slice detection
    SLICE_MAX_SPEED = 60.0      # mph max for slice

    def classify(
        self,
        ball_position: Optional[Point2D],
        ball_velocity_mph: float,
        ball_trajectory_angle: float,
        player_pose: Optional[PlayerPose],
        player_position: Optional[Point2D],
        shot_number_in_rally: int,
        is_serve: bool = False,
        net_clearance_cm: Optional[float] = None,
        ball_height_m: Optional[float] = None,
        prev_trajectory_angle: Optional[float] = None,
    ) -> tuple[ShotType, float]:
        """Classify the shot type with confidence score.

        Returns:
            (shot_type, confidence) where confidence is 0.0-1.0
        """
        # ── 1. Serve ─────────────────────────────────────────
        if is_serve:
            return ShotType.SERVE, 0.95

        # ── 2. Return (2nd shot) ─────────────────────────────
        if shot_number_in_rally <= 2:
            side = self._determine_side(player_pose, player_position, ball_position)
            shot = ShotType.RETURN_FH if side == "fh" else ShotType.RETURN_BH
            return shot, 0.80

        # ── 3. Smash / Overhead ──────────────────────────────
        if self._is_smash(ball_height_m, ball_trajectory_angle, player_position):
            return ShotType.OVERHEAD, 0.85

        # ── 4. Volley (player near net) ──────────────────────
        if player_position and abs(player_position.y) < self.VOLLEY_MAX_DIST:
            side = self._determine_side(player_pose, player_position, ball_position)
            shot = ShotType.VOLLEY_FH if side == "fh" else ShotType.VOLLEY_BH
            conf = 0.80 if player_pose else 0.70
            return shot, conf

        # ── 5. Lob ───────────────────────────────────────────
        if net_clearance_cm and net_clearance_cm > self.LOB_MIN_CLEARANCE:
            return ShotType.LOB, 0.70

        # ── 6. Drop shot ─────────────────────────────────────
        if (ball_velocity_mph < self.DROP_MAX_SPEED
                and ball_trajectory_angle > 30
                and self._has_steep_descent(ball_trajectory_angle)):
            return ShotType.DROP_SHOT, 0.65

        # ── 7. Slice ─────────────────────────────────────────
        if self._is_slice(ball_trajectory_angle, ball_velocity_mph, player_pose):
            side = self._determine_side(player_pose, player_position, ball_position)
            shot = ShotType.SLICE_FH if side == "fh" else ShotType.SLICE_BH
            return shot, 0.72

        # ── 8. Forehand / Backhand (default) ─────────────────
        side = self._determine_side(player_pose, player_position, ball_position)
        conf = 0.85 if player_pose else 0.75
        # Boost confidence if trajectory confirms direction change
        if prev_trajectory_angle is not None:
            angle_delta = abs(ball_trajectory_angle - prev_trajectory_angle)
            if angle_delta > 30:
                conf = min(conf + 0.05, 0.95)
        return (ShotType.FOREHAND if side == "fh" else ShotType.BACKHAND), conf

    def _determine_side(self, pose: Optional[PlayerPose],
                        player_pos: Optional[Point2D],
                        ball_pos: Optional[Point2D]) -> str:
        """Determine forehand or backhand side using pose and position data.

        Priority:
        1. Wrist-to-hip vector from pose (most accurate)
        2. Wrist-to-shoulder from pose (fallback)
        3. Ball position relative to player (spatial fallback)
        """
        # Method 1: Wrist-to-hip vector (best indicator of stroke side)
        if pose:
            rw = pose.right_wrist
            rh = pose.right_hip
            if rw and rh:
                # Right-handed: FH = right wrist right of right hip
                return "fh" if rw.x > rh.x else "bh"

            # Method 2: Wrist-to-shoulder
            rs = pose.right_shoulder
            if rw and rs:
                return "fh" if rw.x > rs.x else "bh"

        # Method 3: Spatial relationship
        if player_pos and ball_pos:
            return "fh" if ball_pos.x > player_pos.x else "bh"

        return "fh"  # default

    def _is_smash(self, ball_height_m: Optional[float],
                  trajectory_angle: float,
                  player_position: Optional[Point2D]) -> bool:
        """Detect smash/overhead conditions."""
        if ball_height_m and ball_height_m > self.SMASH_MIN_HEIGHT:
            if trajectory_angle < -20:  # downward trajectory
                if player_position and abs(player_position.y) < self.SMASH_MAX_NET_DIST:
                    return True
        return False

    def _is_slice(self, trajectory_angle: float, speed_mph: float,
                  pose: Optional[PlayerPose]) -> bool:
        """Detect slice shot from trajectory and speed."""
        if trajectory_angle < self.SLICE_MAX_ANGLE and speed_mph < self.SLICE_MAX_SPEED:
            return True
        # Additional check: low follow-through from pose
        if pose and trajectory_angle < 0:
            rw = pose.right_wrist
            re = pose.right_elbow
            if rw and re and rw.y > re.y:  # wrist below elbow = slice motion
                return True
        return False

    def _has_steep_descent(self, angle: float) -> bool:
        """Check for steep ball descent (drop shot indicator)."""
        return angle > 25

    # Legacy alias
    def _side(self, pose, player_pos, ball_pos) -> str:
        return self._determine_side(pose, player_pos, ball_pos)
