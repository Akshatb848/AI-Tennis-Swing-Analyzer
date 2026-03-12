"""
Pose Analyzer — Swing phase extraction and swing type classification.
Works on COCO keypoints from PlayerDetector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SwingType(str, Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE = "serve"
    VOLLEY = "volley"
    OVERHEAD = "overhead"
    SLICE = "slice"
    DROP_SHOT = "drop_shot"
    UNKNOWN = "unknown"


class SwingPhase(str, Enum):
    READY = "ready"
    PREPARATION = "preparation"
    BACKSWING = "backswing"
    FORWARD_SWING = "forward_swing"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class JointAngles:
    """Joint angles extracted from keypoints."""
    right_elbow: float = 0.0
    left_elbow: float = 0.0
    right_shoulder: float = 0.0
    left_shoulder: float = 0.0
    right_knee: float = 0.0
    left_knee: float = 0.0
    trunk_rotation: float = 0.0  # rotation of shoulder line relative to hip line
    trunk_tilt: float = 0.0


@dataclass
class PoseFrame:
    """A single frame of pose data."""
    frame_number: int = 0
    timestamp_ms: int = 0
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    joint_angles: Optional[JointAngles] = None
    swing_phase: SwingPhase = SwingPhase.READY
    court_position: tuple[float, float] = (0.0, 0.0)


@dataclass
class SwingSequence:
    """A complete swing extracted from a frame sequence."""
    player_id: str = ""
    swing_type: SwingType = SwingType.UNKNOWN
    frames: list[PoseFrame] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    confidence: float = 0.0
    max_racket_speed: float = 0.0  # estimated from wrist velocity
    contact_height: float = 0.0    # normalized 0-1

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def get_phase_frames(self, phase: SwingPhase) -> list[PoseFrame]:
        return [f for f in self.frames if f.swing_phase == phase]


# COCO keypoint indices
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


class PoseAnalyzer:
    """
    Analyze pose keypoints to classify swings and extract phases.

    Input: sequence of COCO-format keypoints per player per frame
    Output: classified SwingSequence with phase annotations
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._player_histories: dict[str, list[PoseFrame]] = {}
        self._pending_swings: dict[str, list[PoseFrame]] = {}

    def process_frame(
        self,
        player_id: str,
        keypoints: list[tuple[float, float, float]],
        frame_number: int,
        court_position: tuple[float, float] = (0.0, 0.0),
    ) -> Optional[SwingSequence]:
        """
        Process a frame of keypoints for one player.
        Returns a SwingSequence if a complete swing is detected.
        """
        if len(keypoints) < 17:
            return None

        kp_dict = {}
        for name, idx in KP.items():
            if idx < len(keypoints):
                kp_dict[name] = keypoints[idx]

        angles = self._compute_joint_angles(kp_dict)
        phase = self._classify_phase(kp_dict, angles, player_id)

        pose_frame = PoseFrame(
            frame_number=frame_number,
            timestamp_ms=int(frame_number / self.fps * 1000),
            keypoints=kp_dict,
            joint_angles=angles,
            swing_phase=phase,
            court_position=court_position,
        )

        # Track history
        if player_id not in self._player_histories:
            self._player_histories[player_id] = []
        self._player_histories[player_id].append(pose_frame)
        # Keep last 90 frames (3 seconds)
        if len(self._player_histories[player_id]) > 90:
            self._player_histories[player_id] = self._player_histories[player_id][-90:]

        # Swing detection: track phase transitions
        if player_id not in self._pending_swings:
            self._pending_swings[player_id] = []

        if phase in (SwingPhase.PREPARATION, SwingPhase.BACKSWING, SwingPhase.FORWARD_SWING, SwingPhase.CONTACT):
            self._pending_swings[player_id].append(pose_frame)
        elif phase == SwingPhase.FOLLOW_THROUGH and self._pending_swings.get(player_id):
            self._pending_swings[player_id].append(pose_frame)
        elif phase == SwingPhase.RECOVERY and self._pending_swings.get(player_id):
            # Swing complete
            swing_frames = self._pending_swings[player_id] + [pose_frame]
            self._pending_swings[player_id] = []
            return self._build_swing(player_id, swing_frames)
        else:
            # Not in a swing — reset if too many idle frames
            if len(self._pending_swings.get(player_id, [])) > 45:
                self._pending_swings[player_id] = []

        return None

    def get_movement_trajectory(self, player_id: str, last_n_frames: int = 30) -> list[tuple[float, float]]:
        """Get recent court positions for trajectory display."""
        history = self._player_histories.get(player_id, [])
        return [f.court_position for f in history[-last_n_frames:]]

    # ── Classification ───────────────────────────────────────────────────────

    def _build_swing(self, player_id: str, frames: list[PoseFrame]) -> SwingSequence:
        swing_type = self._classify_swing_type(frames)
        wrist_speed = self._estimate_racket_speed(frames)

        # Contact height (normalized)
        contact_frames = [f for f in frames if f.swing_phase == SwingPhase.CONTACT]
        contact_height = 0.5
        if contact_frames and "right_wrist" in contact_frames[0].keypoints:
            wrist_y = contact_frames[0].keypoints["right_wrist"][1]
            nose_y = contact_frames[0].keypoints.get("nose", (0, 0, 0))[1]
            ankle_y = contact_frames[0].keypoints.get("right_ankle", (0, 1, 0))[1]
            if ankle_y != nose_y:
                contact_height = max(0, min(1, (ankle_y - wrist_y) / (ankle_y - nose_y)))

        return SwingSequence(
            player_id=player_id,
            swing_type=swing_type,
            frames=frames,
            start_frame=frames[0].frame_number,
            end_frame=frames[-1].frame_number,
            confidence=0.75,
            max_racket_speed=wrist_speed,
            contact_height=contact_height,
        )

    def _classify_swing_type(self, frames: list[PoseFrame]) -> SwingType:
        """Classify swing type from keypoint sequence."""
        if not frames:
            return SwingType.UNKNOWN

        # Check for serve: high contact point + arms raised
        contact = [f for f in frames if f.swing_phase == SwingPhase.CONTACT]
        if contact:
            kp = contact[0].keypoints
            if "right_wrist" in kp and "nose" in kp:
                wrist_y = kp["right_wrist"][1]
                nose_y = kp["nose"][1]
                # In image coords, y increases downward. Wrist above nose = serve
                if wrist_y < nose_y * 0.8:
                    return SwingType.SERVE

        # Check for volley: player in net zone (court_y close to 0)
        avg_court_y = sum(abs(f.court_position[1]) for f in frames) / len(frames)
        if avg_court_y < 4.0:  # within service line
            return SwingType.VOLLEY

        # Forehand vs backhand: wrist position relative to shoulder line
        prep = [f for f in frames if f.swing_phase == SwingPhase.PREPARATION]
        if prep and prep[0].joint_angles:
            # Right-hand dominant: if wrist starts on right side = forehand
            kp = prep[0].keypoints
            if "right_wrist" in kp and "right_shoulder" in kp:
                wrist_x = kp["right_wrist"][0]
                shoulder_x = kp["right_shoulder"][0]
                if wrist_x > shoulder_x:
                    return SwingType.FOREHAND
                else:
                    return SwingType.BACKHAND

        return SwingType.FOREHAND  # default

    def _classify_phase(self, kp: dict, angles: JointAngles, player_id: str) -> SwingPhase:
        """Classify current swing phase from single-frame keypoints."""
        history = self._player_histories.get(player_id, [])

        if not history or len(history) < 2:
            return SwingPhase.READY

        # Wrist velocity indicates swing activity
        prev = history[-1].keypoints if history else {}
        wrist_velocity = 0.0
        if "right_wrist" in kp and "right_wrist" in prev:
            dx = kp["right_wrist"][0] - prev["right_wrist"][0]
            dy = kp["right_wrist"][1] - prev["right_wrist"][1]
            wrist_velocity = math.sqrt(dx * dx + dy * dy) * self.fps

        # Phase classification by wrist velocity thresholds
        if wrist_velocity > 500:
            return SwingPhase.FORWARD_SWING
        elif wrist_velocity > 300:
            # Check if accelerating (forward) or decelerating (follow-through)
            if len(history) >= 2:
                prev_phase = history[-1].swing_phase
                if prev_phase == SwingPhase.FORWARD_SWING:
                    return SwingPhase.FOLLOW_THROUGH
                elif prev_phase == SwingPhase.BACKSWING:
                    return SwingPhase.FORWARD_SWING
            return SwingPhase.CONTACT
        elif wrist_velocity > 100:
            prev_phase = history[-1].swing_phase if history else SwingPhase.READY
            if prev_phase in (SwingPhase.READY, SwingPhase.RECOVERY):
                return SwingPhase.PREPARATION
            elif prev_phase == SwingPhase.PREPARATION:
                return SwingPhase.BACKSWING
            elif prev_phase in (SwingPhase.FOLLOW_THROUGH, SwingPhase.CONTACT):
                return SwingPhase.RECOVERY
            return SwingPhase.PREPARATION
        else:
            prev_phase = history[-1].swing_phase if history else SwingPhase.READY
            if prev_phase == SwingPhase.FOLLOW_THROUGH:
                return SwingPhase.RECOVERY
            return SwingPhase.READY

    # ── Joint Angles ─────────────────────────────────────────────────────────

    def _compute_joint_angles(self, kp: dict) -> JointAngles:
        angles = JointAngles()
        angles.right_elbow = self._angle_at(kp, "right_shoulder", "right_elbow", "right_wrist")
        angles.left_elbow = self._angle_at(kp, "left_shoulder", "left_elbow", "left_wrist")
        angles.right_shoulder = self._angle_at(kp, "right_hip", "right_shoulder", "right_elbow")
        angles.left_shoulder = self._angle_at(kp, "left_hip", "left_shoulder", "left_elbow")
        angles.right_knee = self._angle_at(kp, "right_hip", "right_knee", "right_ankle")
        angles.left_knee = self._angle_at(kp, "left_hip", "left_knee", "left_ankle")
        angles.trunk_rotation = self._trunk_rotation(kp)
        return angles

    def _angle_at(self, kp: dict, a: str, b: str, c: str) -> float:
        """Compute angle at point b formed by a-b-c."""
        if a not in kp or b not in kp or c not in kp:
            return 0.0
        ax, ay, _ = kp[a]
        bx, by, _ = kp[b]
        cx, cy, _ = kp[c]
        ba = (ax - bx, ay - by)
        bc = (cx - bx, cy - by)
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if mag_ba * mag_bc == 0:
            return 0.0
        cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    def _trunk_rotation(self, kp: dict) -> float:
        """Compute trunk rotation from shoulder vs hip lines."""
        if not all(k in kp for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            return 0.0
        shoulder_dx = kp["right_shoulder"][0] - kp["left_shoulder"][0]
        shoulder_dy = kp["right_shoulder"][1] - kp["left_shoulder"][1]
        hip_dx = kp["right_hip"][0] - kp["left_hip"][0]
        hip_dy = kp["right_hip"][1] - kp["left_hip"][1]
        s_angle = math.atan2(shoulder_dy, shoulder_dx)
        h_angle = math.atan2(hip_dy, hip_dx)
        return math.degrees(s_angle - h_angle)

    def _estimate_racket_speed(self, frames: list[PoseFrame]) -> float:
        """Estimate max racket speed from wrist velocity in swing frames."""
        max_speed = 0.0
        for i in range(1, len(frames)):
            prev_kp = frames[i - 1].keypoints
            curr_kp = frames[i].keypoints
            if "right_wrist" in prev_kp and "right_wrist" in curr_kp:
                dx = curr_kp["right_wrist"][0] - prev_kp["right_wrist"][0]
                dy = curr_kp["right_wrist"][1] - prev_kp["right_wrist"][1]
                speed = math.sqrt(dx * dx + dy * dy) * self.fps
                max_speed = max(max_speed, speed)
        return max_speed

    def reset(self):
        self._player_histories.clear()
        self._pending_swings.clear()
