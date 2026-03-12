"""
Avatar Renderer — Semi-realistic anime-style player avatar from pose keypoints.
Visual-only layer. Does not affect analytics. Raw pose data preserved separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AvatarStyle:
    """Visual style configuration for avatar rendering."""
    body_color: str = "#3a4a5c"
    outline_color: str = "#1a2a3c"
    limb_width: float = 6.0
    joint_radius: float = 4.0
    head_radius: float = 14.0
    torso_width: float = 24.0
    opacity: float = 0.85
    show_racket: bool = True
    racket_color: str = "#c0c0c0"


@dataclass
class LimbSegment:
    """A single rendered limb segment."""
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    width: float = 6.0
    color: str = "#3a4a5c"
    joint_visible: bool = True


@dataclass
class AvatarFrame:
    """Complete avatar render data for one frame."""
    player_id: str = ""
    frame_number: int = 0
    # Head
    head_x: float = 0.0
    head_y: float = 0.0
    head_radius: float = 14.0
    head_rotation: float = 0.0
    # Torso
    torso_segments: list[LimbSegment] = field(default_factory=list)
    # Limbs
    limb_segments: list[LimbSegment] = field(default_factory=list)
    # Racket
    racket_segment: Optional[LimbSegment] = None
    # Metadata
    body_rotation: float = 0.0
    facing_direction: str = "right"
    style: AvatarStyle = field(default_factory=AvatarStyle)

    def to_render_data(self) -> dict:
        """Return structured data for frontend rendering."""
        return {
            "player_id": self.player_id,
            "frame": self.frame_number,
            "head": {
                "x": round(self.head_x, 1),
                "y": round(self.head_y, 1),
                "radius": self.head_radius,
                "rotation": round(self.head_rotation, 1),
            },
            "torso": [
                {"x1": s.start_x, "y1": s.start_y, "x2": s.end_x, "y2": s.end_y, "width": s.width}
                for s in self.torso_segments
            ],
            "limbs": [
                {"x1": s.start_x, "y1": s.start_y, "x2": s.end_x, "y2": s.end_y, "width": s.width, "joint": s.joint_visible}
                for s in self.limb_segments
            ],
            "racket": {
                "x1": self.racket_segment.start_x, "y1": self.racket_segment.start_y,
                "x2": self.racket_segment.end_x, "y2": self.racket_segment.end_y,
            } if self.racket_segment else None,
            "rotation": round(self.body_rotation, 1),
            "facing": self.facing_direction,
            "style": {
                "body_color": self.style.body_color,
                "outline": self.style.outline_color,
                "opacity": self.style.opacity,
            },
        }


# COCO keypoint indices for reference
_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Limb connections: (start_keypoint_index, end_keypoint_index, width_multiplier)
_LIMB_CONNECTIONS = [
    # Torso
    (5, 6, 1.5),    # left_shoulder → right_shoulder
    (5, 11, 1.3),   # left_shoulder → left_hip
    (6, 12, 1.3),   # right_shoulder → right_hip
    (11, 12, 1.4),  # left_hip → right_hip
    # Right arm
    (6, 8, 1.0),    # right_shoulder → right_elbow
    (8, 10, 0.9),   # right_elbow → right_wrist
    # Left arm
    (5, 7, 1.0),    # left_shoulder → left_elbow
    (7, 9, 0.9),    # left_elbow → left_wrist
    # Right leg
    (12, 14, 1.1),  # right_hip → right_knee
    (14, 16, 1.0),  # right_knee → right_ankle
    # Left leg
    (11, 13, 1.1),  # left_hip → left_knee
    (13, 15, 1.0),  # left_knee → left_ankle
]

_TORSO_INDICES = {0, 1, 2, 3}   # first 4 connections are torso
_ARM_INDICES = {4, 5, 6, 7}     # arms
_LEG_INDICES = {8, 9, 10, 11}   # legs


class AvatarRenderer:
    """
    Render semi-realistic anime-style avatars from COCO keypoints.

    - Maps pose keypoints to body segments
    - Maintains limb proportions and orientation
    - Adds racket representation on dominant hand
    - Output is structured render data (not raw pixels)
    - Visually neutral and non-distracting
    """

    def __init__(self, style: Optional[AvatarStyle] = None):
        self.style = style or AvatarStyle()
        self._player_styles: dict[str, AvatarStyle] = {}

    def set_player_style(self, player_id: str, style: AvatarStyle):
        self._player_styles[player_id] = style

    def render_frame(
        self,
        player_id: str,
        keypoints: list[tuple[float, float, float]],
        frame_number: int = 0,
        dominant_hand: str = "right",
    ) -> AvatarFrame:
        """
        Generate avatar render data from keypoints.

        Args:
            player_id: persistent player identifier
            keypoints: 17 COCO keypoints as (x, y, confidence)
            frame_number: current frame
            dominant_hand: "right" or "left" for racket placement
        """
        style = self._player_styles.get(player_id, self.style)

        if len(keypoints) < 17:
            return AvatarFrame(player_id=player_id, frame_number=frame_number, style=style)

        kp = keypoints
        frame = AvatarFrame(player_id=player_id, frame_number=frame_number, style=style)

        # Head — position from nose, size proportional
        nose = kp[0]
        if nose[2] > 0.3:
            frame.head_x = nose[0]
            frame.head_y = nose[1] - style.head_radius * 0.5
            frame.head_radius = style.head_radius
            # Head rotation from ear positions
            if kp[3][2] > 0.2 and kp[4][2] > 0.2:
                dx = kp[4][0] - kp[3][0]
                dy = kp[4][1] - kp[3][1]
                frame.head_rotation = math.degrees(math.atan2(dy, dx))

        # Body rotation from shoulder/hip lines
        if kp[5][2] > 0.2 and kp[6][2] > 0.2:
            shoulder_dx = kp[6][0] - kp[5][0]
            frame.body_rotation = math.degrees(math.atan2(0, shoulder_dx))
            frame.facing_direction = "right" if shoulder_dx > 0 else "left"

        # Build limb segments
        for i, (start_idx, end_idx, width_mult) in enumerate(_LIMB_CONNECTIONS):
            start_kp = kp[start_idx]
            end_kp = kp[end_idx]

            # Skip low-confidence keypoints
            if start_kp[2] < 0.2 or end_kp[2] < 0.2:
                continue

            segment = LimbSegment(
                start_x=round(start_kp[0], 1),
                start_y=round(start_kp[1], 1),
                end_x=round(end_kp[0], 1),
                end_y=round(end_kp[1], 1),
                width=style.limb_width * width_mult,
                color=style.body_color,
                joint_visible=True,
            )

            if i in _TORSO_INDICES:
                frame.torso_segments.append(segment)
            else:
                frame.limb_segments.append(segment)

        # Racket — extend from dominant wrist
        if style.show_racket:
            wrist_idx = 10 if dominant_hand == "right" else 9
            elbow_idx = 8 if dominant_hand == "right" else 7
            wrist = kp[wrist_idx]
            elbow = kp[elbow_idx]

            if wrist[2] > 0.3 and elbow[2] > 0.3:
                # Racket extends in forearm direction
                dx = wrist[0] - elbow[0]
                dy = wrist[1] - elbow[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    norm_dx = dx / length
                    norm_dy = dy / length
                    racket_length = length * 1.8
                    frame.racket_segment = LimbSegment(
                        start_x=round(wrist[0], 1),
                        start_y=round(wrist[1], 1),
                        end_x=round(wrist[0] + norm_dx * racket_length, 1),
                        end_y=round(wrist[1] + norm_dy * racket_length, 1),
                        width=3.0,
                        color=style.racket_color,
                        joint_visible=False,
                    )

        return frame

    def render_batch(
        self,
        players: dict[str, list[tuple[float, float, float]]],
        frame_number: int = 0,
    ) -> list[AvatarFrame]:
        """Render avatars for all players in a single frame."""
        return [
            self.render_frame(pid, kps, frame_number)
            for pid, kps in players.items()
        ]
