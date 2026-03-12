"""Tests for avatar renderer — pose-to-avatar mapping."""

import pytest
from tennis.video.avatar_renderer import AvatarRenderer, AvatarStyle, AvatarFrame
from tennis.ml.pose_analyzer import PoseAnalyzer, SwingPhase, SwingType


class TestAvatarRenderer:
    """Avatar render data generation tests."""

    def setup_method(self):
        self.renderer = AvatarRenderer()
        # Realistic COCO keypoints (17 points: x, y, confidence)
        self.keypoints = [
            (320, 100, 0.95),  # nose
            (310, 95, 0.90),   # left_eye
            (330, 95, 0.90),   # right_eye
            (300, 100, 0.85),  # left_ear
            (340, 100, 0.85),  # right_ear
            (280, 150, 0.92),  # left_shoulder
            (360, 150, 0.92),  # right_shoulder
            (260, 220, 0.88),  # left_elbow
            (380, 220, 0.88),  # right_elbow
            (250, 290, 0.85),  # left_wrist
            (400, 290, 0.85),  # right_wrist
            (290, 280, 0.90),  # left_hip
            (350, 280, 0.90),  # right_hip
            (285, 380, 0.88),  # left_knee
            (355, 380, 0.88),  # right_knee
            (280, 470, 0.85),  # left_ankle
            (360, 470, 0.85),  # right_ankle
        ]

    def test_basic_render(self):
        frame = self.renderer.render_frame("player_0", self.keypoints, frame_number=1)
        assert frame.player_id == "player_0"
        assert frame.frame_number == 1
        assert frame.head_x > 0
        assert len(frame.limb_segments) > 0

    def test_torso_segments(self):
        frame = self.renderer.render_frame("player_0", self.keypoints)
        assert len(frame.torso_segments) >= 3  # shoulders, hips connections

    def test_limb_segments(self):
        frame = self.renderer.render_frame("player_0", self.keypoints)
        assert len(frame.limb_segments) >= 6  # arms + legs

    def test_racket_rendered(self):
        frame = self.renderer.render_frame("player_0", self.keypoints)
        assert frame.racket_segment is not None

    def test_no_racket_when_disabled(self):
        no_racket = AvatarStyle(show_racket=False)
        renderer = AvatarRenderer(style=no_racket)
        frame = renderer.render_frame("player_0", self.keypoints)
        assert frame.racket_segment is None

    def test_render_data_structure(self):
        frame = self.renderer.render_frame("player_0", self.keypoints)
        data = frame.to_render_data()
        assert "head" in data
        assert "torso" in data
        assert "limbs" in data
        assert "style" in data
        assert data["player_id"] == "player_0"

    def test_facing_direction(self):
        frame = self.renderer.render_frame("player_0", self.keypoints)
        assert frame.facing_direction in ("left", "right")

    def test_incomplete_keypoints(self):
        partial = [(100, 100, 0.3)] * 5  # too few
        frame = self.renderer.render_frame("player_0", partial)
        assert frame.player_id == "player_0"
        assert len(frame.limb_segments) == 0

    def test_batch_render(self):
        players = {
            "player_0": self.keypoints,
            "player_1": self.keypoints,
        }
        frames = self.renderer.render_batch(players, frame_number=5)
        assert len(frames) == 2

    def test_custom_style(self):
        style = AvatarStyle(body_color="#ff0000", opacity=0.5)
        self.renderer.set_player_style("player_0", style)
        frame = self.renderer.render_frame("player_0", self.keypoints)
        assert frame.style.body_color == "#ff0000"
        assert frame.style.opacity == 0.5


class TestPoseAnalyzer:
    """Pose analysis tests."""

    def setup_method(self):
        self.analyzer = PoseAnalyzer(fps=30.0)
        self.keypoints = [
            (320, 100, 0.95),  # nose
            (310, 95, 0.90),   # left_eye
            (330, 95, 0.90),   # right_eye
            (300, 100, 0.85),  # left_ear
            (340, 100, 0.85),  # right_ear
            (280, 150, 0.92),  # left_shoulder
            (360, 150, 0.92),  # right_shoulder
            (260, 220, 0.88),  # left_elbow
            (380, 220, 0.88),  # right_elbow
            (250, 290, 0.85),  # left_wrist
            (400, 290, 0.85),  # right_wrist
            (290, 280, 0.90),  # left_hip
            (350, 280, 0.90),  # right_hip
            (285, 380, 0.88),  # left_knee
            (355, 380, 0.88),  # right_knee
            (280, 470, 0.85),  # left_ankle
            (360, 470, 0.85),  # right_ankle
        ]

    def test_process_single_frame(self):
        result = self.analyzer.process_frame("player_0", self.keypoints, frame_number=1)
        # Single frame won't complete a swing
        assert result is None

    def test_movement_trajectory(self):
        for i in range(5):
            self.analyzer.process_frame(
                "player_0", self.keypoints, frame_number=i, court_position=(float(i), 5.0)
            )
        trajectory = self.analyzer.get_movement_trajectory("player_0")
        assert len(trajectory) == 5

    def test_joint_angle_computation(self):
        self.analyzer.process_frame("player_0", self.keypoints, frame_number=1)
        history = self.analyzer._player_histories.get("player_0", [])
        assert len(history) == 1
        angles = history[0].joint_angles
        assert angles is not None
        assert angles.right_elbow >= 0

    def test_insufficient_keypoints(self):
        result = self.analyzer.process_frame("player_0", [(0, 0, 0.1)] * 5, frame_number=1)
        assert result is None

    def test_reset(self):
        self.analyzer.process_frame("player_0", self.keypoints, frame_number=1)
        self.analyzer.reset()
        assert len(self.analyzer._player_histories) == 0
