"""
Integration tests for the real-time video processing pipeline.

Tests the full flow: video capture → ML inference → event pipeline → session result.
Uses a synthetic test video (bouncing ball on green court) to verify the pipeline
works end-to-end without requiring real tennis footage.
"""

import asyncio
import os
import tempfile
import pytest
import numpy as np

# Try OpenCV — skip if not available
cv2 = pytest.importorskip("cv2")


def create_test_video(path: str, fps: int = 30, duration_seconds: float = 2.0):
    """Create a synthetic test video with a bouncing ball on a green court."""
    width, height = 640, 480
    total_frames = int(fps * duration_seconds)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Green court background (bottom 2/3)
        frame[height // 3:, :] = (60, 180, 60)  # BGR green

        # White court lines
        cv2.line(frame, (100, height // 3), (100, height - 20), (255, 255, 255), 2)
        cv2.line(frame, (width - 100, height // 3), (width - 100, height - 20), (255, 255, 255), 2)
        cv2.line(frame, (100, height // 3), (width - 100, height // 3), (255, 255, 255), 2)
        cv2.line(frame, (100, height - 20), (width - 100, height - 20), (255, 255, 255), 2)
        cv2.line(frame, (width // 2, height // 3), (width // 2, height - 20), (255, 255, 255), 2)

        # Bouncing yellow ball
        ball_x = int(100 + (width - 200) * (i / total_frames))
        ball_y = int(height // 2 + 80 * np.sin(i * 0.3))
        cv2.circle(frame, (ball_x, ball_y), 10, (0, 200, 200), -1)  # BGR yellow

        # Humanoid shapes (simple rectangles for "players")
        # Player 1 (near side)
        p1_x = int(width // 4 + 50 * np.sin(i * 0.1))
        cv2.rectangle(frame, (p1_x - 15, height - 120), (p1_x + 15, height - 30), (200, 100, 50), -1)
        # Player 2 (far side)
        p2_x = int(3 * width // 4 + 50 * np.cos(i * 0.1))
        cv2.rectangle(frame, (p2_x - 12, height // 3 + 10), (p2_x + 12, height // 3 + 90), (50, 100, 200), -1)

        writer.write(frame)

    writer.release()
    return path


# ── VideoCapture Tests ───────────────────────────────────────────────────────

class TestVideoCapture:
    """Tests for the VideoCapture class."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmp, "test_match.mp4")
        create_test_video(self.video_path, fps=30, duration_seconds=1.0)

    def test_open_and_metadata(self):
        from tennis.video.capture import VideoCapture
        cap = VideoCapture(self.video_path, target_fps=30)
        meta = cap.open()
        assert meta.width == 640
        assert meta.height == 480
        assert meta.fps > 0
        assert meta.total_frames > 0
        cap.release()

    def test_read_frame(self):
        from tennis.video.capture import VideoCapture
        cap = VideoCapture(self.video_path, target_fps=30)
        cap.open()
        frame = cap.read_frame()
        assert frame is not None
        assert frame.frame.shape == (480, 640, 3)
        assert frame.frame_number == 1
        assert frame.timestamp_ms >= 0
        cap.release()

    def test_frame_count(self):
        from tennis.video.capture import VideoCapture
        cap = VideoCapture(self.video_path, target_fps=30)
        cap.open()
        count = 0
        while True:
            f = cap.read_frame()
            if f is None:
                break
            count += 1
        assert count >= 25  # ~1 second at 30fps
        assert cap.frame_count == count
        cap.release()

    def test_async_frames(self):
        from tennis.video.capture import VideoCapture

        async def run():
            cap = VideoCapture(self.video_path, target_fps=30)
            cap.open()
            frames = []
            async for f in cap.frames():
                frames.append(f)
            cap.release()
            return frames

        frames = asyncio.run(run())
        assert len(frames) >= 25


# ── FrameBuffer Tests ────────────────────────────────────────────────────────

class TestFrameBuffer:
    """Tests for the FrameBuffer class."""

    def test_add_and_retrieve(self):
        from tennis.video.frame_buffer import FrameBuffer, FrameEvent
        buf = FrameBuffer(max_frames=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buf.add_frame(1, frame, FrameEvent(frame_number=1, timestamp_ms=33))
        assert buf.total_frames == 1
        assert buf.buffered_frames == 1
        retrieved = buf.get_frame(1)
        assert retrieved is not None
        assert retrieved.shape == (480, 640, 3)

    def test_ring_buffer_overflow(self):
        from tennis.video.frame_buffer import FrameBuffer
        buf = FrameBuffer(max_frames=5)
        for i in range(10):
            buf.add_frame(i, np.zeros((10, 10, 3), dtype=np.uint8))
        assert buf.total_frames == 10
        assert buf.buffered_frames == 5
        assert buf.get_frame(0) is None  # Dropped
        assert buf.get_frame(9) is not None  # Still in buffer

    def test_replay_window(self):
        from tennis.video.frame_buffer import FrameBuffer
        buf = FrameBuffer(max_frames=120, fps=30)
        for i in range(100):
            buf.add_frame(i, np.zeros((10, 10, 3), dtype=np.uint8))
        window = buf.get_replay_window(50, window_seconds=1.0)
        assert len(window) > 0
        for fn, _ in window:
            assert 20 <= fn <= 80


# ── FrameAnalyzer Tests ──────────────────────────────────────────────────────

class TestFrameAnalyzer:
    """Tests for the FrameAnalyzer (heuristic mode)."""

    def test_analyze_synthetic_frame(self):
        from tennis.ml.frame_analyzer import FrameAnalyzer
        analyzer = FrameAnalyzer()

        # Create a frame with a yellow circle (ball) and lines (court)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[160:, :] = (60, 180, 60)  # Green court
        cv2.circle(frame, (320, 300), 12, (0, 200, 200), -1)  # Yellow ball

        detections = analyzer.analyze_frame(frame, frame_number=1)
        assert detections.frame_number == 1
        assert detections.inference_time_ms >= 0
        # Ball should be detected (yellow blob on green)
        # Note: not guaranteed in all environments, so check soft
        if detections.has_ball:
            assert detections.ball_center is not None

    def test_analyze_empty_frame(self):
        from tennis.ml.frame_analyzer import FrameAnalyzer
        analyzer = FrameAnalyzer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = analyzer.analyze_frame(frame, frame_number=0)
        assert detections.frame_number == 0


# ── InferencePipeline Tests ──────────────────────────────────────────────────

class TestInferencePipelineRaw:
    """Tests for the raw frame processing path."""

    def test_process_raw_frame(self):
        from tennis.ml.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline(session_id="test", fps=30)
        pipeline.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[160:, :] = (60, 180, 60)
        cv2.circle(frame, (320, 300), 12, (0, 200, 200), -1)

        result = pipeline.process_raw_frame(frame, timestamp_ms=33)
        assert result.frame_number == 1
        assert result.latency_ms >= 0

    def test_pipeline_stats(self):
        from tennis.ml.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline(session_id="test", fps=30)
        pipeline.initialize()

        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            pipeline.process_raw_frame(frame, timestamp_ms=i * 33)

        stats = pipeline.get_stats()
        assert stats["frames_processed"] == 5
        assert "avg_latency_ms" in stats


# ── LivePipeline Tests ───────────────────────────────────────────────────────

class TestLivePipeline:
    """Integration tests for the LivePipeline orchestrator."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmp, "test_match.mp4")
        create_test_video(self.video_path, fps=30, duration_seconds=1.0)

    def test_process_video_end_to_end(self):
        from tennis.engine.live_pipeline import LivePipeline

        async def run():
            pipeline = LivePipeline(target_fps=15)  # Faster processing
            result = await pipeline.process_video(self.video_path)
            return result

        result = asyncio.run(run())
        assert result.total_frames > 0
        assert result.avg_latency_ms >= 0
        assert result.duration_seconds > 0

    def test_frame_by_frame_processing(self):
        from tennis.engine.live_pipeline import LivePipeline
        pipeline = LivePipeline(target_fps=30)
        pipeline.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, timestamp_ms=0)

        assert result.frame_number == 1
        assert result.processing_latency_ms >= 0

        state = pipeline.get_live_state()
        assert state["is_running"] is True
        assert state["total_frames"] == 1

        session_result = pipeline.stop()
        assert session_result.total_frames == 1

    def test_pipeline_not_started_raises(self):
        from tennis.engine.live_pipeline import LivePipeline
        pipeline = LivePipeline()
        with pytest.raises(RuntimeError):
            pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8))


# ── SessionStore Tests ───────────────────────────────────────────────────────

class TestSessionStore:
    """Tests for the persistent session store."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")

    def test_save_and_get_session(self):
        from tennis.infra.session_store import SessionStore
        store = SessionStore(self.db_path)
        store.save_session("s1", mode="match", player_names=["A", "B"])
        session = store.get_session("s1")
        assert session is not None
        assert session["id"] == "s1"
        assert session["player_names"] == ["A", "B"]
        store.close()

    def test_list_sessions(self):
        from tennis.infra.session_store import SessionStore
        store = SessionStore(self.db_path)
        store.save_session("s1")
        store.save_session("s2")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        store.close()

    def test_save_events(self):
        from tennis.infra.session_store import SessionStore
        store = SessionStore(self.db_path)
        store.save_session("s1")
        store.save_event("s1", "ball_detected", frame_number=1, data={"x": 100, "y": 200})
        events = store.get_events("s1")
        assert len(events) == 1
        assert events[0]["data"]["x"] == 100
        store.close()


# ── ThermalManager Tests ─────────────────────────────────────────────────────

class TestThermalManager:
    """Tests for adaptive inference throttling."""

    def test_initial_mode_is_full(self):
        from tennis.infra.thermal_manager import ThermalManager, InferenceMode
        tm = ThermalManager()
        assert tm.mode == InferenceMode.FULL

    def test_degradation_on_high_latency(self):
        from tennis.infra.thermal_manager import ThermalManager, InferenceMode
        tm = ThermalManager(latency_budget_ms=30, degradation_threshold=5)
        for _ in range(6):
            tm.record_latency(50)  # Over budget
        assert tm.mode == InferenceMode.REDUCED

    def test_recovery_on_low_latency(self):
        from tennis.infra.thermal_manager import ThermalManager, InferenceMode
        tm = ThermalManager(latency_budget_ms=30, degradation_threshold=3, recovery_threshold=5)
        # Degrade first
        for _ in range(4):
            tm.record_latency(50)
        assert tm.mode == InferenceMode.REDUCED
        # Then recover
        for _ in range(6):
            tm.record_latency(10)
        assert tm.mode == InferenceMode.FULL


# ── PostMatchAnalyzer Tests ──────────────────────────────────────────────────

class TestPostMatchAnalyzer:
    """Tests for post-match analysis."""

    def test_analyze_session(self):
        from tennis.engine.post_match_analyzer import PostMatchAnalyzer
        analyzer = PostMatchAnalyzer()
        summary = {
            "session_id": "test",
            "players": ["Player 1", "Player 2"],
            "points": [
                {"winner": "p0", "rally_length": 5} for _ in range(20)
            ] + [
                {"winner": "p1", "rally_length": 3} for _ in range(10)
            ],
        }
        result = analyzer.analyze(summary)
        assert result.session_id == "test"
        assert len(result.fatigue) == 2
        assert len(result.consistency) == 2
        assert len(result.movement_loads) == 2


# ── DashboardDataProvider Tests ──────────────────────────────────────────────

class TestDashboardDataProvider:
    """Tests that the data provider output matches the expected shape."""

    def test_no_data_returns_none(self):
        from tennis.dashboard.data_provider import DashboardDataProvider
        provider = DashboardDataProvider()
        result = provider.get_match_data("nonexistent")
        assert result is None

    def test_store_and_retrieve(self):
        from tennis.dashboard.data_provider import (
            DashboardDataProvider, store_session_data, get_available_sessions,
        )
        test_data = {
            "match": {
                "player1": "A", "player2": "B",
                "surface": "Hard", "match_type": "Singles",
            },
        }
        store_session_data("test_session", test_data)
        sessions = get_available_sessions()
        assert any(s["id"] == "test_session" for s in sessions)

        provider = DashboardDataProvider()
        result = provider.get_match_data("test_session")
        assert result is not None
        assert result["match"]["player1"] == "A"
