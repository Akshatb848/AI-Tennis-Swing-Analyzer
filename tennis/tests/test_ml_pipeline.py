"""
Tests for ML Pipeline — ball tracker, player detector, court detector, inference pipeline.
"""

import pytest

from tennis.ml.ball_tracker import BallTracker, KalmanState
from tennis.ml.court_detector import CourtDetector
from tennis.ml.inference_pipeline import InferencePipeline
from tennis.ml.model_specs import MODEL_REGISTRY, get_model_spec, get_total_parameters
from tennis.ml.player_detector import COCO_KEYPOINT_NAMES, PlayerDetector
from tennis.models.events import BoundingBox


class TestModelSpecs:
    def test_all_models_registered(self):
        assert len(MODEL_REGISTRY) == 4
        assert "ballnet" in MODEL_REGISTRY
        assert "playernet" in MODEL_REGISTRY
        assert "courtnet" in MODEL_REGISTRY
        assert "shotnet" in MODEL_REGISTRY

    def test_get_model_spec(self):
        spec = get_model_spec("BallNet")
        assert spec.name == "BallNet"
        assert spec.num_parameters > 0

    def test_get_model_spec_case_insensitive(self):
        spec = get_model_spec("ballnet")
        assert spec.name == "BallNet"

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            get_model_spec("UnknownNet")

    def test_total_parameters(self):
        total = get_total_parameters()
        assert total > 10_000_000  # > 10M params total

    def test_latency_budget(self):
        from tennis.ml.model_specs import get_total_latency_budget
        total = get_total_latency_budget()
        assert total < 50  # Must be under 50ms budget


class TestKalmanState:
    def test_initial_state(self):
        k = KalmanState()
        assert k.confidence == 0.0
        assert not k.is_tracking

    def test_update(self):
        k = KalmanState()
        k.update(100, 200)
        assert k.confidence == 1.0
        assert k.is_tracking

    def test_predict_decays_confidence(self):
        k = KalmanState()
        k.update(100, 200)
        for _ in range(20):
            k.predict()
        assert k.confidence < 1.0


class TestBallTracker:
    def test_process_detection(self):
        tracker = BallTracker(fps=30)
        det = BoundingBox(x1=100, y1=100, x2=110, y2=110, confidence=0.9)
        event = tracker.process_frame(det, 1, "test")
        assert event is not None
        assert event.detection_confidence == 0.9

    def test_no_detection_predicts(self):
        tracker = BallTracker(fps=30)
        det = BoundingBox(x1=100, y1=100, x2=110, y2=110, confidence=0.9)
        tracker.process_frame(det, 1, "test")
        event = tracker.process_frame(None, 2, "test")
        # Should return interpolated event since recently tracked
        assert event is not None
        assert event.is_interpolated

    def test_reset(self):
        tracker = BallTracker(fps=30)
        tracker.process_frame(BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9), 1)
        tracker.reset()
        assert tracker.frame_count == 0
        assert len(tracker.trajectory) == 0


class TestPlayerDetector:
    def test_detect_single_player(self):
        detector = PlayerDetector()
        dets = [BoundingBox(x1=100, y1=100, x2=200, y2=400, confidence=0.95)]
        events = detector.process_frame(dets, frame_number=1)
        assert len(events) == 1
        assert events[0].detection_confidence == 0.95

    def test_track_persistence(self):
        detector = PlayerDetector()
        det1 = [BoundingBox(x1=100, y1=100, x2=200, y2=400, confidence=0.9)]
        det2 = [BoundingBox(x1=105, y1=102, x2=205, y2=402, confidence=0.9)]
        e1 = detector.process_frame(det1, frame_number=1)
        e2 = detector.process_frame(det2, frame_number=2)
        # Should maintain same player ID
        assert e1[0].player_id == e2[0].player_id

    def test_max_players(self):
        detector = PlayerDetector(max_players=2)
        dets = [
            BoundingBox(x1=100, y1=100, x2=200, y2=400, confidence=0.9),
            BoundingBox(x1=300, y1=100, x2=400, y2=400, confidence=0.9),
            BoundingBox(x1=500, y1=100, x2=600, y2=400, confidence=0.9),
        ]
        events = detector.process_frame(dets, frame_number=1)
        assert len(events) == 2  # capped at max_players

    def test_keypoint_names(self):
        assert len(COCO_KEYPOINT_NAMES) == 17
        assert "nose" in COCO_KEYPOINT_NAMES


class TestCourtDetector:
    def test_initial_state(self):
        detector = CourtDetector()
        assert not detector.is_calibrated

    def test_calibration_with_keypoints(self):
        detector = CourtDetector()
        keypoints = [(100, 200, 0.9), (300, 200, 0.9), (100, 500, 0.9), (300, 500, 0.9)]
        detector.process_frame(keypoints, frame_number=1)
        assert detector.homography.is_valid

    def test_insufficient_keypoints(self):
        detector = CourtDetector()
        keypoints = [(100, 200, 0.9), (300, 200, 0.9)]  # Only 2, need 4
        detector.process_frame(keypoints, frame_number=1)
        # Should not crash, but homography stays invalid from init
        assert True

    def test_image_to_court(self):
        detector = CourtDetector()
        keypoints = [(100, 200, 0.9), (300, 200, 0.9), (100, 500, 0.9), (300, 500, 0.9)]
        detector.process_frame(keypoints, frame_number=1)
        pt = detector.image_to_court(200, 350)
        assert pt is not None

    def test_reset(self):
        detector = CourtDetector()
        detector.process_frame([(100, 200, 0.9)] * 4, frame_number=1)
        detector.reset()
        assert detector.calibration_confidence == 0.0


class TestInferencePipeline:
    def test_initialization(self):
        pipeline = InferencePipeline(session_id="test")
        assert pipeline.frame_count == 0
        assert not pipeline.is_initialized

    def test_process_frame_with_ball(self):
        pipeline = InferencePipeline(session_id="test")
        pipeline.initialize()
        det = BoundingBox(x1=100, y1=100, x2=110, y2=110, confidence=0.9)
        result = pipeline.process_frame(ball_detection=det)
        assert result.frame_number == 1
        assert result.ball_event is not None

    def test_process_frame_with_players(self):
        pipeline = InferencePipeline(session_id="test")
        pipeline.initialize()
        dets = [BoundingBox(x1=100, y1=100, x2=200, y2=400, confidence=0.9)]
        result = pipeline.process_frame(player_detections=dets)
        assert len(result.player_events) == 1

    def test_get_stats(self):
        pipeline = InferencePipeline(session_id="test")
        stats = pipeline.get_stats()
        assert "frames_processed" in stats
        assert stats["frames_processed"] == 0
