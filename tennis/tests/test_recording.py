"""Tests for the recording flow — setup, state transitions, auto-segmentation."""

import pytest
from tennis.engine.recording import (
    RecordingSession, MatchSetupConfig, MatchType, Environment,
    Handedness, RecordingState,
)


class TestMatchSetup:
    """Match setup configuration tests."""

    def test_default_setup(self):
        session = RecordingSession()
        result = session.setup()
        assert result["status"] == "ready"
        assert result["match_type"] == "singles"
        assert len(result["players"]) == 2
        assert session.state == RecordingState.CALIBRATING

    def test_singles_setup(self):
        config = MatchSetupConfig(
            match_type=MatchType.SINGLES,
            player_names=["Alice", "Bob"],
        )
        session = RecordingSession()
        result = session.setup(config)
        assert result["status"] == "ready"
        assert result["players"] == ["Alice", "Bob"]

    def test_doubles_setup(self):
        config = MatchSetupConfig(
            match_type=MatchType.DOUBLES,
            player_count=4,
            player_names=["A", "B", "C", "D"],
            player_handedness=[Handedness.RIGHT, Handedness.LEFT, Handedness.AUTO, Handedness.RIGHT],
        )
        session = RecordingSession()
        result = session.setup(config)
        assert result["status"] == "ready"

    def test_practice_rally_setup(self):
        config = MatchSetupConfig(match_type=MatchType.PRACTICE_RALLY)
        session = RecordingSession()
        result = session.setup(config)
        assert result["status"] == "ready"
        assert result["match_type"] == "practice_rally"

    def test_environment_indoor(self):
        config = MatchSetupConfig(environment=Environment.INDOOR)
        session = RecordingSession()
        result = session.setup(config)
        assert result["environment"] == "indoor"


class TestRecordingStateTransitions:
    """Recording state machine tests."""

    def setup_method(self):
        self.session = RecordingSession()
        self.session.setup()

    def test_start_recording(self):
        result = self.session.start_recording()
        assert result["status"] == "recording"
        assert self.session.state == RecordingState.RECORDING

    def test_stop_recording(self):
        self.session.start_recording()
        result = self.session.stop_recording()
        assert result["status"] == "complete"
        assert self.session.state == RecordingState.COMPLETE

    def test_cannot_start_without_setup(self):
        fresh = RecordingSession()
        result = fresh.start_recording()
        assert result["status"] == "error"

    def test_cannot_stop_without_start(self):
        result = self.session.stop_recording()
        assert result["status"] == "error"

    def test_summary_after_complete(self):
        self.session.start_recording()
        self.session.stop_recording()
        summary = self.session.get_summary()
        assert summary["state"] == "complete"
        assert "points" in summary
        assert "line_calls" in summary


class TestAutoSegmentation:
    """Auto-segmentation during recording."""

    def test_frame_processing_returns_none_when_not_recording(self):
        session = RecordingSession()
        result = session.process_frame()
        assert result is None

    def test_frame_count_increments(self):
        session = RecordingSession()
        session.setup()
        session.start_recording()
        session.process_frame()
        session.process_frame()
        assert session.frame_count == 2
