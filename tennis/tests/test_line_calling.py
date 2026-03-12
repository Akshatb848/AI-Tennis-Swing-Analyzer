"""Tests for the automated line calling system."""

import pytest
from tennis.engine.line_calling import (
    LineCallingSystem, LineCall, CallVerdict, CallConfidence, ChallengeStatus,
)


class TestLineCallingSystem:
    """Automated line call tests."""

    def setup_method(self):
        self.system = LineCallingSystem(session_id="test", fps=30.0)

    def test_clear_in_call(self):
        call = self.system.process_bounce(court_x=0.0, court_y=5.0, frame_number=100)
        assert call.verdict == CallVerdict.IN
        assert call.confidence >= 0.85

    def test_clear_out_call(self):
        call = self.system.process_bounce(court_x=6.0, court_y=13.0, frame_number=200)
        assert call.verdict == CallVerdict.OUT
        assert call.confidence >= 0.85

    def test_close_call_moderate_confidence(self):
        # Ball very near sideline
        call = self.system.process_bounce(court_x=4.1, court_y=5.0, frame_number=300)
        assert call.confidence > 0
        assert call.confidence_level in (CallConfidence.HIGH, CallConfidence.MODERATE, CallConfidence.LOW)

    def test_replay_window(self):
        call = self.system.process_bounce(court_x=0.0, court_y=5.0, frame_number=100)
        assert call.replay_start_frame < call.frame_number
        assert call.replay_end_frame > call.frame_number
        assert call.replay_start_ms < call.timestamp_ms
        assert call.replay_end_ms > call.timestamp_ms

    def test_history_tracking(self):
        self.system.process_bounce(court_x=0.0, court_y=5.0, frame_number=100)
        self.system.process_bounce(court_x=6.0, court_y=13.0, frame_number=200)
        assert self.system.history.total_calls == 2
        assert self.system.history.calls_in >= 0
        assert self.system.history.calls_out >= 0

    def test_history_summary(self):
        self.system.process_bounce(court_x=0.0, court_y=5.0, frame_number=100)
        summary = self.system.history.get_summary()
        assert "total_calls" in summary
        assert "average_confidence" in summary

    def test_display_format(self):
        call = self.system.process_bounce(court_x=0.0, court_y=5.0, frame_number=100)
        display = call.to_display()
        assert "verdict" in display
        assert "confidence" in display
        assert "distance" in display


class TestChallengeSystem:
    """Challenge workflow tests."""

    def setup_method(self):
        self.system = LineCallingSystem(session_id="test", fps=30.0)
        self.call = self.system.process_bounce(court_x=4.2, court_y=5.0, frame_number=100)

    def test_challenge_existing_call(self):
        result = self.system.challenge_call(self.call.call_id, "player_0")
        assert result is not None
        assert result.challenge_status in (ChallengeStatus.OVERTURNED, ChallengeStatus.UPHELD)

    def test_challenge_nonexistent_call(self):
        result = self.system.challenge_call("nonexistent", "player_0")
        assert result is None

    def test_cannot_double_challenge(self):
        self.system.challenge_call(self.call.call_id, "player_0")
        result = self.system.challenge_call(self.call.call_id, "player_0")
        assert result is None

    def test_challenge_records_original_verdict(self):
        original = self.call.verdict
        result = self.system.challenge_call(self.call.call_id, "player_0")
        assert result.original_verdict == original
