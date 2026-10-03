"""
Tests for Event Processor — CV event → tennis semantics verification.
"""

from tennis.engine.event_processor import CourtGeometry, EventProcessor
from tennis.models.events import BallEvent, BoundingBox, EventType, Point2D


class TestCourtGeometry:
    """Test ITF court geometry calculations."""

    def setup_method(self):
        self.court = CourtGeometry(is_doubles=False)

    def test_ball_in_center(self):
        pos = Point2D(x=0.0, y=0.0)
        is_in, dist, line = self.court.is_ball_in(pos)
        assert is_in is True
        assert dist > 0

    def test_ball_clearly_out(self):
        pos = Point2D(x=10.0, y=0.0)  # Way past sideline (4.115m)
        is_in, dist, line = self.court.is_ball_in(pos)
        assert is_in is False

    def test_ball_on_baseline(self):
        pos = Point2D(x=0.0, y=11.885)  # Exactly on baseline
        is_in, dist, line = self.court.is_ball_in(pos)
        assert is_in is True  # On the line = in

    def test_ball_just_out_baseline(self):
        pos = Point2D(x=0.0, y=12.0)  # Just past baseline
        is_in, dist, line = self.court.is_ball_in(pos)
        assert is_in is False

    def test_ball_on_sideline(self):
        pos = Point2D(x=4.115, y=0.0)  # On singles sideline
        is_in, dist, line = self.court.is_ball_in(pos)
        assert is_in is True

    def test_doubles_wider_court(self):
        doubles = CourtGeometry(is_doubles=True)
        pos = Point2D(x=5.0, y=0.0)  # In doubles, out of singles
        is_in, _, _ = doubles.is_ball_in(pos)
        assert is_in is True  # Within doubles sideline (5.485m)

    def test_serve_in_deuce(self):
        pos = Point2D(x=2.0, y=3.0)  # In deuce service box
        is_in, _, _ = self.court.is_ball_in(pos, is_serve=True, serve_side="deuce")
        assert is_in is True

    def test_serve_out_wrong_box(self):
        pos = Point2D(x=-2.0, y=3.0)  # In ad box, not deuce
        is_in, _, _ = self.court.is_ball_in(pos, is_serve=True, serve_side="deuce")
        assert is_in is False

    def test_court_zone_baseline(self):
        pos = Point2D(x=0.0, y=10.0)
        zone = self.court.get_court_zone(pos)
        assert "baseline" in zone.value

    def test_court_zone_net(self):
        pos = Point2D(x=0.0, y=1.0)
        zone = self.court.get_court_zone(pos)
        assert "net" in zone.value


class TestEventProcessor:
    """Test the event processing pipeline."""

    def setup_method(self):
        self.processor = EventProcessor(is_doubles=False)

    def test_process_bounce_in(self):
        event = BallEvent(
            event_type=EventType.BALL_BOUNCE,
            timestamp_ms=1000, frame_number=30,
            session_id="test", position_court=Point2D(x=0, y=5),
            position_image=BoundingBox(x1=100, y1=100, x2=110, y2=110, confidence=0.9),
        )
        result = self.processor.process_ball_event(event)
        assert result is not None
        assert "line_call" in result
        assert result["line_call"].verdict.value == "in"

    def test_process_bounce_out(self):
        event = BallEvent(
            event_type=EventType.BALL_BOUNCE,
            timestamp_ms=1000, frame_number=30,
            session_id="test", position_court=Point2D(x=10, y=0),
        )
        result = self.processor.process_ball_event(event)
        assert result is not None
        assert result["line_call"].verdict.value == "out"

    def test_confidence_far_from_line(self):
        # Use a point far from all lines (center of baseline half)
        event = BallEvent(
            event_type=EventType.BALL_BOUNCE,
            timestamp_ms=1000, frame_number=30,
            session_id="test", position_court=Point2D(x=0, y=6.0),
        )
        result = self.processor.process_ball_event(event)
        assert result["line_call"].confidence >= 0.5  # Reasonable confidence

    def test_confidence_close_to_line(self):
        event = BallEvent(
            event_type=EventType.BALL_BOUNCE,
            timestamp_ms=1000, frame_number=30,
            session_id="test", position_court=Point2D(x=4.10, y=0),  # Very close to sideline
        )
        result = self.processor.process_ball_event(event)
        # Should have lower confidence since it's close to the line
        assert result["line_call"].confidence < 0.95
