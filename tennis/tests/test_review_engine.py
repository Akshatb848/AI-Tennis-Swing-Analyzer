"""Tests for the post-match review engine."""

import pytest
from tennis.engine.review_engine import ReviewEngine


class TestReviewEngine:
    """Review engine tests — factual observations only."""

    def setup_method(self):
        self.engine = ReviewEngine()
        self.session_summary = {
            "session_id": "test-session",
            "match_type": "singles",
            "duration_seconds": 5400.0,
            "players": ["Alice", "Bob"],
            "points": [
                {"number": i, "rally_length": rl, "winner": "p0" if i % 3 != 0 else "p1", "outcome": "winner"}
                for i, rl in enumerate([2, 5, 8, 1, 3, 6, 11, 4, 2, 7, 3, 9, 1, 5, 14], 1)
            ],
            "line_calls": [
                {"verdict": "in", "confidence": 0.92},
                {"verdict": "out", "confidence": 0.88},
                {"verdict": "in", "confidence": 0.95},
            ],
        }

    def test_match_review_structure(self):
        review = self.engine.analyze_match(self.session_summary)
        assert review.session_id == "test-session"
        assert review.match_type == "singles"
        assert review.total_points == 15
        assert len(review.player_reviews) == 2
        assert review.rally_breakdown is not None

    def test_rally_breakdown(self):
        review = self.engine.analyze_match(self.session_summary)
        rb = review.rally_breakdown
        assert rb.total_rallies == 15
        assert rb.avg_rally_length > 0
        assert rb.max_rally_length == 14
        assert "1-3" in rb.rally_length_distribution

    def test_player_reviews_populated(self):
        review = self.engine.analyze_match(self.session_summary)
        for pr in review.player_reviews:
            assert pr.player_name in ("Alice", "Bob")
            assert pr.total_points_played == 15
            assert pr.points_won >= 0

    def test_line_call_summary(self):
        review = self.engine.analyze_match(self.session_summary)
        lc = review.line_call_summary
        assert lc["total"] == 3
        assert lc["in"] == 2
        assert lc["out"] == 1

    def test_observations_are_factual(self):
        review = self.engine.analyze_match(self.session_summary)
        for obs in review.observations:
            # No motivational language
            assert "great" not in obs.lower()
            assert "amazing" not in obs.lower()
            assert "awesome" not in obs.lower()
            assert "!" not in obs

    def test_corrections_are_actionable(self):
        review = self.engine.analyze_match(
            self.session_summary,
            player_stats={
                "p0": {"unforced_errors": 20, "winners": 10, "net_points_total": 5, "net_points_won": 1},
                "p1": {"unforced_errors": 8, "winners": 15},
            },
        )
        p0 = review.player_reviews[0]
        assert len(p0.corrections) <= 3
        for correction in p0.corrections:
            assert isinstance(correction, str)
            assert len(correction) > 10

    def test_empty_match(self):
        empty = {"session_id": "empty", "match_type": "singles", "duration_seconds": 0, "players": [], "points": [], "line_calls": []}
        review = self.engine.analyze_match(empty)
        assert review.total_points == 0
        assert len(review.player_reviews) == 0
