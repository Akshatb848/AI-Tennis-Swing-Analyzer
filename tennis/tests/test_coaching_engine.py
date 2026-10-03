"""
Tests for Coaching Engine — player embedding, flaw detection, feedback.
"""

from tennis.engine.coaching_engine import CoachingEngine
from tennis.models.player import PlayerProfile, PlayerSessionStats


class TestCoachingEngine:
    """Test coaching engine functionality."""

    def setup_method(self):
        self.coaching = CoachingEngine()
        self.player = PlayerProfile(id="p1", name="Test Player")
        self.session_stats = PlayerSessionStats(
            player_id="p1", session_id="s1",
            total_points=40, points_won=22,
            first_serve_pct=65.0, second_serve_pct=85.0,
            winners=14, unforced_errors=10,
            aces=4, double_faults=2,
            net_points_won=6, net_points_total=10,
            forehand_winners=8, backhand_winners=4,
        )

    def test_build_style_embedding(self):
        emb = self.coaching.build_style_embedding(self.player, [self.session_stats])
        assert emb is not None
        assert len(emb.embedding) == 64
        assert all(-1 <= v <= 1 for v in emb.embedding)

    def test_analyze_swing(self):
        analysis = self.coaching.analyze_swing(
            "s1", "p1", [{"mock": True}], "forehand"
        )
        assert analysis is not None
        assert analysis.shot_type == "forehand"

    def test_generate_feedback(self):
        swings = [self.coaching.analyze_swing("s1", "p1", [{}], "forehand")]
        feedback = self.coaching.generate_feedback(
            self.player, self.session_stats, swings
        )
        assert feedback is not None
        assert feedback.performance_rating > 0
        assert len(feedback.strengths) > 0

    def test_generate_feedback_with_embedding(self):
        emb = self.coaching.build_style_embedding(self.player, [self.session_stats])
        swings = [self.coaching.analyze_swing("s1", "p1", [{}], "forehand")]
        feedback = self.coaching.generate_feedback(
            self.player, self.session_stats, swings, style_embedding=emb
        )
        assert feedback is not None

    def test_generate_weekly_goal(self):
        swings = [self.coaching.analyze_swing("s1", "p1", [{}], "forehand")]
        feedback = self.coaching.generate_feedback(
            self.player, self.session_stats, swings
        )
        goal = self.coaching.generate_weekly_goal(self.player, [feedback])
        assert goal is not None
        assert goal.primary_goal != ""
        assert goal.sessions_target > 0

    def test_embedding_components(self):
        emb = self.coaching.build_style_embedding(self.player, [self.session_stats])
        assert emb.aggression_score >= 0
        assert emb.consistency_score >= 0
        assert emb.sessions_analyzed >= 1

    def test_performance_rating_scales(self):
        """Higher winner/error ratio should give higher rating."""
        good_stats = PlayerSessionStats(
            player_id="p1", session_id="s1",
            total_points=40, points_won=30,
            first_serve_pct=75.0, winners=20, unforced_errors=5,
            aces=6,
        )
        bad_stats = PlayerSessionStats(
            player_id="p1", session_id="s2",
            total_points=40, points_won=12,
            first_serve_pct=45.0, winners=5, unforced_errors=18,
            aces=0, double_faults=6,
        )
        swings = [self.coaching.analyze_swing("s1", "p1", [{}], "forehand")]
        fb_good = self.coaching.generate_feedback(self.player, good_stats, swings)
        fb_bad = self.coaching.generate_feedback(self.player, bad_stats, swings)
        assert fb_good.performance_rating > fb_bad.performance_rating
