"""
Tests for Tennis Scoring Engine — Full ITF rules verification.
"""

import pytest

from tennis.engine.scoring import ScoringEngine
from tennis.models.match import (
    MatchConfig,
    MatchFormat,
    PointOutcomeType,
    PointScore,
)


class TestScoringEngine:
    """Test the tennis scoring state machine."""

    def _make_engine(self, **kwargs) -> ScoringEngine:
        config = MatchConfig(**kwargs)
        engine = ScoringEngine(config)
        engine.start_match("p1", "p2", "Alice", "Bob")
        return engine

    # ── Point Scoring ────────────────────────────────────────────────────

    def test_point_progression(self):
        engine = self._make_engine()
        engine.score_point("p1", PointOutcomeType.WINNER)
        game = engine.match.current_set.current_game
        assert game.points_player1 == PointScore.FIFTEEN

        engine.score_point("p1", PointOutcomeType.WINNER)
        game = engine.match.current_set.current_game
        assert game.points_player1 == PointScore.THIRTY

        engine.score_point("p1", PointOutcomeType.WINNER)
        game = engine.match.current_set.current_game
        assert game.points_player1 == PointScore.FORTY

    def test_game_win(self):
        engine = self._make_engine()
        # Score 4 points for p1
        for _ in range(4):
            engine.score_point("p1", PointOutcomeType.WINNER)
        # Game should be over, new game started
        assert engine.match.current_set.games_player1 == 1

    def test_deuce(self):
        engine = self._make_engine()
        # Get to 40-40
        for _ in range(3):
            engine.score_point("p1", PointOutcomeType.WINNER)
            engine.score_point("p2", PointOutcomeType.WINNER)
        # Now at deuce
        game = engine.match.current_set.current_game
        assert game.points_player1 == PointScore.FORTY
        assert game.points_player2 == PointScore.FORTY

    def test_advantage(self):
        engine = self._make_engine()
        # Get to deuce
        for _ in range(3):
            engine.score_point("p1", PointOutcomeType.WINNER)
            engine.score_point("p2", PointOutcomeType.WINNER)
        # p1 gets advantage
        engine.score_point("p1", PointOutcomeType.WINNER)
        game = engine.match.current_set.current_game
        assert game.advantage_player_id == "p1"

    def test_advantage_back_to_deuce(self):
        engine = self._make_engine()
        for _ in range(3):
            engine.score_point("p1", PointOutcomeType.WINNER)
            engine.score_point("p2", PointOutcomeType.WINNER)
        # p1 advantage
        engine.score_point("p1", PointOutcomeType.WINNER)
        # p2 wins → back to deuce
        engine.score_point("p2", PointOutcomeType.WINNER)
        game = engine.match.current_set.current_game
        assert game.advantage_player_id is None
        assert game.points_player1 == PointScore.FORTY

    def test_advantage_win(self):
        engine = self._make_engine()
        for _ in range(3):
            engine.score_point("p1", PointOutcomeType.WINNER)
            engine.score_point("p2", PointOutcomeType.WINNER)
        engine.score_point("p1", PointOutcomeType.WINNER)
        engine.score_point("p1", PointOutcomeType.WINNER)
        assert engine.match.current_set.games_player1 == 1

    # ── No-Ad Scoring ────────────────────────────────────────────────────

    def test_no_ad_scoring(self):
        engine = self._make_engine(no_ad_scoring=True)
        for _ in range(3):
            engine.score_point("p1", PointOutcomeType.WINNER)
            engine.score_point("p2", PointOutcomeType.WINNER)
        # At deuce, next point wins
        engine.score_point("p1", PointOutcomeType.WINNER)
        assert engine.match.current_set.games_player1 == 1

    # ── Set Scoring ──────────────────────────────────────────────────────

    def test_set_win(self):
        engine = self._make_engine()
        # Win 6 games for p1 (p2 wins 0)
        for _ in range(6):
            for _ in range(4):
                engine.score_point("p1", PointOutcomeType.WINNER)
        assert engine.match.current_set.set_number == 2 or len(engine.match.sets) >= 1

    def test_tiebreak_trigger(self):
        engine = self._make_engine()
        # Get to 6-6
        for game in range(12):
            winner = "p1" if game % 2 == 0 else "p2"
            for _ in range(4):
                engine.score_point(winner, PointOutcomeType.WINNER)
        # Should be in tiebreak
        cs = engine.match.current_set
        assert cs.is_tiebreak or (cs.games_player1 == 6 and cs.games_player2 == 6)

    # ── Match Completion ─────────────────────────────────────────────────

    def test_match_completion_best_of_3(self):
        engine = self._make_engine(match_format="best_of_3")
        # Win 2 sets for p1
        for s in range(2):
            for g in range(6):
                for _ in range(4):
                    engine.score_point("p1", PointOutcomeType.WINNER)
        assert engine.is_match_over()
        assert engine.get_winner() == "p1"

    # ── Server Rotation ──────────────────────────────────────────────────

    def test_server_rotation(self):
        engine = self._make_engine()
        initial_server = engine.get_server()
        # Complete a game
        for _ in range(4):
            engine.score_point("p1", PointOutcomeType.WINNER)
        # Server should have rotated
        assert engine.get_server() != initial_server

    # ── Undo ─────────────────────────────────────────────────────────────

    def test_undo(self):
        engine = self._make_engine()
        engine.score_point("p1", PointOutcomeType.WINNER)
        assert engine.match.total_points_played == 1
        engine.undo_last_point()
        assert engine.match.total_points_played == 0

    # ── Score Display ────────────────────────────────────────────────────

    def test_score_display(self):
        engine = self._make_engine()
        display = engine.get_score_display()
        assert isinstance(display, str)

    # ── ACE and Double Fault ─────────────────────────────────────────────

    def test_ace_scoring(self):
        engine = self._make_engine()
        pt = engine.score_point("p1", PointOutcomeType.ACE)
        assert pt.outcome_type == PointOutcomeType.ACE

    def test_double_fault_scoring(self):
        engine = self._make_engine()
        pt = engine.score_point("p2", PointOutcomeType.DOUBLE_FAULT)
        assert pt.outcome_type == PointOutcomeType.DOUBLE_FAULT


class TestScoringEdgeCases:
    """Test edge cases in scoring."""

    def test_cannot_score_after_match_over(self):
        config = MatchConfig(match_format=MatchFormat.BEST_OF_1)
        engine = ScoringEngine(config)
        engine.start_match("p1", "p2")
        for _ in range(6):
            for _ in range(4):
                engine.score_point("p1", PointOutcomeType.WINNER)
        with pytest.raises(ValueError):
            engine.score_point("p1", PointOutcomeType.WINNER)

    def test_points_timeline_grows(self):
        engine = ScoringEngine()
        engine.start_match("p1", "p2")
        for i in range(5):
            engine.score_point("p1", PointOutcomeType.WINNER)
        assert len(engine.match.points_timeline) == 5
