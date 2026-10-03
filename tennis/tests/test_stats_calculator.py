"""
Tests for Stats Calculator — stat computation verification.
"""

from tennis.engine.scoring import ScoringEngine
from tennis.engine.stats_calculator import StatsCalculator
from tennis.models.match import MatchConfig, PointOutcomeType


class TestStatsCalculator:
    """Test stat computation from match data."""

    def _play_sample_match(self) -> ScoringEngine:
        """Create a sample match with varied point types."""
        engine = ScoringEngine(MatchConfig())
        engine.start_match("p1", "p2", "Alice", "Bob")

        outcomes = [
            ("p1", PointOutcomeType.ACE),
            ("p1", PointOutcomeType.WINNER),
            ("p2", PointOutcomeType.WINNER),
            ("p1", PointOutcomeType.WINNER),
            ("p1", PointOutcomeType.WINNER),  # p1 wins game 1
            ("p2", PointOutcomeType.ACE),
            ("p2", PointOutcomeType.WINNER),
            ("p1", PointOutcomeType.UNFORCED_ERROR),
            ("p2", PointOutcomeType.WINNER),
            ("p2", PointOutcomeType.WINNER),  # p2 wins game 2
            ("p1", PointOutcomeType.WINNER),
            ("p1", PointOutcomeType.ACE),
            ("p1", PointOutcomeType.WINNER),
            ("p2", PointOutcomeType.DOUBLE_FAULT),
        ]
        for winner, otype in outcomes:
            try:
                engine.score_point(winner, otype)
            except ValueError:
                break
        return engine

    def test_compute_player_stats(self):
        engine = self._play_sample_match()
        calc = StatsCalculator()
        stats = calc.compute_player_stats("p1", engine.get_match_state())
        assert stats is not None
        assert stats.aces >= 2
        assert stats.winners >= 3
        assert stats.total_points > 0

    def test_compute_match_comparison(self):
        engine = self._play_sample_match()
        calc = StatsCalculator()
        comp = calc.compute_match_comparison(engine.get_match_state())
        assert comp is not None
        assert comp.player1 is not None
        assert comp.player2 is not None

    def test_empty_match_stats(self):
        engine = ScoringEngine()
        engine.start_match("p1", "p2")
        calc = StatsCalculator()
        stats = calc.compute_player_stats("p1", engine.get_match_state())
        assert stats.total_points == 0

    def test_placement_heatmap(self):
        engine = self._play_sample_match()
        calc = StatsCalculator()
        heatmap = calc.generate_placement_heatmap(engine.get_match_state().points_timeline, "p1")
        assert isinstance(heatmap, dict)

    def test_speed_distribution(self):
        engine = self._play_sample_match()
        calc = StatsCalculator()
        speeds = calc.generate_speed_distribution(engine.get_match_state().points_timeline, "p1")
        assert isinstance(speeds, list)
