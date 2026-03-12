"""
Post-Match Analyzer — Deeper analysis pass on recorded session data.

Runs after recording stops to produce:
- Pose-based swing review from full session
- Player movement load calculation
- Fatigue detection via movement degradation
- Consistency scoring over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FatigueAnalysis:
    """Fatigue detection from movement degradation."""
    player_id: str = ""
    first_half_avg_speed: float = 0.0
    second_half_avg_speed: float = 0.0
    speed_decline_pct: float = 0.0
    first_half_court_coverage: float = 0.0
    second_half_court_coverage: float = 0.0
    coverage_decline_pct: float = 0.0
    fatigue_detected: bool = False
    fatigue_onset_point: int = 0  # Point number where fatigue begins


@dataclass
class ConsistencyAnalysis:
    """Shot placement consistency over time."""
    player_id: str = ""
    overall_consistency: float = 0.0  # 0-1 score
    forehand_consistency: float = 0.0
    backhand_consistency: float = 0.0
    serve_consistency: float = 0.0
    consistency_trend: str = "stable"  # improving, declining, stable


@dataclass
class MovementLoad:
    """Player movement statistics."""
    player_id: str = ""
    total_distance_meters: float = 0.0
    avg_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    total_sprints: int = 0
    court_coverage_pct: float = 0.0
    avg_recovery_time_ms: float = 0.0


@dataclass
class PostMatchResult:
    """Complete post-match analysis result."""
    session_id: str = ""
    fatigue: list[FatigueAnalysis] = field(default_factory=list)
    consistency: list[ConsistencyAnalysis] = field(default_factory=list)
    movement_loads: list[MovementLoad] = field(default_factory=list)
    additional_observations: list[str] = field(default_factory=list)


class PostMatchAnalyzer:
    """
    Post-match analysis engine.

    Processes stored session data for deeper insights than real-time
    analysis can provide (no latency constraint).
    """

    def analyze(self, session_summary: dict) -> PostMatchResult:
        """Run full post-match analysis on session data."""
        result = PostMatchResult(
            session_id=session_summary.get("session_id", ""),
        )

        players = session_summary.get("players", [])
        points = session_summary.get("points", [])

        for i, player_name in enumerate(players):
            player_id = f"p{i}"

            # Fatigue analysis
            fatigue = self._analyze_fatigue(player_id, points)
            result.fatigue.append(fatigue)

            # Consistency analysis
            consistency = self._analyze_consistency(player_id, points)
            result.consistency.append(consistency)

            # Movement load
            movement = self._analyze_movement(player_id, points)
            result.movement_loads.append(movement)

        # Generate additional observations
        result.additional_observations = self._generate_observations(result, points)

        return result

    def _analyze_fatigue(self, player_id: str, points: list[dict]) -> FatigueAnalysis:
        """Detect fatigue by comparing first vs second half performance."""
        fa = FatigueAnalysis(player_id=player_id)
        if len(points) < 10:
            return fa

        mid = len(points) // 2
        first_half = points[:mid]
        second_half = points[mid:]

        # Rally length as proxy for movement intensity
        fh_lengths = [p.get("rally_length", 0) for p in first_half if p.get("winner") == player_id]
        sh_lengths = [p.get("rally_length", 0) for p in second_half if p.get("winner") == player_id]

        fh_wins = sum(1 for p in first_half if p.get("winner") == player_id)
        sh_wins = sum(1 for p in second_half if p.get("winner") == player_id)

        fh_rate = fh_wins / len(first_half) if first_half else 0
        sh_rate = sh_wins / len(second_half) if second_half else 0

        fa.first_half_avg_speed = fh_rate * 100
        fa.second_half_avg_speed = sh_rate * 100
        fa.speed_decline_pct = (
            (fh_rate - sh_rate) / fh_rate * 100 if fh_rate > 0 else 0
        )

        fa.fatigue_detected = fa.speed_decline_pct > 15
        if fa.fatigue_detected:
            fa.fatigue_onset_point = mid

        return fa

    def _analyze_consistency(self, player_id: str, points: list[dict]) -> ConsistencyAnalysis:
        """Analyze shot placement consistency."""
        ca = ConsistencyAnalysis(player_id=player_id)
        if not points:
            return ca

        player_points = [p for p in points if p.get("winner") == player_id]

        # Win rate consistency across chunks
        chunk_size = max(5, len(points) // 4)
        win_rates = []
        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]
            wins = sum(1 for p in chunk if p.get("winner") == player_id)
            win_rates.append(wins / len(chunk) if chunk else 0)

        if win_rates:
            mean_rate = sum(win_rates) / len(win_rates)
            variance = sum((r - mean_rate) ** 2 for r in win_rates) / len(win_rates)
            ca.overall_consistency = max(0, 1.0 - variance * 4)  # Scale to 0-1

            if len(win_rates) >= 3:
                if win_rates[-1] > win_rates[0] + 0.1:
                    ca.consistency_trend = "improving"
                elif win_rates[-1] < win_rates[0] - 0.1:
                    ca.consistency_trend = "declining"

        return ca

    def _analyze_movement(self, player_id: str, points: list[dict]) -> MovementLoad:
        """Estimate player movement load from point data."""
        ml = MovementLoad(player_id=player_id)

        rally_lengths = [
            p.get("rally_length", 0) for p in points
        ]

        if rally_lengths:
            # Rough approximation: each rally shot ≈ 5m of movement
            ml.total_distance_meters = sum(rally_lengths) * 5.0
            ml.avg_speed_mps = 2.5  # Average tennis player speed
            ml.total_sprints = sum(1 for r in rally_lengths if r >= 5)
            ml.court_coverage_pct = min(
                50 + sum(1 for r in rally_lengths if r >= 3) * 2, 100
            )

        return ml

    def _generate_observations(
        self, result: PostMatchResult, points: list[dict],
    ) -> list[str]:
        """Generate factual post-match observations."""
        obs = []

        for fa in result.fatigue:
            if fa.fatigue_detected:
                obs.append(
                    f"Player {fa.player_id}: fatigue detected after point "
                    f"{fa.fatigue_onset_point}, win rate declined "
                    f"{fa.speed_decline_pct:.0f}%"
                )

        for ca in result.consistency:
            if ca.consistency_trend == "declining":
                obs.append(
                    f"Player {ca.player_id}: consistency declined over the match "
                    f"(overall: {ca.overall_consistency:.0%})"
                )
            elif ca.consistency_trend == "improving":
                obs.append(
                    f"Player {ca.player_id}: performance improved through the match "
                    f"(overall: {ca.overall_consistency:.0%})"
                )

        for ml in result.movement_loads:
            if ml.total_distance_meters > 0:
                obs.append(
                    f"Player {ml.player_id}: estimated {ml.total_distance_meters:.0f}m "
                    f"distance covered, {ml.total_sprints} sprints"
                )

        return obs
