"""
Stats Calculator — Real-time statistical computation from match events.

Computes per-player and per-match statistics including:
- Serve percentages and speeds
- Winner/error ratios
- Rally length distributions
- Shot placement heatmaps
- Break point conversion rates
"""

from __future__ import annotations
from typing import Optional
from tennis.models.match import MatchState, PointOutcome, PointOutcomeType, ShotType
from tennis.models.player import PlayerSessionStats, PlayerComparison
from tennis.models.events import RallyEvent


class StatsCalculator:
    """Computes match and player statistics from point-level data."""

    def compute_player_stats(
        self,
        player_id: str,
        match: MatchState,
        rallies: Optional[list[RallyEvent]] = None,
    ) -> PlayerSessionStats:
        stats = PlayerSessionStats(player_id=player_id, session_id=match.id, match_id=match.id)
        for pt in match.points_timeline:
            self._process_point(stats, pt, player_id)
        self._finalize(stats)
        if rallies:
            self._process_rallies(stats, rallies, player_id)
        return stats

    def compute_match_comparison(self, match: MatchState) -> PlayerComparison:
        s1 = self.compute_player_stats(match.player1_id, match)
        s2 = self.compute_player_stats(match.player2_id, match)
        highlights = self._find_highlight_stats(s1, s2)
        return PlayerComparison(
            player1=s1, player2=s2,
            player1_name=match.player1_name, player2_name=match.player2_name,
            highlight_stats=highlights,
        )

    def _process_point(self, stats: PlayerSessionStats, pt: PointOutcome, pid: str) -> None:
        stats.total_points += 1
        is_winner = pt.winner_id == pid
        is_server = pt.server_id == pid

        if is_winner:
            stats.points_won += 1

        # Serve stats
        if is_server:
            stats.total_serve_points += 1
            if pt.outcome_type == PointOutcomeType.ACE:
                stats.aces += 1
                stats.first_serves_in += 1
                if is_winner:
                    stats.points_won += 0  # already counted
            elif pt.outcome_type == PointOutcomeType.DOUBLE_FAULT:
                stats.double_faults += 1
            else:
                stats.first_serves_in += 1
        else:
            stats.total_return_points += 1
            if is_winner:
                stats.return_points_won += 1

        # Winner/error classification
        if is_winner and pt.outcome_type == PointOutcomeType.WINNER:
            stats.winners += 1
            if pt.last_shot and pt.last_shot.shot_type in (ShotType.FOREHAND, ShotType.SLICE_FH):
                stats.forehand_winners += 1
            elif pt.last_shot and pt.last_shot.shot_type in (ShotType.BACKHAND, ShotType.SLICE_BH):
                stats.backhand_winners += 1
        elif not is_winner and pt.outcome_type == PointOutcomeType.UNFORCED_ERROR:
            stats.unforced_errors += 1
            if pt.last_shot and "forehand" in pt.last_shot.shot_type.value:
                stats.forehand_errors += 1
            elif pt.last_shot and "backhand" in pt.last_shot.shot_type.value:
                stats.backhand_errors += 1
        elif not is_winner and pt.outcome_type == PointOutcomeType.FORCED_ERROR:
            stats.forced_errors += 1

        # Rally length
        stats.avg_rally_length = (
            (stats.avg_rally_length * (stats.total_points - 1) + pt.rally_length)
            / stats.total_points
        )
        stats.longest_rally = max(stats.longest_rally, pt.rally_length)

        # Break points
        if pt.is_break_point:
            if is_server:
                stats.break_points_faced += 1
                if is_winner:
                    stats.break_points_saved += 1
            else:
                stats.break_points_total += 1
                if is_winner:
                    stats.break_points_won += 1

    def _process_rallies(self, stats: PlayerSessionStats, rallies: list[RallyEvent], pid: str) -> None:
        speeds = []
        for r in rallies:
            for shot_d in r.shots:
                if shot_d.get("player_id") == pid and shot_d.get("speed_mph"):
                    speeds.append(shot_d["speed_mph"])
        if speeds:
            stats.max_serve_speed_mph = max(speeds)
            stats.avg_first_serve_speed_mph = sum(speeds) / len(speeds)

    def _finalize(self, stats: PlayerSessionStats) -> None:
        if stats.total_serve_points > 0:
            stats.first_serve_pct = stats.first_serves_in / stats.total_serve_points * 100
        if stats.total_return_points > 0:
            stats.return_win_pct = stats.return_points_won / stats.total_return_points * 100
        if stats.unforced_errors > 0:
            stats.winner_to_ue_ratio = stats.winners / stats.unforced_errors
        if stats.break_points_total > 0:
            stats.break_point_conversion_pct = stats.break_points_won / stats.break_points_total * 100
        if stats.break_points_faced > 0:
            stats.break_point_save_pct = stats.break_points_saved / stats.break_points_faced * 100

    def _find_highlight_stats(self, s1: PlayerSessionStats, s2: PlayerSessionStats) -> list[str]:
        highlights = []
        comparisons = [
            ("aces", s1.aces, s2.aces),
            ("winners", s1.winners, s2.winners),
            ("first_serve_pct", s1.first_serve_pct, s2.first_serve_pct),
            ("unforced_errors", s1.unforced_errors, s2.unforced_errors),
            ("break_point_conversion", s1.break_point_conversion_pct, s2.break_point_conversion_pct),
        ]
        for name, v1, v2 in comparisons:
            if v1 != v2:
                diff = abs(v1 - v2) / max(v1, v2, 1) * 100
                if diff > 20:
                    highlights.append(name)
        return highlights

    def generate_placement_heatmap(
        self, points: list[PointOutcome], player_id: str
    ) -> dict[str, int]:
        """Generate shot placement heatmap data by court zone."""
        heatmap: dict[str, int] = {}
        for pt in points:
            for shot in pt.shot_sequence:
                if shot.player_id == player_id and shot.placement_zone:
                    zone = shot.placement_zone.value
                    heatmap[zone] = heatmap.get(zone, 0) + 1
        return heatmap

    def generate_speed_distribution(
        self, points: list[PointOutcome], player_id: str
    ) -> list[float]:
        """Get distribution of shot speeds for a player."""
        speeds = []
        for pt in points:
            for shot in pt.shot_sequence:
                if shot.player_id == player_id and shot.speed_mph:
                    speeds.append(shot.speed_mph)
        return sorted(speeds)

    # ── Advanced Analytics ────────────────────────────────────────────────────

    def generate_shot_distribution(
        self, points: list[PointOutcome], player_id: str
    ) -> dict:
        """Shot type distribution with counts, percentages, and success rates."""
        shot_counts: dict[str, int] = {}
        shot_wins: dict[str, int] = {}
        total = 0

        for pt in points:
            for shot in pt.shot_sequence:
                if shot.player_id == player_id:
                    st = shot.shot_type.value
                    shot_counts[st] = shot_counts.get(st, 0) + 1
                    total += 1
            # Count wins by last shot type
            if pt.winner_id == player_id and pt.last_shot and pt.last_shot.player_id == player_id:
                st = pt.last_shot.shot_type.value
                shot_wins[st] = shot_wins.get(st, 0) + 1

        distribution = {}
        for st, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
            distribution[st] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total else 0,
                "wins": shot_wins.get(st, 0),
                "success_rate": round(shot_wins.get(st, 0) / count * 100, 1) if count else 0,
            }
        return {"shots": distribution, "total_shots": total}

    def generate_rally_analytics(
        self, points: list[PointOutcome], player_id: str
    ) -> dict:
        """Rally length analytics with distribution and win rates."""
        lengths = []
        wins_by_length: dict[str, list[int]] = {"short": [], "medium": [], "long": []}

        for pt in points:
            rl = pt.rally_length
            lengths.append(rl)
            bucket = "short" if rl <= 4 else ("medium" if rl <= 9 else "long")
            is_win = 1 if pt.winner_id == player_id else 0
            wins_by_length[bucket].append(is_win)

        # Length distribution histogram (buckets of 3)
        distribution: dict[str, int] = {}
        for length in lengths:
            bucket_label = f"{(length // 3) * 3}-{(length // 3) * 3 + 2}"
            distribution[bucket_label] = distribution.get(bucket_label, 0) + 1

        total = len(lengths)
        short = sum(1 for l in lengths if l <= 4)
        medium = sum(1 for l in lengths if 4 < l <= 9)
        long_ = sum(1 for l in lengths if l > 9)

        win_rates = {}
        for bucket, results in wins_by_length.items():
            if results:
                win_rates[bucket] = round(sum(results) / len(results) * 100, 1)
            else:
                win_rates[bucket] = 0.0

        return {
            "total_rallies": total,
            "avg_length": round(sum(lengths) / total, 1) if total else 0,
            "max_length": max(lengths) if lengths else 0,
            "distribution": distribution,
            "short_pct": round(short / total * 100, 1) if total else 0,
            "medium_pct": round(medium / total * 100, 1) if total else 0,
            "long_pct": round(long_ / total * 100, 1) if total else 0,
            "win_rate_by_length": win_rates,
        }

    def generate_momentum_analysis(
        self, points: list[PointOutcome], player_id: str
    ) -> dict:
        """Momentum analysis with rolling win percentage and shift detection."""
        if not points:
            return {"rolling_win_pct": [], "momentum_shifts": [], "consistency_score": 0}

        window_size = max(5, len(points) // 10)
        rolling_pct: list[dict] = []
        results = [1 if pt.winner_id == player_id else 0 for pt in points]

        for i in range(len(results)):
            start = max(0, i - window_size + 1)
            window = results[start:i + 1]
            pct = sum(window) / len(window) * 100
            rolling_pct.append({
                "point": i + 1,
                "win_pct": round(pct, 1),
                "timestamp_ms": points[i].timestamp_start_ms,
            })

        # Detect momentum shifts (> 20% change in rolling win%)
        shifts = []
        for i in range(1, len(rolling_pct)):
            delta = rolling_pct[i]["win_pct"] - rolling_pct[i - 1]["win_pct"]
            if abs(delta) > 20:
                direction = "positive" if delta > 0 else "negative"
                shifts.append({
                    "point": rolling_pct[i]["point"],
                    "direction": direction,
                    "magnitude": round(abs(delta), 1),
                })

        # Consistency score: inverse of variance in rolling win%
        if len(rolling_pct) > 2:
            pcts = [r["win_pct"] for r in rolling_pct]
            mean_pct = sum(pcts) / len(pcts)
            variance = sum((p - mean_pct) ** 2 for p in pcts) / len(pcts)
            consistency = max(0, round(1.0 - variance / 2500, 2))  # normalize
        else:
            consistency = 0.5

        return {
            "rolling_win_pct": rolling_pct,
            "momentum_shifts": shifts,
            "consistency_score": consistency,
        }

    def generate_fatigue_indicators(
        self, points: list[PointOutcome], player_id: str
    ) -> dict:
        """Fatigue analysis by match phase."""
        if len(points) < 6:
            return {"phases": [], "fatigue_detected": False}

        third = len(points) // 3
        phases = [
            ("early", points[:third]),
            ("middle", points[third: 2 * third]),
            ("late", points[2 * third:]),
        ]

        phase_data = []
        for name, phase_points in phases:
            if not phase_points:
                continue
            wins = sum(1 for p in phase_points if p.winner_id == player_id)
            errors = sum(1 for p in phase_points if p.outcome_type == PointOutcomeType.UNFORCED_ERROR and p.winner_id != player_id)
            avg_rally = sum(p.rally_length for p in phase_points) / len(phase_points)
            speeds = []
            for p in phase_points:
                for s in p.shot_sequence:
                    if s.player_id == player_id and s.speed_mph:
                        speeds.append(s.speed_mph)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0

            phase_data.append({
                "phase": name,
                "points": len(phase_points),
                "win_rate": round(wins / len(phase_points) * 100, 1),
                "unforced_errors": errors,
                "avg_rally_length": round(avg_rally, 1),
                "avg_speed_mph": round(avg_speed, 1),
            })

        # Detect fatigue: late-phase win rate < early by > 15%
        fatigue = False
        if len(phase_data) >= 2:
            early_wr = phase_data[0]["win_rate"]
            late_wr = phase_data[-1]["win_rate"]
            if early_wr > 0 and (early_wr - late_wr) / early_wr > 0.15:
                fatigue = True

        return {"phases": phase_data, "fatigue_detected": fatigue}

    def generate_improvement_areas(
        self, points: list[PointOutcome], player_id: str
    ) -> list[dict]:
        """Identify weakest shot types and court zones for improvement."""
        areas = []

        # Shot-level analysis
        shot_dist = self.generate_shot_distribution(points, player_id)
        for st, data in shot_dist.get("shots", {}).items():
            if data["count"] >= 3 and data["success_rate"] < 40:
                areas.append({
                    "area": f"{st} shot effectiveness",
                    "metric": f"{data['success_rate']}% success rate ({data['count']} attempts)",
                    "recommendation": f"Focus on {st} consistency — practice placement over power",
                    "priority": 0.8 if data["success_rate"] < 20 else 0.6,
                })

        # Zone-level analysis
        heatmap = self.generate_placement_heatmap(points, player_id)
        total_shots = sum(heatmap.values())
        if total_shots > 0:
            # Find underused zones
            expected_pct = 100 / max(len(heatmap), 1)
            for zone, count in heatmap.items():
                pct = count / total_shots * 100
                if pct > expected_pct * 2.5:
                    areas.append({
                        "area": f"Over-reliance on {zone} placement",
                        "metric": f"{pct:.0f}% of shots to {zone} (expected ~{expected_pct:.0f}%)",
                        "recommendation": f"Add variety — distribute shots more evenly to keep opponent guessing",
                        "priority": 0.5,
                    })

        return sorted(areas, key=lambda x: -x["priority"])
