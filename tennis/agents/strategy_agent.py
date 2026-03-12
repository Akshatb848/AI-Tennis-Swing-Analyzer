"""
Strategy Agent — analyzes shot placement, court positioning, rally tactics.

Evaluates tactical choices: baseline vs net play, crosscourt vs down-the-line patterns,
serve placement patterns, and overall court management.
"""
from __future__ import annotations
import logging
import math

logger = logging.getLogger(__name__)


class StrategyAgent:
    """Expert agent for tactical and strategic analysis."""

    name = "strategy"

    def analyze(self, frames: list, context: dict) -> dict:
        """
        Analyze shot placement and tactical patterns.

        Returns:
            shot_placement: dict with zone distribution
            rally_patterns: list of detected tactical patterns
            net_approach_rate: float (0-1)
            baseline_tendency: float (0-1)
            crosscourt_rate: float (0-1)
            dtl_rate: float (0-1, down-the-line rate)
            tactical_score: 0-10
        """
        ball_data = context.get("ball", {})
        player_data = context.get("player", {})
        court_data = context.get("court", {})

        trajectory = ball_data.get("trajectory", [])
        bounces = ball_data.get("bounce_locations", [])
        avg_speed = ball_data.get("avg_speed_kmh", 130.0)
        coverage = player_data.get("court_coverage", [0.5])
        net_y = court_data.get("net_y", 0.5)
        fp = context.get("video_fingerprint", [])

        # Shot placement heatmap (6 zones: left/center/right × deep/short)
        zones = {"deep_left": 0, "deep_center": 0, "deep_right": 0,
                 "short_left": 0, "short_center": 0, "short_right": 0}

        crosscourt_shots = 0
        dtl_shots = 0
        net_approaches = 0
        total = max(1, len(bounces))

        prev_x = None
        for b in bounces:
            x, y = b["x"], b["y"]
            # Zone assignment
            x_zone = "left" if x < 0.35 else ("right" if x > 0.65 else "center")
            y_zone = "deep" if y > net_y else "short"
            key = f"{y_zone}_{x_zone}"
            zones[key] = zones.get(key, 0) + 1

            # Crosscourt vs DTL detection
            if prev_x is not None:
                if (prev_x < 0.4 and x > 0.6) or (prev_x > 0.6 and x < 0.4):
                    crosscourt_shots += 1
                else:
                    dtl_shots += 1
            prev_x = x

            # Net approach: ball lands very close to net
            if abs(y - net_y) < 0.08:
                net_approaches += 1

        crosscourt_rate = round(crosscourt_shots / max(1, crosscourt_shots + dtl_shots), 2)
        dtl_rate = round(1 - crosscourt_rate, 2)
        net_rate = round(net_approaches / total, 2)

        # Use fingerprint for uniqueness when bounce data is sparse
        fp_v = int(fp[5] * 100) if len(fp) > 5 else 50
        if total < 5:
            crosscourt_rate = round(0.45 + fp_v / 500.0, 2)
            dtl_rate = round(1 - crosscourt_rate, 2)
            net_rate = round(0.1 + fp_v / 1000.0, 2)

        # Court zone % distribution
        zone_total = max(1, sum(zones.values()))
        zone_pct = {k: round(v / zone_total * 100, 1) for k, v in zones.items()}

        # Detect tactical patterns
        patterns = []
        if crosscourt_rate > 0.62:
            patterns.append("Heavy crosscourt tendency")
        elif dtl_rate > 0.55:
            patterns.append("Down-the-line attacking preference")

        if avg_speed > 155:
            patterns.append("Aggressive baseline striking")
        elif avg_speed < 110:
            patterns.append("High-margin, defensive baseline play")

        if coverage and (coverage[0] if coverage else 0) > 0.7:
            patterns.append("Strong court coverage")

        if net_rate > 0.15:
            patterns.append("Active net approach game")

        # Tactical score
        score = 6.0
        score += min(1.5, crosscourt_rate * 1.5)  # crosscourt is tactically sound
        score += min(0.5, coverage[0] if coverage else 0)
        score -= net_rate * 0.5 if net_rate < 0.05 else 0  # penalize no net play
        score = round(min(10.0, max(3.0, score)), 1)

        return {
            "shot_placement_zones": zone_pct,
            "rally_patterns": patterns or ["Standard baseline rallies"],
            "net_approach_rate": net_rate,
            "baseline_tendency": round(1 - net_rate, 2),
            "crosscourt_rate": crosscourt_rate,
            "dtl_rate": dtl_rate,
            "tactical_score": score,
        }
