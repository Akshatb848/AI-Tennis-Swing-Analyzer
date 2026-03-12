"""
Coaching Agent — converts technical analysis into structured coaching recommendations.

Takes outputs from all expert agents and generates:
- Structured coaching report with sections
- Prioritized improvement recommendations
- Performance summary narrative
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class CoachingAgent:
    """Expert agent that synthesizes technical data into human coaching language."""

    name = "coaching"

    def analyze(self, frames: list, context: dict) -> dict:
        """
        Generate structured coaching report from all expert outputs.

        Returns:
            performance_summary: str
            overall_rating: str ('Beginner'|'Intermediate'|'Advanced'|'Elite')
            overall_score: float 0-10
            sections: dict with all report sections
            recommendations: list of {priority, category, title, detail, drill}
            strengths: list of str
            improvement_areas: list of str
        """
        bio = context.get("biomechanics", {})
        strategy = context.get("strategy", {})
        player = context.get("player", {})
        ball = context.get("ball", {})
        court = context.get("court", {})
        match_type = context.get("match_type", "singles")

        shot_scores = bio.get("shot_quality_scores", {
            "forehand": 6.5, "backhand": 6.5, "serve": 6.5, "volley": 6.0, "movement": 6.0
        })
        issues = bio.get("issues", [])
        patterns = strategy.get("rally_patterns", ["Standard baseline play"])
        avg_speed = ball.get("avg_speed_kmh", 130.0)
        movement = player.get("movement_intensity", 0.4)
        surface = court.get("surface", "hard")
        tactical_score = strategy.get("tactical_score", 6.0)
        bio_score = bio.get("overall_technique_score", 6.0)

        overall_score = round((bio_score * 0.4 + tactical_score * 0.3 +
                               min(10, avg_speed / 20.0) * 0.15 +
                               movement * 10.0 * 0.15), 1)
        overall_score = min(10.0, max(2.0, overall_score))

        if overall_score >= 8.5:
            rating = "Elite"
        elif overall_score >= 7.0:
            rating = "Advanced"
        elif overall_score >= 5.0:
            rating = "Intermediate"
        else:
            rating = "Beginner"

        # performance summary
        summary = self._build_summary(
            rating, overall_score, avg_speed, patterns, surface, match_type, shot_scores
        )

        # Recommendations
        recs = self._build_recommendations(issues, shot_scores, strategy, player)

        # Strengths
        strengths = self._identify_strengths(shot_scores, avg_speed, strategy, movement)

        # Improvement areas
        improvement_areas = [f"{i['label']}: {i['tip']}" for i in issues[:3]]

        return {
            "performance_summary": summary,
            "overall_rating": rating,
            "overall_score": overall_score,
            "sections": {
                "shot_quality": {
                    "title": "Shot Quality Analysis",
                    "scores": shot_scores,
                    "narrative": self._shot_narrative(shot_scores),
                },
                "movement": {
                    "title": "Movement & Court Coverage",
                    "intensity": round(movement, 2),
                    "speed_kmh": player.get("avg_speed_kmh", 5.5),
                    "coverage": player.get("court_coverage", [0.5]),
                    "narrative": self._movement_narrative(movement, player),
                },
                "tactics": {
                    "title": "Tactical Analysis",
                    "score": tactical_score,
                    "patterns": patterns,
                    "crosscourt_rate": strategy.get("crosscourt_rate", 0.5),
                    "net_rate": strategy.get("net_approach_rate", 0.1),
                    "narrative": self._tactical_narrative(strategy),
                },
                "biomechanics": {
                    "title": "Technique & Biomechanics",
                    "score": bio_score,
                    "issues_detected": len(issues),
                    "narrative": self._bio_narrative(issues, bio_score),
                },
            },
            "recommendations": recs,
            "strengths": strengths,
            "improvement_areas": improvement_areas,
        }

    def _build_summary(self, rating, score, avg_speed, patterns, surface, match_type, scores) -> str:
        top_shot = max(scores, key=scores.get)
        top_val = scores[top_shot]
        pattern_str = patterns[0] if patterns else "standard baseline play"

        return (
            f"Overall performance: {rating} level ({score:.1f}/10). "
            f"Best shot: {top_shot.capitalize()} ({top_val}/10). "
            f"Average ball speed {avg_speed:.0f} km/h on {surface} court. "
            f"Primary tactical pattern: {pattern_str}. "
            f"{'Match play' if match_type != 'practice_rally' else 'Practice session'} analyzed."
        )

    def _build_recommendations(self, issues, shot_scores, strategy, player) -> list:
        recs = []

        # Critical issues first
        for issue in sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.get("severity", "low")]):
            recs.append({
                "priority": issue.get("severity", "medium"),
                "category": "technique",
                "title": issue["label"],
                "detail": issue["coaching"],
                "drill": issue.get("tip", ""),
            })

        # Low-scoring shot types
        for shot, sc in sorted(shot_scores.items(), key=lambda x: x[1]):
            if sc < 6.5 and len(recs) < 6:
                recs.append({
                    "priority": "high" if sc < 5.0 else "medium",
                    "category": "shot_quality",
                    "title": f"Improve {shot.capitalize()}",
                    "detail": f"Your {shot} scores {sc}/10. Focus on consistency and technique refinement.",
                    "drill": f"Hit 50 {shot}s cross-court daily, focusing on a consistent swing path.",
                })

        # Net play if under-utilized
        if strategy.get("net_approach_rate", 0) < 0.08 and len(recs) < 7:
            recs.append({
                "priority": "low",
                "category": "tactics",
                "title": "Develop Net Game",
                "detail": "You rarely approach the net. Incorporating more net play will create short-ball opportunities.",
                "drill": "Chip-and-charge drill: slice approach → volley finish.",
            })

        return recs[:6]

    def _identify_strengths(self, shot_scores, avg_speed, strategy, movement) -> list:
        strengths = []

        # High-scoring shots
        for shot, sc in shot_scores.items():
            if sc >= 7.5:
                strengths.append(f"Strong {shot.capitalize()} ({sc}/10)")

        # Ball speed
        if avg_speed > 140:
            strengths.append(f"Powerful striking ({avg_speed:.0f} km/h average)")

        # Movement
        if movement > 0.6:
            strengths.append("Active court movement and footwork")

        # Tactical patterns
        for p in strategy.get("rally_patterns", []):
            if "Strong" in p or "Aggressive" in p:
                strengths.append(p)

        return strengths[:5] or ["Consistent baseline rallying", "Solid groundstroke mechanics"]

    def _shot_narrative(self, scores) -> str:
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        return (
            f"Strongest shot: {best.capitalize()} ({scores[best]}/10). "
            f"Most needs work: {worst.capitalize()} ({scores[worst]}/10). "
        )

    def _movement_narrative(self, intensity, player) -> str:
        speed = player.get("avg_speed_kmh", 5.0)
        if intensity > 0.6:
            return f"Excellent court movement at {speed:.1f} km/h avg. Good split-stepping and recovery."
        elif intensity > 0.35:
            return f"Moderate movement intensity ({speed:.1f} km/h). Work on quicker first-step reactions."
        else:
            return f"Limited court movement detected. Focus on split-step timing and explosive first steps."

    def _tactical_narrative(self, strategy) -> str:
        cr = strategy.get("crosscourt_rate", 0.5)
        nr = strategy.get("net_approach_rate", 0.1)
        patterns = strategy.get("rally_patterns", [])
        pattern_str = "; ".join(patterns[:2]) if patterns else "consistent groundstroke exchanges"
        return (
            f"Crosscourt rate: {cr*100:.0f}%. Net approach rate: {nr*100:.0f}%. "
            f"Tactical patterns: {pattern_str}."
        )

    def _bio_narrative(self, issues, score) -> str:
        if not issues:
            return f"Technical score: {score}/10. Clean mechanics with no major issues detected."
        issue_names = ", ".join(i["label"] for i in issues[:3])
        return (
            f"Technical score: {score}/10. Issues detected: {issue_names}. "
            f"Focus on correcting these technique points to significantly improve performance."
        )
