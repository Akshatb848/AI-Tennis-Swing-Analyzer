"""
Review Engine — Post-match analysis with factual observations.
No motivational language. No coaching hype. Only data-driven findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


@dataclass
class ShotPattern:
    """A recurring shot pattern detected during the match."""
    description: str = ""
    occurrences: int = 0
    total_opportunities: int = 0
    success_rate: float = 0.0
    recommendation: str = ""

    @property
    def frequency_pct(self) -> float:
        if self.total_opportunities == 0:
            return 0.0
        return (self.occurrences / self.total_opportunities) * 100


@dataclass
class MovementFinding:
    """A movement inefficiency or pattern observation."""
    area: str = ""           # e.g. "court coverage", "recovery position"
    observation: str = ""     # factual statement
    frequency: int = 0        # how often observed
    correction: str = ""      # suggested adjustment


@dataclass
class SwingFinding:
    """A swing-related observation from pose analysis."""
    shot_type: str = ""       # forehand, backhand, serve, volley
    observation: str = ""
    affected_frames: int = 0
    total_frames: int = 0
    correction: str = ""


@dataclass
class PlayerReview:
    """Complete review for one player."""
    player_id: str = ""
    player_name: str = ""

    # Summary stats
    total_points_played: int = 0
    points_won: int = 0
    win_rate: float = 0.0

    # Shot breakdown
    forehand_count: int = 0
    backhand_count: int = 0
    serve_count: int = 0
    volley_count: int = 0
    forehand_win_rate: float = 0.0
    backhand_win_rate: float = 0.0
    serve_win_rate: float = 0.0

    # Speed
    avg_shot_speed_mph: float = 0.0
    max_shot_speed_mph: float = 0.0
    avg_serve_speed_mph: float = 0.0
    max_serve_speed_mph: float = 0.0

    # Placement
    deuce_court_pct: float = 0.0
    ad_court_pct: float = 0.0
    net_approach_count: int = 0

    # Patterns
    shot_patterns: list[ShotPattern] = field(default_factory=list)
    movement_findings: list[MovementFinding] = field(default_factory=list)
    swing_findings: list[SwingFinding] = field(default_factory=list)

    # Top corrections (max 3)
    corrections: list[str] = field(default_factory=list)


@dataclass
class RallyBreakdown:
    """Rally-level statistics."""
    total_rallies: int = 0
    avg_rally_length: float = 0.0
    max_rally_length: int = 0
    rally_length_distribution: dict[str, int] = field(default_factory=dict)
    # e.g. {"1-3": 15, "4-6": 10, "7-9": 5, "10+": 3}

    short_rally_pct: float = 0.0    # 1-3 shots
    medium_rally_pct: float = 0.0   # 4-6 shots
    long_rally_pct: float = 0.0     # 7+ shots


@dataclass
class MatchReview:
    """Complete post-match review."""
    session_id: str = ""
    match_type: str = "singles"
    duration_seconds: float = 0.0
    total_points: int = 0

    player_reviews: list[PlayerReview] = field(default_factory=list)
    rally_breakdown: RallyBreakdown = field(default_factory=RallyBreakdown)
    line_call_summary: dict = field(default_factory=dict)

    # Match-level observations (factual only)
    observations: list[str] = field(default_factory=list)


class ReviewEngine:
    """
    Post-match analysis engine.

    Produces factual observations and suggested corrections.
    No motivational language. No AI-styled wording.
    """

    def analyze_match(
        self,
        session_summary: dict,
        player_stats: Optional[dict] = None,
        swing_data: Optional[list[dict]] = None,
    ) -> MatchReview:
        """Generate complete match review from session data."""
        review = MatchReview(
            session_id=session_summary.get("session_id", ""),
            match_type=session_summary.get("match_type", "singles"),
            duration_seconds=session_summary.get("duration_seconds", 0),
            total_points=len(session_summary.get("points", [])),
        )

        points = session_summary.get("points", [])
        players = session_summary.get("players", [])

        # Build rally breakdown
        review.rally_breakdown = self._compute_rally_breakdown(points)

        # Build per-player reviews
        for i, player_name in enumerate(players):
            player_id = f"p{i}"
            pr = self._build_player_review(
                player_id, player_name, points,
                player_stats.get(player_id, {}) if player_stats else {},
                [s for s in (swing_data or []) if s.get("player_id") == player_id],
            )
            review.player_reviews.append(pr)

        # Line call summary
        line_calls = session_summary.get("line_calls", [])
        review.line_call_summary = self._summarize_line_calls(line_calls)

        # Match-level observations
        review.observations = self._generate_match_observations(review)

        return review

    def _build_player_review(
        self,
        player_id: str,
        player_name: str,
        points: list[dict],
        stats: dict,
        swings: list[dict],
    ) -> PlayerReview:
        pr = PlayerReview(player_id=player_id, player_name=player_name)

        # Points summary
        pr.total_points_played = len(points)
        pr.points_won = sum(1 for p in points if p.get("winner") == player_id)
        pr.win_rate = (pr.points_won / pr.total_points_played * 100) if pr.total_points_played > 0 else 0

        # Stats from existing stats calculator
        pr.forehand_count = stats.get("forehand_winners", 0) + stats.get("forehand_errors", 0)
        pr.backhand_count = stats.get("backhand_winners", 0) + stats.get("backhand_errors", 0)
        pr.serve_count = stats.get("aces", 0) + stats.get("double_faults", 0) + stats.get("first_serves_in", 0)
        pr.avg_serve_speed_mph = stats.get("avg_serve_speed_mph", 0)
        pr.max_serve_speed_mph = stats.get("max_serve_speed_mph", 0)

        # Shot patterns
        pr.shot_patterns = self._detect_shot_patterns(player_id, points, stats)

        # Movement findings
        pr.movement_findings = self._detect_movement_issues(player_id, points, stats)

        # Swing findings from pose data
        pr.swing_findings = self._detect_swing_issues(swings)

        # Top 3 corrections
        pr.corrections = self._prioritize_corrections(pr)

        return pr

    def _compute_rally_breakdown(self, points: list[dict]) -> RallyBreakdown:
        rb = RallyBreakdown()
        if not points:
            return rb

        rally_lengths = [p.get("rally_length", 0) for p in points]
        rb.total_rallies = len(rally_lengths)
        rb.avg_rally_length = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0
        rb.max_rally_length = max(rally_lengths) if rally_lengths else 0

        short = sum(1 for r in rally_lengths if 1 <= r <= 3)
        medium = sum(1 for r in rally_lengths if 4 <= r <= 6)
        long_ = sum(1 for r in rally_lengths if r >= 7)

        total = len(rally_lengths) or 1
        rb.short_rally_pct = short / total * 100
        rb.medium_rally_pct = medium / total * 100
        rb.long_rally_pct = long_ / total * 100
        rb.rally_length_distribution = {
            "1-3": short,
            "4-6": medium,
            "7-9": sum(1 for r in rally_lengths if 7 <= r <= 9),
            "10+": sum(1 for r in rally_lengths if r >= 10),
        }
        return rb

    def _detect_shot_patterns(self, player_id: str, points: list[dict], stats: dict) -> list[ShotPattern]:
        patterns = []

        # Pattern: return direction tendency
        won_points = [p for p in points if p.get("winner") == player_id]
        lost_points = [p for p in points if p.get("winner") != player_id and p.get("winner")]

        if len(won_points) >= 3:
            short_rally_wins = sum(1 for p in won_points if p.get("rally_length", 0) <= 3)
            if short_rally_wins > len(won_points) * 0.5:
                patterns.append(ShotPattern(
                    description=f"{short_rally_wins} of {len(won_points)} points won on short rallies (3 shots or fewer)",
                    occurrences=short_rally_wins,
                    total_opportunities=len(won_points),
                    success_rate=short_rally_wins / len(won_points) * 100,
                    recommendation="Short-rally wins may indicate reliance on serve. Longer rallies could expose consistency gaps.",
                ))

        if len(lost_points) >= 3:
            long_rally_losses = sum(1 for p in lost_points if p.get("rally_length", 0) >= 7)
            if long_rally_losses >= 3:
                patterns.append(ShotPattern(
                    description=f"{long_rally_losses} points lost in rallies of 7+ shots",
                    occurrences=long_rally_losses,
                    total_opportunities=len(lost_points),
                    success_rate=0,
                    recommendation="Extended rallies resulted in errors. Consider earlier shot selection to shorten points.",
                ))

        # Pattern: unforced errors
        ue = stats.get("unforced_errors", 0)
        winners = stats.get("winners", 0)
        if ue > winners and ue >= 5:
            patterns.append(ShotPattern(
                description=f"Unforced errors ({ue}) exceed winners ({winners})",
                occurrences=ue,
                total_opportunities=ue + winners,
                success_rate=winners / (ue + winners) * 100 if (ue + winners) > 0 else 0,
                recommendation="Reduce risk on non-attacking shots. Prioritize placement over power on returns.",
            ))

        return patterns

    def _detect_movement_issues(self, player_id: str, points: list[dict], stats: dict) -> list[MovementFinding]:
        findings = []

        # Recovery position
        net_points = stats.get("net_points_total", 0)
        net_won = stats.get("net_points_won", 0)
        if net_points >= 3 and net_won < net_points * 0.4:
            findings.append(MovementFinding(
                area="Net approaches",
                observation=f"Won {net_won} of {net_points} net approaches ({net_won / net_points * 100:.0f}%)",
                frequency=net_points,
                correction="Approach shots landing short allow passing shots. Deepen approach before moving forward.",
            ))

        # Court coverage
        coverage = stats.get("court_coverage_pct", 50)
        if coverage < 40:
            findings.append(MovementFinding(
                area="Court coverage",
                observation=f"Court coverage at {coverage:.0f}%, below expected range",
                frequency=1,
                correction="Wider split step and earlier preparation for lateral movement.",
            ))

        return findings

    def _detect_swing_issues(self, swings: list[dict]) -> list[SwingFinding]:
        findings = []
        if not swings:
            return findings

        # Group by shot type
        by_type = Counter(s.get("swing_type", "unknown") for s in swings)

        for shot_type, count in by_type.items():
            type_swings = [s for s in swings if s.get("swing_type") == shot_type]

            # Check for low contact height on groundstrokes
            avg_contact = sum(s.get("contact_height", 0.5) for s in type_swings) / len(type_swings)
            if shot_type in ("forehand", "backhand") and avg_contact < 0.35:
                findings.append(SwingFinding(
                    shot_type=shot_type,
                    observation=f"Average {shot_type} contact point is below hip level",
                    affected_frames=count,
                    total_frames=len(swings),
                    correction=f"Take the ball earlier. Step into the {shot_type} to make contact at waist height.",
                ))

        return findings

    def _prioritize_corrections(self, pr: PlayerReview) -> list[str]:
        """Select top 3 most impactful corrections."""
        all_corrections = []

        for p in pr.shot_patterns:
            if p.recommendation:
                all_corrections.append(p.recommendation)
        for m in pr.movement_findings:
            if m.correction:
                all_corrections.append(m.correction)
        for s in pr.swing_findings:
            if s.correction:
                all_corrections.append(s.correction)

        return all_corrections[:3]

    def _summarize_line_calls(self, line_calls: list[dict]) -> dict:
        if not line_calls:
            return {"total": 0, "in": 0, "out": 0, "challenged": 0}
        return {
            "total": len(line_calls),
            "in": sum(1 for lc in line_calls if lc.get("verdict") == "in"),
            "out": sum(1 for lc in line_calls if lc.get("verdict") == "out"),
            "challenged": sum(1 for lc in line_calls if lc.get("challenge_status", "none") != "none"),
            "avg_confidence": round(
                sum(lc.get("confidence", 0) for lc in line_calls) / len(line_calls) * 100, 1
            ),
        }

    def _generate_match_observations(self, review: MatchReview) -> list[str]:
        """Generate factual match-level observations."""
        obs = []
        rb = review.rally_breakdown

        if rb.avg_rally_length > 0:
            obs.append(f"Average rally length: {rb.avg_rally_length:.1f} shots")
        if rb.max_rally_length > 0:
            obs.append(f"Longest rally: {rb.max_rally_length} shots")
        if rb.short_rally_pct > 60:
            obs.append(f"{rb.short_rally_pct:.0f}% of rallies ended within 3 shots")

        for pr in review.player_reviews:
            if pr.points_won > 0:
                obs.append(f"{pr.player_name}: {pr.points_won} of {pr.total_points_played} points won ({pr.win_rate:.0f}%)")

        lc = review.line_call_summary
        if lc.get("total", 0) > 0:
            obs.append(f"Line calls: {lc['total']} total ({lc.get('out', 0)} out calls)")

        return obs
