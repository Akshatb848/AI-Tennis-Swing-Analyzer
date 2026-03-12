"""
Dashboard Data Provider — Fetches real match data from API or in-memory session stores.

Replaces generate_sample_data() with real data while maintaining
identical output shape for zero-change dashboard rendering.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ── In-memory session store (for direct access without API) ──────────────────
_completed_sessions: dict[str, dict] = {}


def store_session_data(session_id: str, data: dict):
    """Store completed session data for dashboard consumption."""
    _completed_sessions[session_id] = data


def get_available_sessions() -> list[dict]:
    """Get list of available sessions for the dashboard selector."""
    sessions = []
    for sid, data in _completed_sessions.items():
        match_info = data.get("match", {})
        sessions.append({
            "id": sid,
            "player1": match_info.get("player1", "Player 1"),
            "player2": match_info.get("player2", "Player 2"),
            "surface": match_info.get("surface", "Unknown"),
            "duration": match_info.get("duration", ""),
            "match_type": match_info.get("match_type", "Singles"),
        })
    return sessions


def get_session_data(session_id: str) -> Optional[dict]:
    """Get data for a specific session in dashboard format."""
    return _completed_sessions.get(session_id)


class DashboardDataProvider:
    """
    Fetches real match data from the API or in-memory session store.

    Output shape matches generate_sample_data() exactly, so dashboard
    code changes are minimal — just swap the data source.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000/api/v1"):
        self.api_base_url = api_base_url

    def get_match_data(self, session_id: Optional[str] = None) -> Optional[dict]:
        """
        Get match data in the same shape as generate_sample_data().

        Tries:
        1. In-memory store (direct access, no API needed)
        2. API call (if httpx available)
        3. Returns None if no data available
        """
        # 1. Check in-memory store
        if session_id and session_id in _completed_sessions:
            return _completed_sessions[session_id]

        # If no specific session, return the most recent one
        if not session_id and _completed_sessions:
            latest_id = list(_completed_sessions.keys())[-1]
            return _completed_sessions[latest_id]

        # 2. Try API
        if session_id and HAS_HTTPX:
            try:
                return self._fetch_from_api(session_id)
            except Exception as e:
                logger.warning("Failed to fetch from API: %s", e)

        return None

    def _fetch_from_api(self, session_id: str) -> Optional[dict]:
        """Fetch session data via the REST API."""
        try:
            with httpx.Client(timeout=10) as client:
                # Get recording summary
                r = client.get(f"{self.api_base_url}/recording/{session_id}/summary")
                if r.status_code != 200:
                    return None
                summary = r.json()

            return self._transform_api_response(summary)
        except Exception as e:
            logger.warning("API fetch failed: %s", e)
            return None

    @staticmethod
    def _transform_api_response(summary: dict) -> dict:
        """Transform API response into dashboard data format."""
        players = summary.get("players", ["Player 1", "Player 2"])
        points = summary.get("points", [])
        line_calls = summary.get("line_calls", [])
        stats = summary.get("stats", {})

        # Build player stats
        p1_stats = stats.get("p1", {})
        p2_stats = stats.get("p2", {})

        # Rally breakdown
        rally_lengths = [p.get("rally_length", 0) for p in points]
        avg_rally = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0
        max_rally = max(rally_lengths) if rally_lengths else 0

        short = sum(1 for r in rally_lengths if 1 <= r <= 3)
        medium = sum(1 for r in rally_lengths if 4 <= r <= 6)
        long_r = sum(1 for r in rally_lengths if 7 <= r <= 9)
        very_long = sum(1 for r in rally_lengths if r >= 10)
        total = len(rally_lengths) or 1

        # Line call summary
        lc_in = sum(1 for lc in line_calls if lc.get("verdict") == "in")
        lc_out = sum(1 for lc in line_calls if lc.get("verdict") == "out")
        lc_conf = (
            sum(lc.get("confidence", 0) for lc in line_calls) / len(line_calls) * 100
            if line_calls else 0
        )
        challenges = [lc for lc in line_calls if lc.get("is_challenged")]

        return {
            "match": {
                "player1": players[0] if players else "Player 1",
                "player2": players[1] if len(players) > 1 else "Player 2",
                "score": summary.get("score", [[0], [0]]),
                "duration": summary.get("duration", ""),
                "surface": summary.get("court_surface", "Hard Court"),
                "match_type": summary.get("match_type", "Singles"),
            },
            "p1_stats": {
                "aces": p1_stats.get("aces", 0),
                "double_faults": p1_stats.get("double_faults", 0),
                "first_serve_pct": p1_stats.get("first_serve_pct", 0),
                "winners": p1_stats.get("winners", 0),
                "unforced_errors": p1_stats.get("unforced_errors", 0),
                "break_points_won": p1_stats.get("break_points_won", "0/0"),
                "net_points_won": p1_stats.get("net_points_won", 0),
                "net_points_total": p1_stats.get("net_points_total", 0),
                "forehand_winners": p1_stats.get("forehand_winners", 0),
                "backhand_winners": p1_stats.get("backhand_winners", 0),
                "forehand_errors": p1_stats.get("forehand_errors", 0),
                "backhand_errors": p1_stats.get("backhand_errors", 0),
                "max_serve_speed": p1_stats.get("max_serve_speed", 0),
                "avg_serve_speed": p1_stats.get("avg_serve_speed", 0),
            },
            "p2_stats": {
                "aces": p2_stats.get("aces", 0),
                "double_faults": p2_stats.get("double_faults", 0),
                "first_serve_pct": p2_stats.get("first_serve_pct", 0),
                "winners": p2_stats.get("winners", 0),
                "unforced_errors": p2_stats.get("unforced_errors", 0),
                "break_points_won": p2_stats.get("break_points_won", "0/0"),
                "net_points_won": p2_stats.get("net_points_won", 0),
                "net_points_total": p2_stats.get("net_points_total", 0),
                "forehand_winners": p2_stats.get("forehand_winners", 0),
                "backhand_winners": p2_stats.get("backhand_winners", 0),
                "forehand_errors": p2_stats.get("forehand_errors", 0),
                "backhand_errors": p2_stats.get("backhand_errors", 0),
                "max_serve_speed": p2_stats.get("max_serve_speed", 0),
                "avg_serve_speed": p2_stats.get("avg_serve_speed", 0),
            },
            "rally_breakdown": {
                "avg_length": round(avg_rally, 1),
                "max_length": max_rally,
                "distribution": {
                    "1-3": short, "4-6": medium,
                    "7-9": long_r, "10+": very_long,
                },
                "short_pct": round(short / total * 100, 1),
                "medium_pct": round(medium / total * 100, 1),
                "long_pct": round((long_r + very_long) / total * 100, 1),
            },
            "line_calls": {
                "total": len(line_calls),
                "in": lc_in,
                "out": lc_out,
                "avg_confidence": round(lc_conf, 1),
                "challenges_made": len(challenges),
                "challenges_successful": sum(
                    1 for c in challenges
                    if c.get("challenge_status") == "overturned"
                ),
                "calls": [
                    {
                        "verdict": lc.get("verdict", "").upper(),
                        "confidence": f"{lc.get('confidence', 0) * 100:.0f}%",
                        "distance": f"{lc.get('distance_from_line_cm', 0):.1f}cm",
                        "line": lc.get("line_name", ""),
                        "point": lc.get("point_number", 0),
                    }
                    for lc in line_calls[:10]
                ],
            },
            "swing_distribution": summary.get(
                "swing_distribution",
                {"Forehand": 0, "Backhand": 0, "Serve": 0, "Volley": 0},
            ),
            "review": summary.get("review", {
                "observations": [],
                "p1_corrections": [],
                "p2_corrections": [],
                "drills": [],
            }),
        }
