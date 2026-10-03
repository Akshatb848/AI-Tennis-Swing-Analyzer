"""
Video Overlay Engine — Score, speed, bounce, and analytics overlays.
Renders real-time and post-match visual overlays on match video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OverlayStyle(str, Enum):
    BROADCAST = "broadcast"       # ESPN/Tennis Channel style
    MINIMAL = "minimal"           # Clean, modern, minimal info
    DETAILED = "detailed"         # Full analytics overlay
    COACHING = "coaching"         # Biomechanical annotations


@dataclass
class ScoreOverlay:
    """Real-time scoreboard overlay data."""
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"
    sets_p1: list[int] = field(default_factory=list)
    sets_p2: list[int] = field(default_factory=list)
    current_game_p1: str = "0"
    current_game_p2: str = "0"
    server: int = 1  # 1 or 2
    is_tiebreak: bool = False
    set_number: int = 1
    match_time: str = "0:00"
    style: OverlayStyle = OverlayStyle.BROADCAST

    def to_render_data(self) -> dict:
        return {
            "type": "scoreboard",
            "position": "top_center",
            "data": {
                "p1": {"name": self.player1_name, "sets": self.sets_p1, "game": self.current_game_p1, "serving": self.server == 1},
                "p2": {"name": self.player2_name, "sets": self.sets_p2, "game": self.current_game_p2, "serving": self.server == 2},
                "set": self.set_number,
                "tiebreak": self.is_tiebreak,
                "time": self.match_time,
            },
        }


@dataclass
class SpeedOverlay:
    """Shot speed indicator overlay."""
    speed_mph: float = 0.0
    shot_type: str = "forehand"
    player_name: str = ""
    is_serve: bool = False
    position_x: float = 0.5  # normalized screen position
    position_y: float = 0.7

    def to_render_data(self) -> dict:
        return {
            "type": "speed",
            "position": "dynamic",
            "data": {
                "speed_mph": round(self.speed_mph, 1),
                "speed_kmh": round(self.speed_mph * 1.609, 1),
                "shot_type": self.shot_type,
                "player": self.player_name,
                "is_serve": self.is_serve,
            },
        }


@dataclass
class BounceOverlay:
    """Ball bounce position and line call overlay."""
    court_x: float = 0.0
    court_y: float = 0.0
    is_in: bool = True
    distance_from_line_cm: float = 0.0
    confidence: float = 1.0
    closest_line: str = ""
    is_challenged: bool = False
    show_trail: bool = True

    def to_render_data(self) -> dict:
        return {
            "type": "bounce",
            "position": "court",
            "data": {
                "x": self.court_x,
                "y": self.court_y,
                "verdict": "IN" if self.is_in else "OUT",
                "distance_cm": round(self.distance_from_line_cm, 1),
                "confidence": round(self.confidence * 100, 1),
                "line": self.closest_line,
                "challenged": self.is_challenged,
                "color": "#00ff88" if self.is_in else "#ff4444",
                "trail": self.show_trail,
            },
        }


@dataclass
class TrajectoryOverlay:
    """Ball trajectory arc overlay."""
    points: list[tuple[float, float]] = field(default_factory=list)
    color: str = "#ffaa00"
    width: float = 2.0
    show_prediction: bool = False

    def to_render_data(self) -> dict:
        return {
            "type": "trajectory",
            "data": {
                "points": [{"x": x, "y": y} for x, y in self.points],
                "color": self.color,
                "width": self.width,
                "prediction": self.show_prediction,
            },
        }


@dataclass
class AnalyticsOverlay:
    """Side-panel analytics overlay."""
    stat_name: str = ""
    stat_value: str = ""
    stat_comparison: str | None = None
    chart_type: str = "bar"  # bar, pie, radar
    chart_data: dict = field(default_factory=dict)

    def to_render_data(self) -> dict:
        return {
            "type": "analytics",
            "position": "sidebar",
            "data": {
                "stat": self.stat_name,
                "value": self.stat_value,
                "comparison": self.stat_comparison,
                "chart": {"type": self.chart_type, "data": self.chart_data},
            },
        }


class OverlayEngine:
    """
    Manages overlay rendering for match video.

    Produces frame-by-frame overlay instructions that can be rendered
    by the iOS app (real-time) or video processor (post-match).
    """

    def __init__(self, style: OverlayStyle = OverlayStyle.BROADCAST):
        self.style = style
        self._overlays: list[dict] = []

    def add_score_update(self, score: ScoreOverlay, timestamp_ms: int):
        self._overlays.append({"timestamp_ms": timestamp_ms, "overlay": score.to_render_data()})

    def add_speed_flash(self, speed: SpeedOverlay, timestamp_ms: int, duration_ms: int = 2000):
        self._overlays.append({
            "timestamp_ms": timestamp_ms,
            "duration_ms": duration_ms,
            "overlay": speed.to_render_data(),
        })

    def add_bounce_mark(self, bounce: BounceOverlay, timestamp_ms: int, duration_ms: int = 3000):
        self._overlays.append({
            "timestamp_ms": timestamp_ms,
            "duration_ms": duration_ms,
            "overlay": bounce.to_render_data(),
        })

    def add_trajectory(self, traj: TrajectoryOverlay, timestamp_ms: int, duration_ms: int = 2000):
        self._overlays.append({
            "timestamp_ms": timestamp_ms,
            "duration_ms": duration_ms,
            "overlay": traj.to_render_data(),
        })

    def add_analytics(self, analytics: AnalyticsOverlay, timestamp_ms: int, duration_ms: int = 5000):
        self._overlays.append({
            "timestamp_ms": timestamp_ms,
            "duration_ms": duration_ms,
            "overlay": analytics.to_render_data(),
        })

    def get_overlays_at(self, timestamp_ms: int) -> list[dict]:
        """Get all active overlays at a given timestamp."""
        active = []
        for entry in self._overlays:
            start = entry["timestamp_ms"]
            duration = entry.get("duration_ms", float("inf"))
            if start <= timestamp_ms < start + duration:
                active.append(entry["overlay"])
        return active

    def get_all_overlays(self) -> list[dict]:
        """Get all overlay instructions sorted by timestamp."""
        return sorted(self._overlays, key=lambda x: x["timestamp_ms"])

    def generate_overlay_track(self, fps: float = 30.0, duration_ms: int = 0) -> list[dict]:
        """Generate frame-by-frame overlay track for video rendering."""
        if not duration_ms:
            duration_ms = max((o["timestamp_ms"] + o.get("duration_ms", 5000)) for o in self._overlays) if self._overlays else 0
        frame_interval = int(1000 / fps)
        track = []
        for ts in range(0, duration_ms, frame_interval):
            overlays = self.get_overlays_at(ts)
            if overlays:
                track.append({"frame_time_ms": ts, "overlays": overlays})
        return track

    def clear(self):
        self._overlays.clear()
