"""
Event Processor — Converts raw CV detections into tennis-meaningful events.

Handles:
- Ball bounce → in/out decision with uncertainty radius
- Hit detection → shot segmentation via trajectory + acceleration + proximity
- Rally boundary detection
- Point outcome inference
- Frame-accurate shot timeline
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

from tennis.engine.shot_classifier import ShotClassifier
from tennis.models.events import (
    BallEvent,
    BounceConfidence,
    EventType,
    LineCallEvent,
    LineCallVerdict,
    PlayerEvent,
    Point2D,
    RallyEvent,
)
from tennis.models.match import (
    CourtZone,
    PointOutcomeType,
    ShotDetail,
    ShotType,
)


# ── Court Geometry (ITF standard dimensions in meters) ───────────────────────

# Court origin is at center of court. All measurements in meters.
COURT_LENGTH = 23.77  # baseline to baseline
COURT_WIDTH_SINGLES = 8.23
COURT_WIDTH_DOUBLES = 10.97
SERVICE_LINE_DEPTH = 6.40  # from net
NET_TO_BASELINE = COURT_LENGTH / 2  # 11.885m

# Line positions relative to center
SINGLES_SIDELINE = COURT_WIDTH_SINGLES / 2     # ±4.115m
DOUBLES_SIDELINE = COURT_WIDTH_DOUBLES / 2     # ±5.485m
BASELINE = COURT_LENGTH / 2                     # ±11.885m
SERVICE_LINE = SERVICE_LINE_DEPTH               # ±6.40m from center
CENTER_SERVICE_LINE = 0.0                       # x = 0


class CourtGeometry:
    """ITF-standard tennis court geometry for line call calculations."""

    def __init__(self, is_doubles: bool = False):
        self.is_doubles = is_doubles
        self.sideline = DOUBLES_SIDELINE if is_doubles else SINGLES_SIDELINE

    def is_ball_in(
        self,
        position: Point2D,
        is_serve: bool = False,
        serve_side: str = "deuce",
    ) -> tuple[bool, float, str]:
        """
        Determine if a ball bounce is in or out.
        
        Returns:
            (is_in, distance_from_nearest_line_cm, closest_line_name)
        """
        x, y = position.x, position.y

        if is_serve:
            return self._check_serve_in(x, y, serve_side)
        else:
            return self._check_rally_in(x, y)

    def _check_rally_in(self, x: float, y: float) -> tuple[bool, float, str]:
        """Check if a ball during rally is in bounds."""
        # Distance from each boundary
        dist_to_left = abs(x - (-self.sideline))
        dist_to_right = abs(x - self.sideline)
        dist_to_far_baseline = abs(y - BASELINE)
        dist_to_near_baseline = abs(y - (-BASELINE))

        # Find closest line
        distances = {
            "left_sideline": (dist_to_left, x >= -self.sideline),
            "right_sideline": (dist_to_right, x <= self.sideline),
            "far_baseline": (dist_to_far_baseline, y <= BASELINE),
            "near_baseline": (dist_to_near_baseline, y >= -BASELINE),
        }

        closest_line = min(distances, key=lambda k: distances[k][0])
        closest_dist_m = distances[closest_line][0]
        closest_dist_cm = closest_dist_m * 100

        # Ball is in if within all boundaries
        is_in = (
            abs(x) <= self.sideline
            and abs(y) <= BASELINE
        )

        return is_in, closest_dist_cm, closest_line

    def _check_serve_in(self, x: float, y: float, side: str) -> tuple[bool, float, str]:
        """Check if serve lands in correct service box."""
        # Service box boundaries (looking at target side)
        if side == "deuce":
            # Right service box (from server's perspective, ball goes to left side of receiver)
            box_x_min = 0.0
            box_x_max = self.sideline
        else:
            # Ad service box
            box_x_min = -self.sideline
            box_x_max = 0.0

        # Service box is between net and service line on the target side
        # If server is on near side (y < 0), target is 0 < y < SERVICE_LINE
        box_y_min = 0.0
        box_y_max = SERVICE_LINE

        # Distance calculations
        dist_to_left = abs(x - box_x_min)
        dist_to_right = abs(x - box_x_max)
        dist_to_service = abs(y - box_y_max)
        dist_to_net = abs(y - box_y_min)
        dist_to_center = abs(x - CENTER_SERVICE_LINE)

        distances = {
            "service_line": (dist_to_service, y <= box_y_max),
            "center_service_line": (dist_to_center, True),
            "sideline": (min(dist_to_left, dist_to_right), box_x_min <= x <= box_x_max),
            "net": (dist_to_net, y >= box_y_min),
        }

        closest_line = min(distances, key=lambda k: distances[k][0])
        closest_dist_cm = distances[closest_line][0] * 100

        is_in = (
            box_x_min <= x <= box_x_max
            and box_y_min <= y <= box_y_max
        )

        return is_in, closest_dist_cm, closest_line

    def get_court_zone(self, position: Point2D) -> CourtZone:
        """Determine which court zone a position falls in."""
        x, y = position.x, position.y

        # Determine depth (baseline / no-man's land / net)
        if abs(y) > SERVICE_LINE:
            depth = "baseline"
        elif abs(y) > SERVICE_LINE * 0.5:
            depth = "no_mans_land"
        else:
            depth = "net"

        # Determine width
        third = self.sideline * 2 / 3
        if x < -third:
            width = "left"
        elif x > third:
            width = "right"
        else:
            width = "center"

        zone_map = {
            ("baseline", "left"): CourtZone.BASELINE_LEFT,
            ("baseline", "center"): CourtZone.BASELINE_CENTER,
            ("baseline", "right"): CourtZone.BASELINE_RIGHT,
            ("no_mans_land", "left"): CourtZone.NO_MANS_LAND_LEFT,
            ("no_mans_land", "center"): CourtZone.NO_MANS_LAND_CENTER,
            ("no_mans_land", "right"): CourtZone.NO_MANS_LAND_RIGHT,
            ("net", "left"): CourtZone.NET_LEFT,
            ("net", "center"): CourtZone.NET_CENTER,
            ("net", "right"): CourtZone.NET_RIGHT,
        }

        return zone_map.get((depth, width), CourtZone.BASELINE_CENTER)


class EventProcessor:
    """
    Processes raw CV events into tennis-meaningful events.
    
    Takes streams of ball and player detections and produces:
    - Line call decisions
    - Rally boundaries
    - Shot classifications
    - Point outcomes
    """

    def __init__(self, is_doubles: bool = False):
        self.court = CourtGeometry(is_doubles=is_doubles)
        self.shot_classifier = ShotClassifier()
        self.current_rally_events: list[BallEvent] = []
        self.current_rally_shots: list[ShotDetail] = []
        self.rally_counter: int = 0
        self.is_serving: bool = True
        self.serve_side: str = "deuce"

        # Tracking state
        self._last_hit_frame: int = 0
        self._last_bounce_frame: int = 0
        self._last_hit_player: Optional[str] = None
        self._rally_active: bool = False
        self._shot_count: int = 0

        # Player roster for opponent lookup
        self._player_ids: list[str] = []

        # Shot detection state for trajectory analysis
        self._prev_trajectory_angle: Optional[float] = None
        self._trajectory_speeds: list[float] = []  # recent speeds for spike detection

    def process_ball_event(
        self,
        event: BallEvent,
        player_events: Optional[list[PlayerEvent]] = None,
    ) -> Optional[dict]:
        """
        Process a ball event and return any tennis events generated.
        
        Returns dict with keys: line_call, rally_event, point_outcome (if applicable)
        """
        result = {}

        if event.event_type == EventType.BALL_BOUNCE:
            line_call = self._process_bounce(event)
            if line_call:
                result["line_call"] = line_call

                # If ball is out, rally ends
                if line_call.verdict == LineCallVerdict.OUT:
                    rally = self._end_rally(
                        event,
                        outcome_type=PointOutcomeType.UNFORCED_ERROR,
                        winner_id=self._get_opponent(self._last_hit_player) if self._last_hit_player else None,
                    )
                    if rally:
                        result["rally_event"] = rally

        elif event.event_type == EventType.BALL_HIT:
            self._process_hit(event, player_events)

        elif event.event_type == EventType.BALL_NET:
            # Net fault
            rally = self._end_rally(
                event,
                outcome_type=PointOutcomeType.NET,
                winner_id=self._get_opponent(self._last_hit_player) if self._last_hit_player else None,
            )
            if rally:
                result["rally_event"] = rally

        elif event.event_type == EventType.SERVE:
            self._start_rally(event)

        return result if result else None

    def _process_bounce(self, event: BallEvent) -> Optional[LineCallEvent]:
        """Process a ball bounce and generate a line call."""
        if not event.position_court:
            return None

        is_in, dist_cm, closest_line = self.court.is_ball_in(
            event.position_court,
            is_serve=self.is_serving and self._shot_count <= 1,
            serve_side=self.serve_side,
        )

        # Determine confidence based on distance from line
        if dist_cm > 20:
            confidence = 0.99
            bounce_conf = BounceConfidence.HIGH
        elif dist_cm > 10:
            confidence = 0.90
            bounce_conf = BounceConfidence.HIGH
        elif dist_cm > 5:
            confidence = 0.80
            bounce_conf = BounceConfidence.MEDIUM
        elif dist_cm > 2:
            confidence = 0.65
            bounce_conf = BounceConfidence.LOW
        else:
            confidence = 0.50
            bounce_conf = BounceConfidence.UNCERTAIN

        # Update ball event
        event.line_call = LineCallVerdict.IN if is_in else LineCallVerdict.OUT
        event.line_call_confidence = confidence
        event.distance_from_line_cm = dist_cm

        # Calculate uncertainty radius (depends on CV model accuracy)
        uncertainty_cm = max(2.0, 10.0 - confidence * 8.0)

        line_call = LineCallEvent(
            session_id=event.session_id,
            rally_id=str(self.rally_counter),
            ball_event_id=event.id,
            verdict=LineCallVerdict.IN if is_in else LineCallVerdict.OUT,
            confidence=confidence,
            bounce_position_court=event.position_court,
            closest_line=closest_line,
            distance_from_line_cm=dist_cm,
            uncertainty_radius_cm=uncertainty_cm,
            timestamp_ms=event.timestamp_ms,
        )

        self._last_bounce_frame = event.frame_number
        self.current_rally_events.append(event)
        return line_call

    def _process_hit(
        self,
        event: BallEvent,
        player_events: Optional[list[PlayerEvent]] = None,
    ) -> None:
        """Process a ball hit event with trajectory-based detection."""
        self._shot_count += 1
        self._last_hit_frame = event.frame_number

        if not self._rally_active:
            self._start_rally(event)

        self.current_rally_events.append(event)

        # Track player roster for opponent lookup
        if player_events:
            for pe in player_events:
                if pe.player_id and pe.player_id not in self._player_ids:
                    self._player_ids.append(pe.player_id)

        # Classify the shot using the full classifier
        shot_type, shot_confidence = self._classify_shot(event, player_events)
        player_id = self._determine_hitting_player(event, player_events)
        self._last_hit_player = player_id

        # Track speed for trajectory analysis
        if event.velocity_mph:
            self._trajectory_speeds.append(event.velocity_mph)
            if len(self._trajectory_speeds) > 10:
                self._trajectory_speeds = self._trajectory_speeds[-10:]

        if player_id:
            speed_mph = event.velocity_mph or 0.0
            speed_kph = speed_mph * 1.60934 if speed_mph else None
            zone = None
            if event.position_court:
                zone = self.court.get_court_zone(event.position_court)

            shot = ShotDetail(
                shot_type=shot_type,
                player_id=player_id,
                speed_mph=speed_mph,
                speed_kph=speed_kph,
                placement_zone=zone,
                timestamp_ms=event.timestamp_ms,
                frame_number=event.frame_number,
            )
            self.current_rally_shots.append(shot)

    def _start_rally(self, event: BallEvent) -> None:
        """Start a new rally."""
        self.rally_counter += 1
        self._rally_active = True
        self._shot_count = 0
        self.current_rally_events = [event]
        self.current_rally_shots = []

    def _end_rally(
        self,
        event: BallEvent,
        outcome_type: PointOutcomeType,
        winner_id: Optional[str] = None,
    ) -> Optional[RallyEvent]:
        """End the current rally and produce a RallyEvent."""
        if not self._rally_active:
            return None

        self._rally_active = False

        if not self.current_rally_events:
            return None

        first = self.current_rally_events[0]
        last = event

        # Calculate excitement score
        excitement = self._calculate_excitement()

        rally = RallyEvent(
            session_id=event.session_id,
            point_number=self.rally_counter,
            start_frame=first.frame_number,
            end_frame=last.frame_number,
            start_timestamp_ms=first.timestamp_ms,
            end_timestamp_ms=last.timestamp_ms,
            duration_seconds=(last.timestamp_ms - first.timestamp_ms) / 1000.0,
            server_id=self._last_hit_player or "",
            winner_id=winner_id,
            rally_length=len(self.current_rally_shots),
            shots=[s.model_dump() for s in self.current_rally_shots],
            ball_events=[e.id for e in self.current_rally_events],
            outcome_type=outcome_type.value,
            last_shot_type=self.current_rally_shots[-1].shot_type.value if self.current_rally_shots else None,
            last_shot_player_id=self.current_rally_shots[-1].player_id if self.current_rally_shots else None,
            max_shot_speed_mph=max((s.speed_mph or 0 for s in self.current_rally_shots), default=0),
            avg_shot_speed_mph=(
                sum(s.speed_mph or 0 for s in self.current_rally_shots) / len(self.current_rally_shots)
                if self.current_rally_shots else 0
            ),
            excitement_score=excitement,
        )

        # Reset state
        self.current_rally_events = []
        self.current_rally_shots = []
        self._shot_count = 0

        return rally

    def _classify_shot(
        self,
        event: BallEvent,
        player_events: Optional[list[PlayerEvent]] = None,
    ) -> tuple[ShotType, float]:
        """Classify the shot type using full trajectory + pose analysis."""
        # Extract pose from nearest player
        player_pose = None
        player_position = None
        if player_events:
            nearest = self._find_nearest_player(event, player_events)
            if nearest:
                player_pose = nearest.pose
                player_position = nearest.position_court

        # Ball trajectory angle (compute from velocity if available)
        trajectory_angle = 0.0
        if event.velocity_mph:
            speed = event.velocity_mph
            trajectory_angle = self._compute_trajectory_angle(event)

        shot_type, confidence = self.shot_classifier.classify(
            ball_position=event.position_court,
            ball_velocity_mph=event.velocity_mph or 0.0,
            ball_trajectory_angle=trajectory_angle,
            player_pose=player_pose,
            player_position=player_position,
            shot_number_in_rally=self._shot_count,
            is_serve=(self._shot_count <= 1 and self.is_serving),
            prev_trajectory_angle=self._prev_trajectory_angle,
        )

        self._prev_trajectory_angle = trajectory_angle
        return shot_type, confidence

    def _compute_trajectory_angle(self, event: BallEvent) -> float:
        """Compute ball trajectory angle from recent events."""
        if len(self.current_rally_events) < 2:
            return 0.0
        prev = self.current_rally_events[-1]
        if prev.position_court and event.position_court:
            dx = event.position_court.x - prev.position_court.x
            dy = event.position_court.y - prev.position_court.y
            return math.degrees(math.atan2(dy, dx))
        return 0.0

    def _find_nearest_player(
        self, ball_event: BallEvent, player_events: list[PlayerEvent]
    ) -> Optional[PlayerEvent]:
        """Find the player closest to the ball position."""
        if not ball_event.position_court:
            return player_events[0] if player_events else None

        min_dist = float("inf")
        nearest = None
        for pe in player_events:
            if pe.position_court:
                dx = pe.position_court.x - ball_event.position_court.x
                dy = pe.position_court.y - ball_event.position_court.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
                    nearest = pe
        return nearest

    def _determine_hitting_player(
        self,
        ball_event: BallEvent,
        player_events: Optional[list[PlayerEvent]] = None,
    ) -> Optional[str]:
        """Determine which player hit the ball.

        Uses two signals:
        1. Spatial proximity (closest player to ball)
        2. Alternation logic (players alternate hits in rallies)
        """
        if not player_events:
            # Alternate from last hit player if known
            return self._alternate_player()

        if not ball_event.position_court:
            # Try alternation first, then fall back to first player
            alt = self._alternate_player()
            return alt if alt else (player_events[0].player_id if player_events else None)

        # Find closest player
        nearest = self._find_nearest_player(ball_event, player_events)
        closest_id = nearest.player_id if nearest else None

        # Cross-validate with alternation logic
        expected = self._alternate_player()
        if expected and closest_id and expected != closest_id:
            # Proximity wins, but only if the distance is reasonable (<3m)
            if nearest and nearest.position_court:
                dx = nearest.position_court.x - ball_event.position_court.x
                dy = nearest.position_court.y - ball_event.position_court.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 3.0:
                    return closest_id
            return expected  # alternation wins if proximity is ambiguous

        return closest_id or expected

    def _alternate_player(self) -> Optional[str]:
        """Get the expected next hitter via alternation logic."""
        if not self._last_hit_player or len(self._player_ids) < 2:
            return self._last_hit_player
        for pid in self._player_ids:
            if pid != self._last_hit_player:
                return pid
        return self._last_hit_player

    def _get_opponent(self, player_id: Optional[str]) -> Optional[str]:
        """Get opponent player ID using tracked player roster."""
        if not player_id:
            return None
        for pid in self._player_ids:
            if pid != player_id:
                return pid
        return None

    def _calculate_excitement(self) -> float:
        """Calculate an excitement score for highlight ranking."""
        score = 0.0

        # Longer rallies are more exciting
        rally_len = len(self.current_rally_shots)
        score += min(rally_len / 20.0, 0.3)

        # Faster shots are more exciting
        max_speed = max((s.speed_mph or 0 for s in self.current_rally_shots), default=0)
        score += min(max_speed / 150.0, 0.3)

        # Close line calls are exciting
        close_calls = sum(
            1 for e in self.current_rally_events
            if e.distance_from_line_cm is not None and e.distance_from_line_cm < 10
        )
        score += min(close_calls * 0.1, 0.2)

        # Variety of shots
        shot_types = set(s.shot_type for s in self.current_rally_shots)
        score += min(len(shot_types) / 6.0, 0.2)

        return min(score, 1.0)

    def set_serve_side(self, side: str) -> None:
        """Set the current serve side (deuce/ad)."""
        self.serve_side = side

    def set_is_serving(self, is_serving: bool) -> None:
        """Set whether current point is a serve."""
        self.is_serving = is_serving
