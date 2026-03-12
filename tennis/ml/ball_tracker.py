"""
Ball Tracker — Continuous ball detection, Kalman-filtered tracking,
trajectory smoothing, occlusion recovery, and physics-based speed measurement.

Production-grade ball state stream for commercial analytics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from tennis.models.events import BallEvent, BoundingBox, EventType, Point2D, Point3D


# ── Shot Speed Record ────────────────────────────────────────────────────────

@dataclass
class ShotSpeedRecord:
    """Physics-based speed measurement for a single shot."""
    initial_velocity_kph: float = 0.0
    initial_velocity_mph: float = 0.0
    flight_duration_ms: int = 0
    bounce_location: Optional[Point2D] = None
    is_calibrated: bool = False
    raw_pixel_speed: float = 0.0


# ── Trajectory Point ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Single point in the trajectory buffer."""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed_px: float = 0.0
    frame_number: int = 0
    timestamp_ms: int = 0
    confidence: float = 0.0
    is_detected: bool = False
    is_occluded: bool = False
    is_interpolated: bool = False


# ── Ball State ───────────────────────────────────────────────────────────────

@dataclass
class BallState:
    """Per-frame ball state for continuous output stream."""
    frame_number: int = 0
    timestamp_ms: int = 0
    position_x: float = 0.0
    position_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed_px_per_sec: float = 0.0
    speed_kph: float = 0.0
    speed_mph: float = 0.0
    trajectory_angle_deg: float = 0.0
    visibility_confidence: float = 0.0
    is_detected: bool = False
    is_occluded: bool = False
    is_bounce: bool = False
    is_shot_contact: bool = False


# ── Kalman Tracker ───────────────────────────────────────────────────────────

class KalmanTracker:
    """2D Kalman filter with position + velocity state vector.

    State: [x, y, vx, vy]
    Uses constant-velocity model with gravity correction on vy.
    """

    def __init__(self, process_noise: float = 0.5, measurement_noise: float = 2.0):
        # State
        self.x: float = 0.0
        self.y: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0

        # Uncertainty (diagonal covariance approximation)
        self.px: float = 100.0  # position uncertainty
        self.py: float = 100.0
        self.pvx: float = 50.0  # velocity uncertainty
        self.pvy: float = 50.0

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.confidence: float = 0.0
        self.frames_since_detection: int = 999
        self._initialized: bool = False

    def predict(self, dt: float = 1 / 30) -> tuple[float, float]:
        """Predict next state. Returns predicted (x, y)."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 9.81 * dt  # gravity

        # Increase uncertainty
        self.px += self.pvx * dt + self.process_noise
        self.py += self.pvy * dt + self.process_noise
        self.pvx += self.process_noise
        self.pvy += self.process_noise

        self.frames_since_detection += 1
        self.confidence *= 0.92  # decay
        return self.x, self.y

    def update(self, x: float, y: float) -> None:
        """Update state with new measurement."""
        if not self._initialized:
            self.x = x
            self.y = y
            self.vx = 0.0
            self.vy = 0.0
            self._initialized = True
        else:
            # Kalman gain (simplified diagonal)
            kx = self.px / (self.px + self.measurement_noise)
            ky = self.py / (self.py + self.measurement_noise)

            # Innovation
            innov_x = x - self.x
            innov_y = y - self.y

            # Update velocity from position change
            dt = 1.0 / 30.0
            new_vx = innov_x / dt
            new_vy = innov_y / dt

            # Smooth velocity update
            alpha = 0.6
            self.vx = alpha * new_vx + (1 - alpha) * self.vx
            self.vy = alpha * new_vy + (1 - alpha) * self.vy

            # Update position
            self.x += kx * innov_x
            self.y += ky * innov_y

            # Update uncertainty
            self.px *= (1 - kx)
            self.py *= (1 - ky)

        self.frames_since_detection = 0
        self.confidence = 1.0

    @property
    def is_tracking(self) -> bool:
        return self._initialized and self.frames_since_detection < 15 and self.confidence > 0.15

    @property
    def speed_pixels_per_frame(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    def reset(self) -> None:
        self.__init__(self.process_noise, self.measurement_noise)


# ── Trajectory Buffer ────────────────────────────────────────────────────────

class TrajectoryBuffer:
    """Rolling window of recent trajectory points with smoothing."""

    def __init__(self, max_frames: int = 90):
        self.max_frames = max_frames
        self.points: list[TrajectoryPoint] = []

    def add(self, point: TrajectoryPoint) -> None:
        self.points.append(point)
        if len(self.points) > self.max_frames:
            self.points = self.points[-self.max_frames:]

    def get_smoothed_position(self, window: int = 5) -> Optional[tuple[float, float]]:
        """Weighted moving average over recent points."""
        detected = [p for p in self.points[-window:] if p.is_detected]
        if not detected:
            return None
        total_w = 0.0
        sx, sy = 0.0, 0.0
        for i, p in enumerate(detected):
            w = (i + 1) * p.confidence  # recency + confidence weighting
            sx += p.x * w
            sy += p.y * w
            total_w += w
        if total_w == 0:
            return None
        return sx / total_w, sy / total_w

    def get_trajectory_angle(self) -> float:
        """Compute trajectory direction in degrees from recent points."""
        if len(self.points) < 2:
            return 0.0
        detected = [p for p in self.points[-10:] if p.is_detected]
        if len(detected) < 2:
            return 0.0
        p1, p2 = detected[-2], detected[-1]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.degrees(math.atan2(dy, dx))

    def detect_direction_change(self, angle_threshold: float = 30.0) -> bool:
        """Detect trajectory direction change (> threshold degrees) in last 5 frames."""
        detected = [p for p in self.points[-8:] if p.is_detected]
        if len(detected) < 3:
            return False
        # Compute angles for consecutive segments
        angles = []
        for i in range(1, len(detected)):
            dx = detected[i].x - detected[i - 1].x
            dy = detected[i].y - detected[i - 1].y
            angles.append(math.degrees(math.atan2(dy, dx)))

        # Check for sharp change
        for i in range(1, len(angles)):
            delta = abs(angles[i] - angles[i - 1])
            if delta > 180:
                delta = 360 - delta
            if delta >= angle_threshold:
                return True
        return False

    def detect_speed_spike(self, ratio_threshold: float = 1.4) -> bool:
        """Detect acceleration spike (speed change > ratio) in last 5 frames."""
        detected = [p for p in self.points[-6:] if p.is_detected and p.speed_px > 0]
        if len(detected) < 3:
            return False
        speeds = [p.speed_px for p in detected]
        for i in range(1, len(speeds)):
            if speeds[i - 1] > 0:
                ratio = speeds[i] / speeds[i - 1]
                if ratio >= ratio_threshold or ratio <= (1.0 / ratio_threshold):
                    return True
        return False

    def get_recent_speed_trend(self, window: int = 10) -> list[float]:
        """Get recent speed values for trend analysis."""
        return [p.speed_px for p in self.points[-window:] if p.is_detected]

    def check_trajectory_continuity(self, new_x: float, new_y: float,
                                     max_jump_px: float = 200.0) -> bool:
        """Check if a new detection is consistent with predicted trajectory."""
        if not self.points:
            return True
        last = self.points[-1]
        dist = math.sqrt((new_x - last.x) ** 2 + (new_y - last.y) ** 2)
        return dist < max_jump_px

    def clear(self) -> None:
        self.points.clear()

    def __len__(self) -> int:
        return len(self.points)


# ── Speed Calculator ─────────────────────────────────────────────────────────

class SpeedCalculator:
    """Physics-based speed measurement using court calibration.

    When a homography matrix is available, converts pixel distances to
    real-world court meters. Otherwise uses a configurable px-to-meter ratio.
    """

    MAX_SPEED_KPH = 260.0      # World record serve
    MIN_VALID_SPEED_KPH = 15.0  # Below this is likely noise

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._px_to_meters: float = 0.01  # default fallback (1px ≈ 1cm)
        self._is_calibrated: bool = False

    def calibrate(self, px_to_meters: float) -> None:
        """Set the pixel-to-meters conversion factor from court homography."""
        if px_to_meters > 0:
            self._px_to_meters = px_to_meters
            self._is_calibrated = True

    def calibrate_from_court_width(self, court_width_px: float,
                                     is_doubles: bool = False) -> None:
        """Calibrate from known court width in pixels."""
        real_width = 10.97 if is_doubles else 8.23  # meters
        if court_width_px > 0:
            self._px_to_meters = real_width / court_width_px
            self._is_calibrated = True

    def compute_speed(self, dx_px: float, dy_px: float,
                      dt_frames: int = 1) -> ShotSpeedRecord:
        """Compute speed from pixel displacement over frame count."""
        if dt_frames <= 0:
            return ShotSpeedRecord()

        dist_px = math.sqrt(dx_px ** 2 + dy_px ** 2)
        dist_m = dist_px * self._px_to_meters
        time_s = dt_frames / self.fps

        speed_mps = dist_m / time_s if time_s > 0 else 0
        speed_kph = speed_mps * 3.6
        speed_mph = speed_kph * 0.621371

        # Clamp outliers
        if speed_kph > self.MAX_SPEED_KPH:
            speed_kph = self.MAX_SPEED_KPH
            speed_mph = speed_kph * 0.621371

        record = ShotSpeedRecord(
            initial_velocity_kph=round(speed_kph, 1),
            initial_velocity_mph=round(speed_mph, 1),
            flight_duration_ms=int(time_s * 1000),
            is_calibrated=self._is_calibrated,
            raw_pixel_speed=dist_px / dt_frames,
        )
        return record

    def is_valid_speed(self, speed_kph: float) -> bool:
        return self.MIN_VALID_SPEED_KPH <= speed_kph <= self.MAX_SPEED_KPH


# ── Ball Tracker ─────────────────────────────────────────────────────────────

class BallTracker:
    """Continuous ball detection + Kalman tracking + trajectory analysis.

    Production-grade tracker with:
    - Multi-frame trajectory buffer with smoothing
    - Occlusion recovery with trajectory consistency check
    - Bounce detection via trajectory curvature (2nd derivative)
    - Physics-based speed measurement with court calibration
    - Continuous ball state stream output
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.kalman = KalmanTracker()
        self.trajectory = TrajectoryBuffer(max_frames=90)
        self.speed_calc = SpeedCalculator(fps=fps)
        self.state_stream: list[BallState] = []

        # Legacy-compatible trajectory list
        self._legacy_trajectory: list[tuple[float, float, int]] = []

        # Bounce detection state
        self._prev_vy: float = 0.0
        self._prev_prev_vy: float = 0.0
        self._bounce_cooldown: int = 0
        self.frame_count: int = 0

        # Shot detection state
        self._last_shot_frame: int = -30

    def process_frame(
        self, detection: Optional[BoundingBox], frame_number: int, session_id: str = ""
    ) -> Optional[BallEvent]:
        """Process a single frame detection. Returns a BallEvent if significant."""
        self.frame_count = frame_number
        timestamp_ms = int(frame_number / self.fps * 1000)

        if self._bounce_cooldown > 0:
            self._bounce_cooldown -= 1

        state = BallState(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
        )

        if detection and detection.confidence > 0.25:
            cx = (detection.x1 + detection.x2) / 2
            cy = (detection.y1 + detection.y2) / 2

            # Trajectory continuity check for occlusion recovery
            is_continuous = self.trajectory.check_trajectory_continuity(cx, cy)

            if is_continuous or not self.kalman.is_tracking:
                self.kalman.update(cx, cy)
            else:
                # Large jump — possible re-detection after occlusion
                # Accept if Kalman was lost, otherwise ignore as noise
                if self.kalman.frames_since_detection > 5:
                    self.kalman.update(cx, cy)
                else:
                    self.kalman.predict(1 / self.fps)

            speed_px = self.kalman.speed_pixels_per_frame
            traj_point = TrajectoryPoint(
                x=self.kalman.x, y=self.kalman.y,
                vx=self.kalman.vx, vy=self.kalman.vy,
                speed_px=speed_px,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                confidence=detection.confidence,
                is_detected=True,
            )
            self.trajectory.add(traj_point)

            # Legacy trajectory
            self._legacy_trajectory.append((self.kalman.x, self.kalman.y, frame_number))
            if len(self._legacy_trajectory) > 300:
                self._legacy_trajectory = self._legacy_trajectory[-200:]

            # Bounce detection (curvature-based)
            is_bounce = self._detect_bounce_curvature()
            # Shot contact detection (trajectory change + speed spike)
            is_shot = self._detect_shot_contact(frame_number)

            event_type = EventType.BALL_BOUNCE if is_bounce else EventType.BALL_HIT

            # Speed measurement
            speed_record = self._compute_current_speed()
            speed_mph = speed_record.initial_velocity_mph
            speed_kph = speed_record.initial_velocity_kph

            # Build ball state
            state.position_x = self.kalman.x
            state.position_y = self.kalman.y
            state.velocity_x = self.kalman.vx
            state.velocity_y = self.kalman.vy
            state.speed_px_per_sec = speed_px * self.fps
            state.speed_kph = speed_kph
            state.speed_mph = speed_mph
            state.trajectory_angle_deg = self.trajectory.get_trajectory_angle()
            state.visibility_confidence = detection.confidence
            state.is_detected = True
            state.is_bounce = is_bounce
            state.is_shot_contact = is_shot
            self.state_stream.append(state)

            return BallEvent(
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                session_id=session_id,
                position_image=detection,
                velocity_mph=speed_mph,
                detection_confidence=detection.confidence,
                tracker_id=0,
            )
        else:
            # No detection — predict and mark occluded
            self.kalman.predict(1 / self.fps)

            if self.kalman.is_tracking:
                speed_px = self.kalman.speed_pixels_per_frame
                traj_point = TrajectoryPoint(
                    x=self.kalman.x, y=self.kalman.y,
                    vx=self.kalman.vx, vy=self.kalman.vy,
                    speed_px=speed_px,
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    confidence=self.kalman.confidence,
                    is_detected=False,
                    is_occluded=True,
                    is_interpolated=True,
                )
                self.trajectory.add(traj_point)

                state.position_x = self.kalman.x
                state.position_y = self.kalman.y
                state.velocity_x = self.kalman.vx
                state.velocity_y = self.kalman.vy
                state.speed_px_per_sec = speed_px * self.fps
                state.visibility_confidence = self.kalman.confidence
                state.is_occluded = True
                self.state_stream.append(state)

                return BallEvent(
                    event_type=EventType.BALL_HIT,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame_number,
                    session_id=session_id,
                    detection_confidence=self.kalman.confidence,
                    is_interpolated=True,
                    is_occluded=True,
                    tracker_id=0,
                )

            # Fully lost — still record state for continuity
            state.visibility_confidence = 0.0
            state.is_occluded = True
            self.state_stream.append(state)
            return None

    def _detect_bounce_curvature(self) -> bool:
        """Detect ball bounce using trajectory curvature (2nd derivative of vy).

        A bounce causes vy to change from positive (falling) to negative (rising),
        with a sharp 2nd-derivative spike.
        """
        if self._bounce_cooldown > 0:
            return False

        cur_vy = self.kalman.vy

        # 2nd derivative check: acceleration of vertical velocity
        accel = cur_vy - 2 * self._prev_vy + self._prev_prev_vy
        self._prev_prev_vy = self._prev_vy
        self._prev_vy = cur_vy

        # Bounce: vy was positive (falling) and reverses, with significant accel
        if self._prev_vy > 2.0 and cur_vy < -1.0 and abs(accel) > 3.0:
            self._bounce_cooldown = 10
            return True

        # Fallback: simple sign-change detection
        if self._prev_prev_vy > 0 and cur_vy < 0 and abs(self._prev_prev_vy - cur_vy) > 5.0:
            self._bounce_cooldown = 10
            return True

        return False

    def _detect_shot_contact(self, frame_number: int) -> bool:
        """Detect shot contact via trajectory direction change + speed spike."""
        if frame_number - self._last_shot_frame < 8:
            return False

        dir_change = self.trajectory.detect_direction_change(angle_threshold=30.0)
        speed_spike = self.trajectory.detect_speed_spike(ratio_threshold=1.4)

        if dir_change or speed_spike:
            self._last_shot_frame = frame_number
            return True
        return False

    def _compute_current_speed(self) -> ShotSpeedRecord:
        """Compute current ball speed from Kalman state."""
        speed_px = self.kalman.speed_pixels_per_frame
        return self.speed_calc.compute_speed(
            dx_px=self.kalman.vx / self.fps,
            dy_px=self.kalman.vy / self.fps,
            dt_frames=1,
        )

    def get_state_stream(self) -> list[BallState]:
        """Return the full continuous ball state stream."""
        return self.state_stream

    def get_latest_state(self) -> Optional[BallState]:
        """Return the most recent ball state."""
        return self.state_stream[-1] if self.state_stream else None

    def get_trajectory_points(self) -> list[TrajectoryPoint]:
        """Return trajectory buffer points."""
        return self.trajectory.points

    def calibrate_speed(self, court_width_px: float, is_doubles: bool = False) -> None:
        """Calibrate speed measurement from court width in pixels."""
        self.speed_calc.calibrate_from_court_width(court_width_px, is_doubles)

    def reset(self):
        self.kalman = KalmanTracker()
        self.trajectory.clear()
        self.state_stream.clear()
        self._legacy_trajectory.clear()
        self._prev_vy = 0.0
        self._prev_prev_vy = 0.0
        self._bounce_cooldown = 0
        self.frame_count = 0
        self._last_shot_frame = -30


# ── Legacy compatibility aliases ─────────────────────────────────────────────
# Keep KalmanState accessible for backward compatibility with existing tests
KalmanState = KalmanTracker
