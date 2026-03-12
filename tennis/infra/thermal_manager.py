"""
Thermal Manager — Adaptive inference throttling for sustained performance.

Monitors processing latency and adjusts inference depth:
- Full mode: all detections (ball + player + court + pose)
- Reduced mode: skip court recalibration + pose, run ball + player only
- Minimal mode: ball tracking only
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InferenceMode(str, Enum):
    FULL = "full"         # All models
    REDUCED = "reduced"   # Ball + player only
    MINIMAL = "minimal"   # Ball only


@dataclass
class ThermalState:
    """Current thermal/performance state."""
    mode: InferenceMode = InferenceMode.FULL
    avg_latency_ms: float = 0.0
    latency_budget_ms: float = 33.0  # 30fps = 33ms per frame
    frames_over_budget: int = 0
    total_frames: int = 0
    memory_pressure: bool = False


class ThermalManager:
    """
    Adaptive inference throttling based on processing latency.

    Automatically reduces inference depth when the system can't
    maintain real-time processing, and recovers when headroom returns.
    """

    def __init__(
        self,
        latency_budget_ms: float = 33.0,
        degradation_threshold: int = 10,
        recovery_threshold: int = 30,
    ):
        self.latency_budget_ms = latency_budget_ms
        self.degradation_threshold = degradation_threshold
        self.recovery_threshold = recovery_threshold

        self._mode = InferenceMode.FULL
        self._latencies: list[float] = []
        self._frames_over_budget = 0
        self._frames_under_budget = 0
        self._total_frames = 0
        self._window_size = 30  # Rolling window

    def record_latency(self, latency_ms: float) -> InferenceMode:
        """
        Record a frame's processing latency and return current mode.

        Call this after each frame is processed. The manager will
        automatically adjust the inference mode based on sustained
        over/under-budget frames.
        """
        self._total_frames += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._window_size:
            self._latencies.pop(0)

        if latency_ms > self.latency_budget_ms:
            self._frames_over_budget += 1
            self._frames_under_budget = 0
        else:
            self._frames_under_budget += 1
            self._frames_over_budget = 0

        # Degrade if too many frames over budget
        if self._frames_over_budget >= self.degradation_threshold:
            self._degrade()

        # Recover if enough frames under budget
        if self._frames_under_budget >= self.recovery_threshold:
            self._recover()

        return self._mode

    def get_state(self) -> ThermalState:
        """Get current thermal state."""
        avg = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        return ThermalState(
            mode=self._mode,
            avg_latency_ms=round(avg, 2),
            latency_budget_ms=self.latency_budget_ms,
            frames_over_budget=self._frames_over_budget,
            total_frames=self._total_frames,
        )

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    def _degrade(self):
        """Step down inference mode."""
        if self._mode == InferenceMode.FULL:
            self._mode = InferenceMode.REDUCED
            logger.warning("Thermal: degraded to REDUCED mode (avg=%.1fms, budget=%.1fms)",
                           self._avg_latency, self.latency_budget_ms)
        elif self._mode == InferenceMode.REDUCED:
            self._mode = InferenceMode.MINIMAL
            logger.warning("Thermal: degraded to MINIMAL mode")
        self._frames_over_budget = 0

    def _recover(self):
        """Step up inference mode."""
        if self._mode == InferenceMode.MINIMAL:
            self._mode = InferenceMode.REDUCED
            logger.info("Thermal: recovered to REDUCED mode")
        elif self._mode == InferenceMode.REDUCED:
            self._mode = InferenceMode.FULL
            logger.info("Thermal: recovered to FULL mode")
        self._frames_under_budget = 0

    @property
    def _avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0
