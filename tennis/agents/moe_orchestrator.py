"""
MoE Orchestrator — Mixture-of-Experts central coordinator.

Runs 7 specialized agents in dependency order:
  Court → Ball + Player → Biomechanics + Strategy → Coaching → Voice

Each agent receives outputs from prior agents via a shared context dict.
The orchestrator returns a unified MoEResult containing all coaching data.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .court_agent import CourtAgent
from .ball_agent import BallAgent
from .player_agent import PlayerAgent
from .biomechanics_agent import BiomechanicsAgent
from .strategy_agent import StrategyAgent
from .coaching_agent import CoachingAgent
from .voice_agent import VoiceAgent

logger = logging.getLogger(__name__)


@dataclass
class MoEResult:
    """Full output from the MoE pipeline."""
    court: dict = field(default_factory=dict)
    ball: dict = field(default_factory=dict)
    player: dict = field(default_factory=dict)
    biomechanics: dict = field(default_factory=dict)
    strategy: dict = field(default_factory=dict)
    coaching_report: dict = field(default_factory=dict)
    voice_script: dict = field(default_factory=dict)

    # Synthesized outputs for the frontend
    frame_overlays: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    agents_run: list = field(default_factory=list)


class MoEOrchestrator:
    """
    Orchestrates all 7 specialist agents in dependency order.

    Dependency graph:
      Stage 1 (parallel): CourtAgent
      Stage 2 (parallel): BallAgent, PlayerAgent
      Stage 3 (parallel): BiomechanicsAgent, StrategyAgent
      Stage 4 (sequential): CoachingAgent
      Stage 5 (sequential): VoiceAgent
    """

    AGENT_TIMEOUT = 20.0  # seconds per agent/stage

    def __init__(self):
        self._court   = CourtAgent()
        self._ball    = BallAgent()
        self._player  = PlayerAgent()
        self._bio     = BiomechanicsAgent()
        self._strat   = StrategyAgent()
        self._coach   = CoachingAgent()
        self._voice   = VoiceAgent()

    def run(self, frames: list, base_context: dict) -> MoEResult:
        """
        Run the full MoE pipeline synchronously.

        Args:
            frames: list of OpenCV frames (BGR numpy arrays), may be empty
            base_context: dict with {fps, duration_seconds, match_type,
                          video_fingerprint, motion_timeline, ...}
        Returns:
            MoEResult with all agent outputs
        """
        t0 = time.monotonic()
        result = MoEResult()
        ctx = dict(base_context)

        # ── Stage 1: Court ──────────────────────────────────────
        result.court = self._safe_run(self._court, frames, ctx, "Court")
        ctx["court"] = result.court
        result.agents_run.append("court")

        # ── Stage 2: Ball + Player (independent) ────────────────
        result.ball = self._safe_run(self._ball, frames, ctx, "Ball")
        ctx["ball"] = result.ball
        result.agents_run.append("ball")

        result.player = self._safe_run(self._player, frames, ctx, "Player")
        ctx["player"] = result.player
        result.agents_run.append("player")

        # ── Stage 3: Biomechanics + Strategy (independent) ──────
        result.biomechanics = self._safe_run(self._bio, frames, ctx, "Biomechanics")
        ctx["biomechanics"] = result.biomechanics
        result.agents_run.append("biomechanics")

        result.strategy = self._safe_run(self._strat, frames, ctx, "Strategy")
        ctx["strategy"] = result.strategy
        result.agents_run.append("strategy")

        # ── Stage 4: Coaching ────────────────────────────────────
        result.coaching_report = self._safe_run(self._coach, frames, ctx, "Coaching")
        ctx["coaching"] = result.coaching_report
        result.agents_run.append("coaching")

        # ── Stage 5: Voice ───────────────────────────────────────
        result.voice_script = self._safe_run(self._voice, frames, ctx, "Voice")
        result.agents_run.append("voice")

        # ── Synthesize frame overlays ────────────────────────────
        result.frame_overlays = self._build_frame_overlays(result)

        result.elapsed_seconds = round(time.monotonic() - t0, 2)
        logger.info(
            "MoE pipeline complete in %.1fs — agents: %s",
            result.elapsed_seconds, ", ".join(result.agents_run)
        )
        return result

    def _safe_run(self, agent, frames: list, ctx: dict, name: str) -> dict:
        """Run an agent with error isolation — never let one agent break the pipeline."""
        try:
            t = time.monotonic()
            out = agent.analyze(frames, ctx)
            logger.info("Agent %s completed in %.2fs", name, time.monotonic() - t)
            return out
        except Exception as exc:
            logger.exception("Agent %s failed: %s", name, exc)
            return {"error": str(exc), "agent": name}

    def _build_frame_overlays(self, result: MoEResult) -> list:
        """
        Merge frame annotations from biomechanics + ball trajectory
        into a single timed overlay track for the frontend VideoPlayer.

        Each entry:
        {
          timestamp: float,
          duration: float,
          label: str,
          color: str,
          overlays: [
            {type, x, y, ...},
            ...
          ],
          coaching_text: str (shown in CoachingPanel)
        }
        """
        overlays = []

        # Biomechanics annotations (coaching markers)
        frame_annotations = result.biomechanics.get("frame_annotations", [])
        for ann in frame_annotations:
            overlays.append({
                "timestamp": ann["timestamp"],
                "duration": ann.get("duration", 3.0),
                "label": ann.get("label", ""),
                "color": ann.get("color", "#f59e0b"),
                "severity": ann.get("severity", "medium"),
                "overlays": ann.get("overlays", []),
                "coaching_text": ann.get("coaching_text", ""),
                "tip": ann.get("tip", ""),
                "source": "biomechanics",
            })

        # Ball trajectory markers (speed badges on fast shots)
        trajectory = result.ball.get("trajectory", [])
        speeds = result.ball.get("speeds_kmh", [])
        fps = 30.0

        for i, speed in enumerate(speeds[:20]):
            if speed > 140 and i < len(trajectory):
                ball = trajectory[i]
                overlays.append({
                    "timestamp": round(ball["frame_idx"] / fps, 2),
                    "duration": 1.5,
                    "label": f"{speed:.0f} km/h",
                    "color": "#10b981",
                    "severity": "info",
                    "overlays": [
                        {"type": "speed_badge", "x": ball["x"], "y": ball["y"],
                         "label": f"{speed:.0f} km/h", "color": "#10b981"},
                    ],
                    "coaching_text": f"Ball speed: {speed:.0f} km/h",
                    "source": "ball",
                })

        # Sort chronologically
        overlays.sort(key=lambda o: o["timestamp"])
        return overlays
