"""Generate a short synthetic tennis clip for demos and smoke tests.

The clip is drawn with OpenCV: a blue hard court with white lines, two player
figures moving along the baselines and a yellow-green ball travelling between
them. It is *not* real match footage. It exists so the full upload -> analyze
-> results flow can be exercised (and screenshotted) without shipping video files.

Usage:
    python scripts/make_demo_clip.py demo_clip.mp4 [--seconds 12]
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30

# Court corners in image space (simple perspective: far baseline is narrower).
FAR_L, FAR_R, FAR_Y = (430, 140), (850, 140), 140
NEAR_L, NEAR_R, NEAR_Y = (170, 640), (1110, 640), 640


def _lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)


def _court_point(u: float, v: float) -> tuple[int, int]:
    """Map court coords (u: 0=left..1=right, v: 0=far..1=near) to pixels."""
    left = _lerp(FAR_L, NEAR_L, v)
    right = _lerp(FAR_R, NEAR_R, v)
    return _lerp(left, right, u)


def draw_court(frame: np.ndarray) -> None:
    frame[:] = (60, 140, 60)  # green surround (BGR)
    poly = np.array([FAR_L, FAR_R, NEAR_R, NEAR_L], np.int32)
    cv2.fillPoly(frame, [poly], (150, 90, 30))  # blue hard court
    white = (245, 245, 245)
    cv2.polylines(frame, [poly], True, white, 3)
    for u in (0.125, 0.875):  # singles sidelines
        cv2.line(frame, _court_point(u, 0), _court_point(u, 1), white, 2)
    for v in (0.25, 0.75):  # service lines
        cv2.line(frame, _court_point(0.125, v), _court_point(0.875, v), white, 2)
    cv2.line(frame, _court_point(0.5, 0.25), _court_point(0.5, 0.75), white, 2)
    # net
    cv2.line(frame, _court_point(-0.03, 0.5), _court_point(1.03, 0.5), (40, 40, 40), 5)


def draw_player(frame: np.ndarray, x: int, y: int, scale: float) -> None:
    h = int(90 * scale)
    w = int(28 * scale)
    cv2.rectangle(frame, (x - w // 2, y - h), (x + w // 2, y), (30, 30, 200), -1)
    cv2.circle(frame, (x, y - h - int(10 * scale)), int(10 * scale), (120, 160, 210), -1)


def make_clip(path: str, seconds: float) -> None:
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(seconds * FPS)
    rally_frames = 45  # one baseline-to-baseline flight
    for i in range(n):
        frame = np.zeros((H, W, 3), np.uint8)
        draw_court(frame)
        t = i / FPS
        far_u = 0.5 + 0.3 * math.sin(t * 1.3)
        near_u = 0.5 + 0.3 * math.sin(t * 1.1 + 1.0)
        fx, fy = _court_point(far_u, -0.02)
        nx, ny = _court_point(near_u, 1.02)
        draw_player(frame, fx, fy, 0.6)
        draw_player(frame, nx, ny, 1.0)

        leg = (i // rally_frames) % 2
        p = (i % rally_frames) / rally_frames
        start = (near_u, 0.98) if leg == 0 else (far_u, 0.02)
        end = (far_u, 0.02) if leg == 0 else (near_u, 0.98)
        u = start[0] + (end[0] - start[0]) * p
        v = start[1] + (end[1] - start[1]) * p
        bx, by = _court_point(u, v)
        by -= int(120 * math.sin(math.pi * p))  # arc
        radius = int(6 + 5 * v)
        cv2.circle(frame, (bx, by), radius, (40, 230, 210), -1)  # yellow-green ball
        out.write(frame)
    out.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", help="Path of the .mp4 file to write")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    make_clip(args.output, args.seconds)
    print(f"wrote {args.output}")
