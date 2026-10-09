"""Create a synthetic street scene (no download needed) to demo the tracker.

Four coloured objects cross the frame in separate lanes; the person crosses
both lanes and the ball bounces, so paths intersect and IDs must stay apart. Ground truth is also returned, which is
how the tests measure identity switches.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

W, H, FPS = 640, 360, 30
DEMO_PATH = Path(__file__).resolve().parent / "samples" / "demo_scene.mp4"
# name -> (BGR colour, size (w,h), start (x,y), velocity (vx,vy), enters at frame, shape)
OBJECTS = {
    "car":    ((200, 90, 30),  (90, 40), (-100, 250), (4.2, 0.0),  20, "rect"),
    "bus":    ((40, 40, 210),  (130, 50), (740, 120), (-3.6, 0.0), 20, "rect"),
    "person": ((30, 140, 240), (26, 70),  (120, -80),  (1.1, 2.6), 25, "rect"),
    "ball":   ((60, 200, 70),  (30, 30),  (-40, 60),   (3.4, 0.0), 30, "ball"),
}


def _background() -> np.ndarray:
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):  # dusk gradient
        bg[y] = (60 + y // 9, 45 + y // 12, 40 + y // 14)
    cv2.rectangle(bg, (0, 110), (W, 310), (55, 55, 60), -1)          # two-lane road
    for x in range(0, W, 60):
        cv2.line(bg, (x, 250), (x + 30, 250), (170, 170, 175), 2)     # lane marks
    return bg


def position(name: str, frame: int) -> Tuple[float, float, float, float]:
    """Ground-truth box of ``name`` at ``frame`` (may be off-screen)."""
    _, (w, h), (x0, y0), (vx, vy), enter, shape = OBJECTS[name]
    t = max(frame - enter, 0)
    x, y = x0 + vx * t, y0 + vy * t
    if shape == "ball":  # bouncing
        y += abs(np.sin(t / 9.0)) * -32 + 250   # bounces on the ground below the road
    return x, y, x + w, y + h


def render_frame(bg: np.ndarray, frame: int, rng: np.random.Generator) -> np.ndarray:
    img = bg.copy()
    for name, (color, _, _, _, enter, shape) in OBJECTS.items():
        if frame < enter:
            continue
        x1, y1, x2, y2 = (int(v) for v in position(name, frame))
        if shape == "ball":
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    noise = rng.normal(0, 3, img.shape)  # sensor noise, so detection is not trivial
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_demo_video(path: Path, frames: int = 210, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg, rng = _background(), np.random.default_rng(seed)
    for f in range(frames):
        writer.write(render_frame(bg, f, rng))
    writer.release()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO_PATH
    print("wrote", make_demo_video(out))
