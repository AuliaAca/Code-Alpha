"""Constant-velocity Kalman filter for a bounding box (pure NumPy).

State  x = [cx, cy, s, r, vx, vy, vs]   s = box area, r = aspect ratio (w/h)
Measurement z = [cx, cy, s, r]

Using area and aspect ratio (instead of width/height) is the parametrisation
from the original SORT paper; it keeps the aspect ratio constant over time.
"""
from __future__ import annotations

import numpy as np


def box_to_z(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / max(h, 1e-6)], dtype=float)


def x_to_box(x: np.ndarray) -> tuple[float, float, float, float]:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return (x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2)


class BoxKalman:
    def __init__(self, box):
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)
        # Transition: position += velocity each frame (aspect ratio has no velocity).
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)  # we observe the first four state entries
        # Noise settings follow the reference SORT implementation.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])  # unknown initial velocity
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])

    def predict(self):
        if self.x[6] + self.x[2] <= 0:  # area must never go negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_box(self.x)

    def update(self, box) -> None:
        z = box_to_z(box)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self):
        return x_to_box(self.x)
