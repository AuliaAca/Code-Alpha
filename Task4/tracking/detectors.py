"""Detectors share one tiny interface: ``detect(frame) -> list[Detection]``.

* :class:`YoloDetector`   - pre-trained YOLO via ``ultralytics`` (80 COCO classes:
  person, car, bus, dog, bottle, ...). Weights download automatically the first time.
* :class:`MotionDetector` - OpenCV background subtraction. No model download,
  so the whole pipeline runs anywhere; finds anything that moves (static camera).

Use :func:`build_detector` instead of constructing them directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, List, Optional, Protocol

import cv2
import numpy as np

from common.contracts import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "yolov8n.pt"   # ~6 MB, downloaded on first use


def yolo_installed() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


class YoloDetector:
    """Wraps an ultralytics YOLO model (default ``yolov8n.pt``, COCO classes)."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, confidence: float = 0.35,
                 classes: Optional[List[int]] = None):
        if not yolo_installed():
            raise ImportError("YOLO needs the ultralytics package: pip install ultralytics")
        from ultralytics import YOLO

        weights = Path(weights)
        weights.parent.mkdir(parents=True, exist_ok=True)
        # A known file name (yolov8n.pt, yolo11n.pt, ...) that is not on disk yet is
        # downloaded by ultralytics to exactly this path.
        self._model = YOLO(str(weights))
        self._confidence, self._classes = confidence, classes

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # agnostic_nms: one box per object even when YOLO is unsure between two
        # classes (a bus also scored as "truck" would otherwise become two tracks)
        result = self._model.predict(frame, conf=self._confidence, classes=self._classes,
                                     agnostic_nms=True, verbose=False)[0]
        names = result.names
        return [Detection(names[int(c)], float(p), tuple(map(float, b)))
                for b, p, c in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy())]


# --- motion detector -----------------------------------------------------

Labeler = Callable[[np.ndarray, np.ndarray, tuple], str]  # (frame, mask, box) -> label


class ColorLabeler:
    """Names a blob by its dominant hue. Hue is OpenCV's 0-179 scale."""

    DEFAULT = {"ball": (35, 85), "car": (95, 130), "person": (8, 25)}  # + red -> "bus"

    def __init__(self, hue_ranges: Optional[dict] = None, default: str = "bus"):
        self.hue_ranges, self.default = hue_ranges or self.DEFAULT, default

    def __call__(self, frame: np.ndarray, mask: np.ndarray, box: tuple) -> str:
        x1, y1, x2, y2 = (int(v) for v in box)
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        inside = (mask[y1:y2, x1:x2] > 0) & (hsv[..., 1] > 80)  # saturated foreground only
        if not inside.any():
            return self.default
        hue = float(np.median(hsv[..., 0][inside]))
        for label, (lo, hi) in self.hue_ranges.items():
            if lo <= hue <= hi:
                return label
        return self.default


class MotionDetector:
    def __init__(self, min_area: int = 250, labeler: Optional[Labeler] = None,
                 history: int = 100, warmup: int = 10, learning_rate: float = 0.005):
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=40,
                                                              detectShadows=False)
        # With OpenCV's defaults a pixel covered by a slow object for ~10 frames is
        # "absorbed" into the background, so large objects lose their trailing half.
        # A lower ratio + fixed low learning rate keeps them whole (static camera assumed).
        self._subtractor.setBackgroundRatio(0.7)
        self._learning_rate = learning_rate
        self._min_area, self._labeler = min_area, labeler
        self._warmup, self._seen = warmup, 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        mask = self._subtractor.apply(cv2.GaussianBlur(frame, (5, 5), 0), learningRate=self._learning_rate)
        self._seen += 1
        if self._seen <= self._warmup:  # let the background model settle first
            return []
        # open removes speckle noise, close fills holes inside a moving object
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            box = (float(x), float(y), float(x + w), float(y + h))
            label = self._labeler(frame, mask, box) if self._labeler else "object"
            detections.append(Detection(label, 1.0, box))
        return detections


def build_detector(kind: str = "auto", weights: str | Path = DEFAULT_WEIGHTS,
                   color_labels: bool = False, confidence: float = 0.35) -> Detector:
    """Factory used by the CLI, the hub and the web app (one place, DRY).

    kind: ``"auto"`` (YOLO when ultralytics is installed, else motion),
          ``"yolo"`` or ``"motion"``.
    color_labels: name motion blobs by colour - only meaningful for the synthetic
          demo scene, so real videos get the neutral label ``"object"``.
    """
    if kind not in {"auto", "yolo", "motion"}:
        raise ValueError(f"Unknown detector {kind!r}; use auto, yolo or motion.")
    if kind == "yolo" or (kind == "auto" and yolo_installed()):
        return YoloDetector(weights, confidence)
    return MotionDetector(labeler=ColorLabeler() if color_labels else None)
