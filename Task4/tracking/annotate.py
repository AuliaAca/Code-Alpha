"""Drawing helpers: boxes, ``#id label`` tags and motion trails."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from common.contracts import TrackedObject

Point = Tuple[float, float]


def id_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, well-spread BGR colour per ID (golden-angle hue steps)."""
    hue = int((track_id * 137.508) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(frame: np.ndarray, tracks: Sequence[TrackedObject],
                trails: Dict[int, List[Point]] | None = None) -> np.ndarray:
    out = frame.copy()
    for tid, points in (trails or {}).items():
        if len(points) > 1:
            pts = np.array(points[-40:], np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, id_color(tid), 2, cv2.LINE_AA)
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        color = id_color(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"#{t.track_id} {t.label}"
        (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - h - 6, 0)
        cv2.rectangle(out, (x1, top), (x1 + w + 6, top + h + 6), color, -1)  # label background
        cv2.putText(out, tag, (x1 + 3, top + h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out
