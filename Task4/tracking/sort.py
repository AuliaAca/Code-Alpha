"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Per frame:  predict every track (Kalman) -> match detections to predictions with
the Hungarian algorithm on an IoU cost -> update matched tracks, create tracks
for unmatched detections, delete tracks that were unseen for ``max_age`` frames.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.contracts import Detection, TrackedObject

from .kalman import BoxKalman


def iou_matrix(a: Sequence, b: Sequence) -> np.ndarray:
    """Pairwise intersection-over-union between two lists of boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class _Track:
    def __init__(self, track_id: int, det: Detection):
        self.id = track_id
        self.kf = BoxKalman(det.box)
        self.labels = Counter({det.label: 1})   # majority vote keeps the label stable
        self.confidence = det.confidence
        self.hits = 1
        self.since_update = 0

    def update(self, det: Detection) -> None:
        self.kf.update(det.box)
        self.labels[det.label] += 1
        self.confidence = det.confidence
        self.hits += 1
        self.since_update = 0

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]


class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.2):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self._tracks: List[_Track] = []
        self._next_id = 1   # instance state (not global) so trackers are independent
        self._frame = 0

    def update(self, detections: Sequence[Detection]) -> List[TrackedObject]:
        self._frame += 1
        # 1. predict where every existing track should be now
        predicted = [t.kf.predict() for t in self._tracks]
        for t in self._tracks:
            t.since_update += 1

        # 2. associate detections with predictions
        matches, unmatched = self._associate(predicted, detections)
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(detections[det_idx])
        for det_idx in unmatched:
            self._tracks.append(_Track(self._next_id, detections[det_idx]))
            self._next_id += 1

        # 3. report confirmed tracks; drop stale ones
        out = [TrackedObject(t.id, t.label, t.confidence, tuple(map(float, t.kf.box)))
               for t in self._tracks
               if t.since_update == 0 and (t.hits >= self.min_hits or self._frame <= self.min_hits)]
        self._tracks = [t for t in self._tracks if t.since_update <= self.max_age]
        return out

    def _associate(self, predicted, detections):
        """Hungarian matching on 1 - IoU; pairs below the IoU threshold are rejected."""
        if not predicted:
            return [], list(range(len(detections)))
        iou = iou_matrix(predicted, [d.box for d in detections])
        rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
        matches, matched_dets = [], set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_dets.add(c)
        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched
