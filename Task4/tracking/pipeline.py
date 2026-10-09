"""Video in -> detect -> track -> annotated video / SceneReport out."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from common.contracts import SceneReport, TrackedObject

from .annotate import draw_tracks
from .detectors import Detector
from .sort import Sort

Source = Union[int, str, Path]


class TrackingPipeline:
    def __init__(self, detector: Detector, tracker: Optional[Sort] = None,
                 min_track_frames: int = 8):
        self.detector = detector
        self.tracker = tracker or Sort()
        # Tracks seen fewer times than this are treated as noise in the report.
        self.min_track_frames = min_track_frames

    @staticmethod
    def _open(source: Source) -> cv2.VideoCapture:
        """Open a webcam index ("0", "1", ...) or a video file."""
        if str(source).isdigit():
            index = int(source)
            # DirectShow opens webcams faster and more reliably than the default on Windows.
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else None
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise IOError(f"Cannot open webcam {index}. Is it connected and not used by "
                              "another app? Try --source 1 for a second camera.")
            return cap
        if not Path(source).exists():
            raise IOError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise IOError(f"Cannot read video file: {source} (unsupported format?)")
        return cap

    def frames(self, cap: cv2.VideoCapture, max_frames: Optional[int] = None
               ) -> Iterator[Tuple[int, np.ndarray, List[TrackedObject]]]:
        """Yield ``(index, frame, tracked_objects)`` for every frame."""
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame, self.tracker.update(self.detector.detect(frame))
            index += 1

    WINDOW = "CodeLab tracking (press q to quit)"

    def run(self, source: Source, output: Optional[Path] = None, show: bool = False,
            max_frames: Optional[int] = None, codec: str = "mp4v",
            on_frame: Optional[Callable[[int, np.ndarray, List[TrackedObject]], None]] = None
            ) -> SceneReport:
        cap = self._open(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = None
        labels: Dict[int, List[str]] = defaultdict(list)
        trails: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        n_frames, tick, live_fps = 0, time.perf_counter(), 0.0
        try:
            for index, frame, tracked in self.frames(cap, max_frames):
                n_frames = index + 1
                for t in tracked:
                    labels[t.track_id].append(t.label)
                    trails[t.track_id].append(t.center)
                if output or show or on_frame:
                    drawn = draw_tracks(frame, tracked, trails)
                    if output:
                        if writer is None:  # size is only known after the first frame
                            Path(output).parent.mkdir(parents=True, exist_ok=True)
                            h, w = drawn.shape[:2]
                            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec),
                                                     fps, (w, h))
                            if not writer.isOpened():
                                raise IOError(f"Cannot write video {output} with codec {codec!r}")
                        writer.write(drawn)
                    if show:
                        now = time.perf_counter()  # smoothed processing speed for the overlay
                        live_fps = 0.9 * live_fps + 0.1 / max(now - tick, 1e-6)
                        tick = now
                        cv2.putText(drawn, f"{live_fps:4.1f} fps  |  {len(tracked)} tracked",
                                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                        cv2.imshow(self.WINDOW, drawn)
                        key = cv2.waitKey(1) & 0xFF
                        # stop on q / Esc, or when the window's X button closed it
                        if key in (ord("q"), 27) or cv2.getWindowProperty(
                                self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    if on_frame:
                        on_frame(index, drawn, tracked)
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        return self._report(n_frames, fps, labels, trails)

    def _report(self, n_frames, fps, labels, trails) -> SceneReport:
        keep = {tid for tid, seen in labels.items() if len(seen) >= self.min_track_frames}
        speeds = [np.hypot(*np.diff(np.array(pts), axis=0).T).mean()
                  for tid, pts in trails.items() if tid in keep and len(pts) > 1]
        return SceneReport(
            frames=n_frames, fps=fps,
            tracks={tid: max(set(labels[tid]), key=labels[tid].count) for tid in keep},
            trails={tid: trails[tid] for tid in keep},
            mean_speed=float(np.mean(speeds)) if speeds else 0.0)
