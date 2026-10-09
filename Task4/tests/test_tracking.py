import sys
from pathlib import Path

import numpy as np
import pytest

from common.contracts import Detection
from tracking import ColorLabeler, MotionDetector, Sort, TrackingPipeline, build_detector
from tracking.kalman import BoxKalman
from tracking.sort import iou_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_demo_video import make_demo_video  # noqa: E402


def det(x, y, w=40, h=40, label="thing"):
    return Detection(label, 0.9, (x, y, x + w, y + h))


def test_iou_known_values():
    m = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 15, 10), (20, 20, 30, 30)])
    assert m[0].round(3).tolist() == [1.0, 0.333, 0.0]
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)


def test_kalman_learns_constant_velocity():
    kf = BoxKalman((0, 0, 40, 40))
    for step in range(1, 15):  # box moves +5 px/frame
        kf.predict()
        kf.update((5 * step, 0, 5 * step + 40, 40))
    x1 = kf.predict()[0]
    assert x1 == pytest.approx(75, abs=2.0)  # next position: 14*5 + 5


def test_ids_stay_stable_when_paths_cross():
    tracker, ids = Sort(min_hits=1), {}
    for f in range(30):
        # A moves right, B moves left, on different rows, so they cross in x
        out = tracker.update([det(10 + 6 * f, 20, label="A"), det(190 - 6 * f, 120, label="B")])
        for t in out:
            ids.setdefault(t.label, set()).add(t.track_id)
    assert all(len(v) == 1 for v in ids.values()) and len(ids) == 2


def test_track_survives_short_gap_and_dies_after_max_age():
    tracker = Sort(max_age=5, min_hits=1)
    first = tracker.update([det(10, 10)])[0].track_id
    for f in range(1, 4):                         # object hidden for 3 frames
        assert tracker.update([]) == []
    back = tracker.update([det(10, 10)])
    assert back[0].track_id == first              # same identity after the gap
    for _ in range(8):                            # hidden longer than max_age
        tracker.update([])
    assert tracker.update([det(10, 10)])[0].track_id != first


def test_label_is_majority_vote():
    tracker = Sort(min_hits=1)
    labels = ["car", "car", "truck", "car"]
    out = [tracker.update([det(10 + f, 10, label=l)])[0] for f, l in enumerate(labels)]
    assert out[-1].label == "car"


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory):
    return make_demo_video(tmp_path_factory.mktemp("vid") / "demo.mp4")


def test_pipeline_finds_one_track_per_object(demo_video, tmp_path):
    pipeline = TrackingPipeline(build_detector("motion", color_labels=True))
    out = tmp_path / "annotated.mp4"
    report = pipeline.run(demo_video, output=out)
    assert report.frames == 210
    assert report.label_counts == {"car": 1, "bus": 1, "person": 1, "ball": 1}
    assert report.mean_speed > 1.0 and out.stat().st_size > 10_000


def test_bad_source_raises():
    with pytest.raises(IOError):
        TrackingPipeline(MotionDetector()).run("/no/such/video.mp4")


def test_build_detector_choices():
    assert isinstance(build_detector("motion"), MotionDetector)
    with pytest.raises(ValueError):
        build_detector("magic")


def test_webcam_that_does_not_exist_gives_clear_error():
    with pytest.raises(IOError, match="webcam"):
        TrackingPipeline(MotionDetector()).run("97")


# Real YOLO check. Runs only when ultralytics AND the weights are present, so the
# suite stays offline (weights download on first `python Task4/cli.py` run).
from tracking.detectors import DEFAULT_WEIGHTS, yolo_installed  # noqa: E402


@pytest.mark.skipif(not (yolo_installed() and DEFAULT_WEIGHTS.exists()),
                    reason="ultralytics or Task4/models/yolov8n.pt not available")
def test_yolo_detects_real_objects():
    import cv2
    import ultralytics

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    labels = [d.label for d in build_detector("yolo").detect(image)]
    assert "bus" in labels and labels.count("person") >= 3
