"""Task4 command line.

    python Task4/cli.py --source 0 --show                          # webcam, live window
    python Task4/cli.py --source my_video.mp4 --output tracked.mp4
    python Task4/cli.py --demo --output tracked.mp4                # bundled demo scene

--detector auto (default) uses YOLO when ultralytics is installed, else the
OpenCV motion detector.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.console import use_utf8_console  # noqa: E402
from common.paths import add_task_paths  # noqa: E402

use_utf8_console()
add_task_paths()
from make_demo_video import DEMO_PATH, make_demo_video  # noqa: E402
from tracking import Sort, TrackingPipeline, build_detector  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index (0, 1, ...) or a video file path")
    p.add_argument("--demo", action="store_true", help="use the synthetic demo scene")
    p.add_argument("--detector", choices=["auto", "yolo", "motion"], default="auto")
    p.add_argument("--weights", default=None, help="YOLO weights (default Task4/models/yolov8n.pt)")
    p.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--output", type=Path, help="write the annotated video here (.mp4)")
    p.add_argument("--show", action="store_true", help="live window (q, Esc or X to quit)")
    p.add_argument("--max-age", type=int, default=30, help="frames a lost track survives")
    p.add_argument("--max-frames", type=int)
    args = p.parse_args(argv)

    source = args.source
    detector_kind = args.detector
    if args.demo:
        source = DEMO_PATH if DEMO_PATH.exists() else make_demo_video(DEMO_PATH)
        # The demo scene is coloured shapes, which YOLO is not trained on.
        detector_kind = "motion" if args.detector == "auto" else args.detector

    kwargs = {"weights": args.weights} if args.weights else {}
    try:
        detector = build_detector(detector_kind, color_labels=args.demo,
                                  confidence=args.confidence, **kwargs)
        print(f"detector: {type(detector).__name__}  source: {source}")
        report = TrackingPipeline(detector, Sort(max_age=args.max_age)).run(
            source, args.output, args.show, args.max_frames)
    except (IOError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"frames: {report.frames}  tracks: {len(report.tracks)}  "
          f"objects: {report.label_counts}  mean speed: {report.mean_speed:.2f}px/frame")
    if args.output:
        print("saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
