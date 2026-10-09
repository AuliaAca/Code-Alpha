"""Streamlit view for detection + tracking (used by Task4/app.py and the dashboard).

Real-time webcam tracking runs in a desktop window via the CLI
(``python Task4/cli.py --source 0 --show``); the browser page handles video
files and single webcam snapshots.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from common.contracts import TrackedObject
from common.webapp import save_upload, session_dir

from .annotate import draw_tracks
from .detectors import build_detector, yolo_installed
from .pipeline import TrackingPipeline

DEMO = "Demo scene (no download)"
UPLOAD = "Upload a video"
SNAPSHOT = "Webcam snapshot (detection)"


@st.cache_resource
def _yolo():
    return build_detector("yolo")


def _demo_video() -> Path:
    from make_demo_video import DEMO_PATH, make_demo_video
    return DEMO_PATH if DEMO_PATH.exists() else make_demo_video(DEMO_PATH)


def track_for_browser(video: Path, detector, max_frames: int, out_stem: Path, progress=None):
    """Run the pipeline and write a video the browser can play.

    VP8/WebM plays in every modern browser; if this OpenCV build cannot write it,
    fall back to MP4 (download only).
    """
    total = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    total = min(total, max_frames)

    def on_frame(index, _frame, _tracked):
        if progress:
            progress.progress(min((index + 1) / total, 1.0), text=f"frame {index + 1}/{total}")

    for codec, ext, playable in (("VP80", ".webm", True), ("mp4v", ".mp4", False)):
        out = out_stem.with_suffix(ext)
        try:
            report = TrackingPipeline(detector).run(video, output=out, max_frames=max_frames,
                                                    codec=codec, on_frame=on_frame)
            return report, out, playable
        except IOError:
            if codec == "mp4v":
                raise
    raise RuntimeError("unreachable")


def _render_video_tracking(source: str) -> None:
    if source == DEMO:
        video, detector_kind = _demo_video(), "demo"
        st.caption("A generated street scene: car, bus, person and a bouncing ball.")
    else:
        uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"],
                                    key="t4_upload")
        if not uploaded:
            return
        video = save_upload(uploaded, session_dir())
        detector_kind = st.selectbox(
            "Detector", ["auto", "yolo", "motion"], key="t4_det",
            help="auto = YOLO if installed, else motion (static camera only)")
    max_frames = st.slider("Max frames to process", 30, 1500, 300, step=30, key="t4_max")

    if st.button("Run tracking", type="primary", key="t4_go"):
        try:
            if detector_kind == "demo":
                detector = build_detector("motion", color_labels=True)
            elif detector_kind in ("auto", "yolo") and yolo_installed():
                detector = _yolo()
            else:
                detector = build_detector(detector_kind)
            bar = st.progress(0.0, text="starting…")
            st.session_state["t4_result"] = track_for_browser(
                video, detector, max_frames, session_dir() / f"tracked_{video.stem}", bar)
            bar.empty()
        except (IOError, ImportError) as exc:
            st.error(str(exc))

    result = st.session_state.get("t4_result")
    if result:
        report, out, playable = result
        video_col, info_col = st.columns([3, 1])
        if playable:
            video_col.video(out.read_bytes(), format="video/webm")
        else:
            video_col.info("This OpenCV build cannot write browser video; download the MP4 instead.")
        info_col.metric("Frames", report.frames)
        info_col.metric("Tracked objects", len(report.tracks))
        info_col.metric("Mean speed", f"{report.mean_speed:.1f} px/frame")
        info_col.write({label: count for label, count in sorted(report.label_counts.items())})
        info_col.download_button("Download video", out.read_bytes(), out.name, key="t4_dl")


def _render_snapshot() -> None:
    if not yolo_installed():
        st.info("Snapshots need YOLO: `pip install ultralytics`.")
        return
    shot = st.camera_input("Take a picture", key="t4_cam")
    if not shot:
        return
    frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    detections = _yolo().detect(frame)
    # A single image has no motion, so each detection simply gets a number.
    objects = [TrackedObject(i, d.label, d.confidence, d.box) for i, d in enumerate(detections, 1)]
    st.image(cv2.cvtColor(draw_tracks(frame, objects), cv2.COLOR_BGR2RGB))
    st.write({o.track_id: f"{o.label} ({o.confidence:.0%})" for o in objects} or "Nothing detected.")


def render_tracking() -> None:
    st.title("🎯 Object Detection & Tracking")
    st.caption("YOLO (or motion) detection + SORT tracking with persistent IDs. "
               "For live webcam tracking run `python Task4/cli.py --source 0 --show`.")
    source = st.radio("Input", [DEMO, UPLOAD, SNAPSHOT], horizontal=True, key="t4_src")
    if source == SNAPSHOT:
        _render_snapshot()
    else:
        if st.session_state.get("t4_last_src") != source:   # new input -> clear old result
            st.session_state.pop("t4_result", None)
            st.session_state["t4_last_src"] = source
        _render_video_tracking(source)
