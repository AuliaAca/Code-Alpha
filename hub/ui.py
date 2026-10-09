"""Streamlit page for the connected pipeline: video -> words -> music."""
from __future__ import annotations

import streamlit as st

from common.paths import add_task_paths
from common.webapp import session_dir

add_task_paths()
from translator.languages import LANGUAGES, language_name  # noqa: E402
from translator.errors import TranslationError  # noqa: E402
from translator.service import is_offline_result  # noqa: E402

from .pipeline import CodeLabHub  # noqa: E402


@st.cache_resource
def _hub() -> CodeLabHub:
    return CodeLabHub()


def render_connected_demo() -> None:
    st.title("🔗 Connected demo")
    st.markdown("**Task 4** tracks the demo video → **Task 1** describes what it saw in your "
                "language → **Task 3** turns the objects and their speed into a melody.")
    codes = list(LANGUAGES)
    lang = st.selectbox("Narration language", codes, index=codes.index("id"),
                        format_func=language_name, key="hub_lang")

    if st.button("Run the pipeline", type="primary", key="hub_go"):
        hub, folder = _hub(), session_dir()
        try:
            with st.status("Running…", expanded=True) as status:
                st.write("Task 4: tracking objects…")
                try:
                    video = folder / "connected_tracked.webm"
                    report = hub.track_scene(output=video, codec="VP80")
                except IOError:  # OpenCV without VP8: keep going without the preview
                    video, report = None, hub.track_scene()
                st.write("Task 1: narrating…")
                try:
                    english, translated = hub.narrate(report, lang)
                except TranslationError as exc:
                    english, translated = str(exc), None
                st.write("Task 3: composing…")
                _, midi, wav = hub.sonify(report, folder, "connected_music")
                status.update(label="Done", state="complete")
            st.session_state["hub_result"] = (video, report, english, translated, midi, wav)
        except FileNotFoundError as exc:  # no trained music model
            st.error(str(exc))

    result = st.session_state.get("hub_result")
    if not result:
        return
    video, report, english, translated, midi, wav = result
    left, right = st.columns([3, 2])
    if video:
        left.video(video.read_bytes(), format="video/webm")
    right.markdown(f"**Seen:** {report.label_counts}")
    if translated is None:
        right.warning(english)  # holds the error message
    else:
        right.markdown(f"**English:** {english}")
        right.markdown(f"**{language_name(translated.target)}:** {translated.text}")
        right.caption(f"via {translated.provider}")
        if is_offline_result(translated):
            right.warning("Offline word-by-word fallback (no internet).")
    right.audio(wav.read_bytes(), format="audio/wav")
    right.download_button("Download MIDI", midi.read_bytes(), midi.name, "audio/midi", key="hub_dm")
