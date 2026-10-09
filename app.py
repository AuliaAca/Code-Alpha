"""CodeLab dashboard - all four tasks in one web app.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.paths import add_task_paths  # noqa: E402

add_task_paths()

import streamlit as st  # noqa: E402
from faq_bot.ui import render_faq_bot  # noqa: E402
from hub.ui import render_connected_demo  # noqa: E402
from music_gen.ui import render_music  # noqa: E402
from tracking.ui import render_tracking  # noqa: E402
from translator.ui import render_translator  # noqa: E402

st.set_page_config(page_title="CodeAlpha CodeLab", page_icon="🧪", layout="wide")


def render_home() -> None:
    st.title("🧪 CodeAlpha CodeLab")
    st.markdown("Four AI projects in one app. Pick a task in the sidebar.")
    st.image(str(ROOT / "docs" / "assets" / "metro_map.png"))


PAGES = {
    "Home": render_home,
    "1 · Translator": render_translator,
    "2 · FAQ Chatbot": render_faq_bot,
    "3 · Music Generator": render_music,
    "4 · Object Tracking": render_tracking,
    "🔗 Connected demo": render_connected_demo,
}

choice = st.sidebar.radio("Task", list(PAGES), key="nav")
st.sidebar.caption("Live webcam tracking: `python Task4/cli.py --source 0 --show`")
PAGES[choice]()
