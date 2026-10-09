"""Streamlit UI:  streamlit run Task3/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import add_task_paths  # noqa: E402

add_task_paths()

import streamlit as st  # noqa: E402
from music_gen.ui import render_music  # noqa: E402

st.set_page_config(page_title="CodeLab Music Generator", page_icon="🎹")
render_music()
