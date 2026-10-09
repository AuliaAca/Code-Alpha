"""Streamlit UI:  streamlit run Task1/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import add_task_paths  # noqa: E402

add_task_paths()

import streamlit as st  # noqa: E402
from translator.ui import render_translator  # noqa: E402

st.set_page_config(page_title="Language Translation Tool", page_icon="🌐")
render_translator()
