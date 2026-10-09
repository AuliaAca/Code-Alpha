"""Small helpers shared by the Streamlit pages."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st


def session_dir() -> Path:
    """A temp folder per browser session for generated files (videos, audio)."""
    if "work_dir" not in st.session_state:
        st.session_state["work_dir"] = tempfile.mkdtemp(prefix="codelab_")
    return Path(st.session_state["work_dir"])


def save_upload(uploaded, folder: Path) -> Path:
    """Write a Streamlit upload to disk (OpenCV needs a real file path).

    The file is closed before use, which matters on Windows.
    """
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())
    return path
