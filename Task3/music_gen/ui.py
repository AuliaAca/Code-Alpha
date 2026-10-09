"""Streamlit view for the music generator (used by Task3/app.py and the dashboard)."""
from __future__ import annotations

import streamlit as st

from common.webapp import session_dir

from .export import tokens_to_midi, tokens_to_wav
from .generate import DEFAULT_CHECKPOINT, MusicGenerator


@st.cache_resource
def _generator() -> MusicGenerator:
    return MusicGenerator.load(DEFAULT_CHECKPOINT)


def render_music() -> None:
    st.title("🎹 Music Generator (LSTM)")
    st.caption("A 2-layer LSTM trained on 150 Bach chorales writes a new melody note by note.")
    if not DEFAULT_CHECKPOINT.exists():
        st.error("No trained model found. Run `python Task3/cli.py train` first.")
        return

    c1, c2, c3 = st.columns(3)
    length = c1.slider("Notes", 16, 128, 64, step=8, key="t3_len")
    temperature = c2.slider("Temperature", 0.5, 1.5, 0.9, step=0.1, key="t3_temp",
                            help="Low = safe and repetitive, high = adventurous")
    bpm = c3.slider("Tempo (BPM)", 60, 140, 84, step=4, key="t3_bpm")
    seed = st.number_input("Random seed (same seed = same melody)", 0, 99999, 7, key="t3_seed")

    if st.button("Generate", type="primary", key="t3_go"):
        with st.spinner("Composing…"):
            tokens = _generator().generate(length, temperature, rng_seed=int(seed))
            folder = session_dir()
            midi = tokens_to_midi(tokens, folder / f"melody_{seed}.mid", bpm)
            wav = tokens_to_wav(tokens, folder / f"melody_{seed}.wav", bpm)
        st.session_state["t3_files"] = (midi, wav, len(tokens))

    files = st.session_state.get("t3_files")
    if files:
        midi, wav, count = files
        st.audio(wav.read_bytes(), format="audio/wav")
        d1, d2 = st.columns(2)
        d1.download_button("Download MIDI", midi.read_bytes(), midi.name, "audio/midi", key="t3_dm")
        d2.download_button("Download WAV", wav.read_bytes(), wav.name, "audio/wav", key="t3_dw")
        st.caption(f"{count} notes")
