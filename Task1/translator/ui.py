"""Streamlit view for the translator, shared by Task1/app.py and the root dashboard."""
from __future__ import annotations

import streamlit as st

from .errors import TranslationError
from .languages import AUTO, LANGUAGES, language_name
from .service import MAX_CHARS, build_default_service, is_offline_result
from .tts import speak


@st.cache_resource
def _service():
    return build_default_service()


def render_translator() -> None:
    st.title("🌐 Language Translation Tool")
    service = _service()
    st.caption("Provider chain: " + " → ".join(service.provider_names))

    codes = list(LANGUAGES)
    left, right = st.columns(2)
    source = left.selectbox("From", [AUTO] + codes, format_func=language_name, key="t1_src")
    target = right.selectbox("To", codes, index=codes.index("id"), format_func=language_name, key="t1_tgt")
    text = st.text_area("Text", height=150, max_chars=MAX_CHARS, key="t1_text",
                        placeholder="Type or paste text…")

    if st.button("Translate", type="primary", key="t1_go"):
        try:
            with st.spinner("Translating…"):
                st.session_state["t1_result"] = service.translate(text, source, target)
        except TranslationError as exc:
            st.session_state.pop("t1_result", None)
            st.error(str(exc))

    result = st.session_state.get("t1_result")
    if not result:
        return
    st.subheader(language_name(result.target))
    # st.code has a built-in copy-to-clipboard icon in its top-right corner
    st.code(result.text, language=None, wrap_lines=True)
    st.caption(f"via {result.provider}" + (" (cached)" if result.cached else ""))
    if is_offline_result(result):
        st.warning("No online translator could be reached, so this is a word-by-word "
                   "fallback. Check your internet connection or add an API key in .env.")
    if st.button("🔊 Listen", key="t1_tts"):
        try:
            st.audio(speak(result.text, result.target), format="audio/mp3")
        except TranslationError as exc:
            st.warning(str(exc))
