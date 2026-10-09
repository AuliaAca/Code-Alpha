"""Streamlit chat view for the FAQ bot (used by Task2/app.py and the dashboard)."""
from __future__ import annotations

import streamlit as st

from .bot import FAQBot

GREETING = "Hi! Ask me anything about the CodeLab toolkit."


@st.cache_resource
def _bot() -> FAQBot:
    return FAQBot()


def render_faq_bot() -> None:
    st.title("💬 FAQ Chatbot")
    st.caption("TF-IDF + cosine similarity over Task2/data/faqs.csv")
    history = st.session_state.setdefault("t2_history", [("assistant", GREETING)])

    for role, text in history:
        st.chat_message(role).write(text)

    if prompt := st.chat_input("Ask a question…", key="t2_input"):
        reply = _bot().reply(prompt)
        history.append(("user", prompt))
        history.append(("assistant", reply.text))
        st.rerun()

    if len(history) > 1 and st.button("Clear chat", key="t2_clear"):
        st.session_state["t2_history"] = [("assistant", GREETING)]
        st.rerun()
