"""Optional text-to-speech using gTTS (needs internet). Returns MP3 bytes."""
from __future__ import annotations

import io

from .errors import ProviderUnavailable, TranslationError

_GTTS_CODES = {"zh": "zh-CN"}


def speak(text: str, lang: str) -> bytes:
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise ProviderUnavailable("Install gTTS to enable speech.") from exc
    try:
        buffer = io.BytesIO()
        gTTS(text=text, lang=_GTTS_CODES.get(lang, lang)).write_to_fp(buffer)
        return buffer.getvalue()
    except Exception as exc:  # network / unsupported language
        raise TranslationError(f"Text-to-speech failed: {exc}") from exc
