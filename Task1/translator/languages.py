"""Supported languages and validation of language codes (single source of truth)."""
from __future__ import annotations

from .errors import TranslationError

AUTO = "auto"

LANGUAGES = {
    "en": "English", "id": "Indonesian", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish",
}


def language_name(code: str) -> str:
    """Human readable name for a code (``auto`` -> ``Detect language``)."""
    return "Detect language" if code == AUTO else LANGUAGES.get(code, code)


def validate_pair(source: str, target: str) -> None:
    """Raise :class:`TranslationError` for unsupported or identical languages."""
    if source != AUTO and source not in LANGUAGES:
        raise TranslationError(f"Unsupported source language: {source!r}")
    if target not in LANGUAGES:
        raise TranslationError(f"Unsupported target language: {target!r}")
    if source == target:
        raise TranslationError("Source and target languages must differ.")
