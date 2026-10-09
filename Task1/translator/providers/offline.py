"""Offline word-level lexicon provider.

It is deliberately simple: it translates word by word from a small bundled
lexicon (``data/offline_lexicon.json``), pivoting through English for pairs
that are not stored directly. It exists so the app, the tests and the hub demo
work without internet or API keys. It is NOT a real machine-translation engine.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..base import TranslationProvider
from ..errors import TranslationError

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "offline_lexicon.json"
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class OfflineLexiconProvider(TranslationProvider):
    name = "Offline lexicon"

    def __init__(self, lexicon_path: Path = LEXICON_PATH):
        # {"en": {"hello": {"id": "halo", "es": "hola", ...}, ...}}
        data = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self._to_lang: dict[str, dict[str, str]] = data["entries"]  # en word -> {lang: word}
        self._to_english: dict[str, dict[str, str]] = {}            # lang -> {word: en word}
        for en_word, forms in self._to_lang.items():
            for lang, word in forms.items():
                # setdefault: when two English words share a translation
                # ("hello"/"hi" -> "hola") the first, preferred entry wins.
                self._to_english.setdefault(lang, {}).setdefault(word.lower(), en_word)

    def is_available(self) -> bool:
        return True

    @property
    def languages(self) -> set[str]:
        return {"en", *self._to_english}

    def translate(self, text: str, source: str, target: str) -> str:
        if source == "auto":
            source = self._guess_source(text)
        missing = {source, target} - self.languages
        if missing:  # better a clear error than silently returning the input
            raise TranslationError(
                f"Offline fallback only covers {', '.join(sorted(self.languages))}.")
        out = [self._convert(tok, source, target) for tok in _TOKEN.findall(text)]
        result = " ".join(out)
        result = re.sub(r"\s+([.,!?;:])", r"\1", result)  # tidy punctuation spacing
        if not result.strip():
            raise TranslationError("Nothing to translate.")
        return result

    # -- helpers ---------------------------------------------------------
    def _convert(self, token: str, source: str, target: str) -> str:
        if not token[0].isalnum():
            return token
        key = token.lower()
        english = key if source == "en" else self._to_english.get(source, {}).get(key)
        if english is None:
            return token  # unknown word: leave untouched
        word = english if target == "en" else self._to_lang.get(english, {}).get(target, token)
        return word.capitalize() if token[0].isupper() else word

    def _guess_source(self, text: str) -> str:
        """Pick the language whose lexicon covers most of the words."""
        words = [w.lower() for w in _TOKEN.findall(text) if w[0].isalnum()]
        scores = {"en": sum(w in self._to_lang for w in words)}
        for lang, table in self._to_english.items():
            scores[lang] = sum(w in table for w in words)
        return max(scores, key=scores.get)
