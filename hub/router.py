"""One chat box for everything: routes a message to the right task.

    "translate good morning to spanish"  -> Task1
    "compose a calm melody"              -> Task3
    "describe the demo scene in french"  -> Task4 (+ Task1)
    anything else                        -> Task2 FAQ bot
"""
from __future__ import annotations

import re
from typing import Optional

from common.contracts import Reply
from common.paths import add_task_paths

add_task_paths()
from translator import TranslationError  # noqa: E402
from translator.languages import LANGUAGES  # noqa: E402

from .pipeline import CodeLabHub, describe_scene  # noqa: E402

_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGES.items()}
_TRANSLATE = re.compile(r"^\s*translate\s+(?P<text>.+?)\s+(?:in)?to\s+(?P<lang>[a-z]+)\s*$", re.I)
_MUSIC = re.compile(r"\b(compose|generate|make|create|play)\b.*\b(music|melody|song|tune)\b", re.I)
_SCENE = re.compile(r"\b(describe|analy[sz]e|track)\b.*\b(scene|video)\b", re.I)
_IN_LANG = re.compile(r"\bin\s+([a-z]+)\s*$", re.I)


def _language_code(word: str) -> Optional[str]:
    word = word.lower()
    return word if word in LANGUAGES else _NAME_TO_CODE.get(word)


class Router:
    def __init__(self, hub: Optional[CodeLabHub] = None):
        self.hub = hub or CodeLabHub()

    def handle(self, message: str) -> Reply:
        if (m := _TRANSLATE.match(message)):
            return self._translate(m["text"].strip("\"' "), m["lang"])
        if _SCENE.search(message):
            return self._scene(message)
        if _MUSIC.search(message):
            return self._music()
        return Reply(self.hub.bot.reply(message).text, "Task2")

    # -- handlers -------------------------------------------------------------
    def _translate(self, text: str, lang_word: str) -> Reply:
        code = _language_code(lang_word)
        if not code:
            return Reply(f"I don't know the language {lang_word!r}. Try: {', '.join(_NAME_TO_CODE)}.", "Task1")
        try:
            result = self.hub.translator.translate(text, "auto", code)
        except TranslationError as exc:
            return Reply(f"Translation failed: {exc}", "Task1")
        return Reply(f"{result.text}  [{result.provider}]", "Task1")

    def _scene(self, message: str) -> Reply:
        """Track the demo video; narrate in the requested language (default: English)."""
        match = _IN_LANG.search(message)
        code = _language_code(match.group(1)) if match else None
        report = self.hub.track_scene()
        if code in (None, "en"):
            return Reply(describe_scene(report), "Task4")
        try:
            english, translated = self.hub.narrate(report, code)
        except TranslationError as exc:
            return Reply(f"{describe_scene(report)}\n(translation failed: {exc})", "Task4")
        return Reply(f"{english}\n{translated.text}  [{translated.provider}]", "Task4")

    def _music(self) -> Reply:
        report = self.hub.track_scene()
        _, midi, wav = self.hub.sonify(report)
        return Reply(f"Composed a melody from the demo scene ({', '.join(report.tracks.values())}).",
                     "Task3", artifact=str(wav))
