"""Conversation logic on top of the matcher: greetings, thresholds, fallbacks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .matcher import FAQMatcher, Match

# Score bands (tuned on the bundled FAQs, see tests):
CONFIDENT = 0.35   # answer directly
UNSURE = 0.15      # "did you mean ...?"

_GREETING = re.compile(r"^\s*(hi|hello|hey|halo|good (morning|afternoon|evening))\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|terima kasih)\b", re.I)
_BYE = re.compile(r"^\s*(bye|goodbye|see you|quit|exit)\b", re.I)


@dataclass
class BotReply:
    text: str
    confidence: float = 0.0
    matches: List[Match] = field(default_factory=list)


class FAQBot:
    def __init__(self, matcher: FAQMatcher | None = None):
        self._matcher = matcher or FAQMatcher()

    def reply(self, message: str) -> BotReply:
        if _BYE.match(message):
            return BotReply("Goodbye! 👋")
        if _GREETING.match(message) and len(message.split()) <= 4:
            return BotReply("Hello! Ask me anything about the CodeLab toolkit.")
        if _THANKS.search(message) and len(message.split()) <= 5:
            return BotReply("You're welcome!")

        matches = self._matcher.top(message, k=3)
        if not matches or matches[0].score < UNSURE:
            return BotReply("Sorry, I don't know that one. Try asking about the translator, "
                            "chatbot, music generator or object tracker.", 0.0, matches)
        best = matches[0]
        if best.score >= CONFIDENT:
            return BotReply(best.answer, best.score, matches)
        # Middle band: don't guess, offer the closest questions instead.
        options = "\n".join(f"• {m.question}" for m in matches if m.score >= UNSURE)
        return BotReply(f"I'm not sure I understood. Did you mean:\n{options}", best.score, matches)
