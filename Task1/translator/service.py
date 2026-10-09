"""Orchestrates providers: validation -> cache -> first provider that works."""
from __future__ import annotations

import logging
from typing import Sequence

from common.contracts import TranslationResult

from .base import TranslationProvider
from .cache import LRUCache
from .errors import TranslationError
from .languages import validate_pair
from .providers import (GoogleCloudProvider, KeylessGoogleProvider,
                        MicrosoftProvider, MyMemoryProvider, OfflineLexiconProvider)

log = logging.getLogger(__name__)
MAX_CHARS = 5000  # protects API quota and the UI


class TranslationService:
    """Chain of Responsibility over providers, in priority order.

    The service only depends on the :class:`TranslationProvider` abstraction,
    so it never needs to change when a new backend is added.
    """

    def __init__(self, providers: Sequence[TranslationProvider],
                 cache: LRUCache | None = None):
        if not providers:
            raise ValueError("At least one provider is required.")
        self._providers = list(providers)
        self._cache = cache or LRUCache()

    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        text = text.strip()
        if not text:
            raise TranslationError("Please enter some text to translate.")
        if len(text) > MAX_CHARS:
            raise TranslationError(f"Text is too long (max {MAX_CHARS} characters).")
        validate_pair(source, target)

        key = (text, source, target)
        hit = self._cache.get(key)
        if hit:
            return TranslationResult(hit.text, source, target, hit.provider, cached=True)

        errors: list[str] = []
        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                translated = provider.translate(text, source, target)
            except TranslationError as exc:  # try the next provider
                log.info("%s", exc)  # quiet by default; full reasons are in the final error
                errors.append(str(exc))
                continue
            result = TranslationResult(translated, source, target, provider.name)
            if not isinstance(provider, OfflineLexiconProvider):
                # don't cache the word-by-word fallback: retry online next time
                self._cache.put(key, result)
            return result
        detail = "; ".join(errors) or "no provider is configured"
        raise TranslationError(f"Translation failed. {detail}")

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers if p.is_available()]


def build_default_service(offline_only: bool = False) -> TranslationService:
    """Priority: Google Cloud -> Microsoft -> keyless Google -> MyMemory -> offline lexicon.

    Keyed providers are skipped automatically when their key is not set.
    """
    if offline_only:
        return TranslationService([OfflineLexiconProvider()])
    return TranslationService([
        GoogleCloudProvider(), MicrosoftProvider(),
        KeylessGoogleProvider(), MyMemoryProvider(), OfflineLexiconProvider(),
    ])


def is_offline_result(result: TranslationResult) -> bool:
    """True when only the word-by-word fallback could answer (UI shows a warning)."""
    return result.provider == OfflineLexiconProvider.name
