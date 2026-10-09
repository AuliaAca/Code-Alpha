"""The provider contract. Every backend implements this one small interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """Strategy interface for a translation backend (Open/Closed principle:
    new backends are added by subclassing, never by editing the service)."""

    #: short display name, shown next to results
    name: str = "provider"

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check (no network) that this provider can be attempted."""

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Translate ``text``. ``source`` may be ``"auto"``.

        Raises :class:`~translator.errors.TranslationError` on failure.
        """
