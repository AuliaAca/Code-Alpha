"""Tiny LRU cache so repeated translations cost nothing (and save API quota)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Optional

from common.contracts import TranslationResult


class LRUCache:
    def __init__(self, capacity: int = 256):
        self._capacity = capacity
        self._items: "OrderedDict[Hashable, TranslationResult]" = OrderedDict()

    def get(self, key: Hashable) -> Optional[TranslationResult]:
        if key not in self._items:
            return None
        self._items.move_to_end(key)  # mark as most recently used
        return self._items[key]

    def put(self, key: Hashable, value: TranslationResult) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)  # evict least recently used

    def __len__(self) -> int:
        return len(self._items)
