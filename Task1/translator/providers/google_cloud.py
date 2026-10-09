"""Google Cloud Translation API (v2, REST + API key)."""
from __future__ import annotations

import os

import requests

from ..base import TranslationProvider
from ..errors import TranslationError, describe_request_error

ENDPOINT = "https://translation.googleapis.com/language/translate/v2"


class GoogleCloudProvider(TranslationProvider):
    name = "Google Cloud Translation"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0,
                 session: requests.Session | None = None):
        self._key = api_key or os.getenv("GOOGLE_TRANSLATE_API_KEY", "")
        self._timeout = timeout
        self._http = session or requests.Session()

    def is_available(self) -> bool:
        return bool(self._key)

    def translate(self, text: str, source: str, target: str) -> str:
        payload = {"q": text, "target": target, "format": "text", "key": self._key}
        if source != "auto":  # omit `source` to let Google detect it
            payload["source"] = source
        try:
            resp = self._http.post(ENDPOINT, data=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["data"]["translations"][0]["translatedText"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise TranslationError(f"{self.name}: {describe_request_error(exc)}") from exc
