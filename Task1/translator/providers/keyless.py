"""Keyless providers: work without an account, so the app is usable out of the box.

* :class:`KeylessGoogleProvider` - the public ``translate.googleapis.com`` endpoint
  used by Google's own web widgets (``client=gtx``). Supports auto-detection.
* :class:`MyMemoryProvider`      - the free MyMemory REST API (needs a source language).

Both are fine for personal use and demos; for production traffic use the official
Google Cloud or Microsoft providers with an API key.
"""
from __future__ import annotations

import requests

from ..base import TranslationProvider
from ..errors import TranslationError, describe_request_error

GTX_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
MYMEMORY_ENDPOINT = "https://api.mymemory.translated.net/get"
_GOOGLE_CODES = {"zh": "zh-CN"}


class _HttpProvider(TranslationProvider):
    """Shared plumbing: a session and a timeout so a bad network never freezes the UI."""

    def __init__(self, timeout: float = 8.0, session: requests.Session | None = None):
        self._timeout = timeout
        self._http = session or requests.Session()

    def is_available(self) -> bool:
        return True  # no key needed; network errors are reported by translate()

    def _get_json(self, url: str, params: dict):
        try:
            resp = self._http.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise TranslationError(f"{self.name}: {describe_request_error(exc)}") from exc


class KeylessGoogleProvider(_HttpProvider):
    name = "Google Translate (keyless)"

    def translate(self, text: str, source: str, target: str) -> str:
        params = {"client": "gtx", "dt": "t", "q": text,
                  "sl": _GOOGLE_CODES.get(source, source),   # "auto" is accepted
                  "tl": _GOOGLE_CODES.get(target, target)}
        data = self._get_json(GTX_ENDPOINT, params)
        try:
            # data[0] is a list of [translated_chunk, original_chunk, ...] per sentence
            return "".join(chunk[0] for chunk in data[0] if chunk and chunk[0])
        except (TypeError, IndexError) as exc:
            raise TranslationError(f"{self.name}: unexpected response") from exc


class MyMemoryProvider(_HttpProvider):
    name = "MyMemory (keyless)"

    def translate(self, text: str, source: str, target: str) -> str:
        if source == "auto":
            raise TranslationError(f"{self.name} needs an explicit source language")
        params = {"q": text, "langpair": f"{source}|{_GOOGLE_CODES.get(target, target)}"}
        data = self._get_json(MYMEMORY_ENDPOINT, params)
        status = int(data.get("responseStatus", 0) or 0)
        translated = (data.get("responseData") or {}).get("translatedText", "")
        # Quota/limit problems come back as HTTP 200 with a warning in the text.
        if status != 200 or not translated or "MYMEMORY WARNING" in translated.upper():
            raise TranslationError(f"{self.name} failed: {data.get('responseDetails') or status}")
        return translated
