"""Microsoft Translator (Azure AI Translator, v3.0 REST)."""
from __future__ import annotations

import os

import requests

from ..base import TranslationProvider
from ..errors import TranslationError, describe_request_error

ENDPOINT = "https://api.cognitive.microsofttranslator.com/translate"

# Microsoft uses a few different codes than ISO-639-1.
_CODE_MAP = {"zh": "zh-Hans"}


class MicrosoftProvider(TranslationProvider):
    name = "Microsoft Translator"

    def __init__(self, api_key: str | None = None, region: str | None = None,
                 timeout: float = 10.0, session: requests.Session | None = None):
        self._key = api_key or os.getenv("MICROSOFT_TRANSLATOR_KEY", "")
        self._region = region or os.getenv("MICROSOFT_TRANSLATOR_REGION", "global")
        self._timeout = timeout
        self._http = session or requests.Session()

    def is_available(self) -> bool:
        return bool(self._key)

    def translate(self, text: str, source: str, target: str) -> str:
        params = {"api-version": "3.0", "to": _CODE_MAP.get(target, target)}
        if source != "auto":
            params["from"] = _CODE_MAP.get(source, source)
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Ocp-Apim-Subscription-Region": self._region,
            "Content-Type": "application/json",
        }
        try:
            resp = self._http.post(ENDPOINT, params=params, headers=headers,
                                   json=[{"Text": text}], timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()[0]["translations"][0]["text"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise TranslationError(f"{self.name}: {describe_request_error(exc)}") from exc
