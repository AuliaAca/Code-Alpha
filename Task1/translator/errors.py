"""Exceptions raised by the translator package."""


class TranslationError(Exception):
    """Base error for anything that goes wrong while translating."""


class ProviderUnavailable(TranslationError):
    """A provider cannot be used (missing key, missing library, offline)."""


def describe_request_error(exc: Exception) -> str:
    """Short, human readable cause for a failed HTTP call (no long URLs in the UI)."""
    import requests

    if isinstance(exc, requests.Timeout):
        return "timed out"
    if isinstance(exc, requests.ConnectionError):
        return "no internet connection"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        hint = {401: " (check the API key)", 403: " (key not allowed)", 429: " (rate limit)"}
        return f"HTTP {code}{hint.get(code, '')}"
    if isinstance(exc, ValueError):
        return "unexpected response"
    return type(exc).__name__
