import pytest

from common.contracts import TranslationResult
from translator import TranslationError, TranslationService
from translator.base import TranslationProvider
from translator.providers import (GoogleCloudProvider, MicrosoftProvider,
                                  OfflineLexiconProvider)


class Fake(TranslationProvider):
    def __init__(self, name, available=True, fail=False):
        self.name, self._available, self._fail, self.calls = name, available, fail, 0

    def is_available(self):
        return self._available

    def translate(self, text, source, target):
        self.calls += 1
        if self._fail:
            raise TranslationError("boom")
        return f"{self.name}:{text}"


def test_falls_back_to_next_provider():
    a, b = Fake("a", fail=True), Fake("b")
    result = TranslationService([a, b]).translate("hi", "en", "id")
    assert result.provider == "b" and result.text == "b:hi"


def test_skips_unavailable_provider():
    a, b = Fake("a", available=False), Fake("b")
    TranslationService([a, b]).translate("hi", "en", "id")
    assert a.calls == 0


def test_cache_avoids_second_call():
    a = Fake("a")
    svc = TranslationService([a])
    svc.translate("hi", "en", "id")
    again = svc.translate("hi", "en", "id")
    assert a.calls == 1 and again.cached


@pytest.mark.parametrize("text,src,tgt", [("", "en", "id"), ("hi", "en", "en"),
                                          ("hi", "xx", "id"), ("hi", "en", "zz")])
def test_validation_errors(text, src, tgt):
    with pytest.raises(TranslationError):
        TranslationService([Fake("a")]).translate(text, src, tgt)


def test_all_providers_failing_raises():
    with pytest.raises(TranslationError, match="Translation failed"):
        TranslationService([Fake("a", fail=True)]).translate("hi", "en", "id")


def test_offline_lexicon_basic_and_pivot():
    p = OfflineLexiconProvider()
    assert p.translate("Hello world", "en", "id") == "Halo dunia"
    assert p.translate("halo dunia", "id", "es") == "hola mundo"  # pivot via English
    assert p.translate("hola", "auto", "en") == "hello"           # auto-detect


class FakeSession:
    def __init__(self, payload):
        self.payload, self.kwargs = payload, None

    def post(self, url, **kwargs):
        self.kwargs = kwargs

        class Resp:
            def raise_for_status(_): pass
            def json(_): return self.payload
        return Resp()


def test_google_cloud_request_and_parse():
    s = FakeSession({"data": {"translations": [{"translatedText": "halo"}]}})
    p = GoogleCloudProvider(api_key="K", session=s)
    assert p.translate("hello", "en", "id") == "halo"
    assert s.kwargs["data"]["source"] == "en" and s.kwargs["data"]["key"] == "K"


def test_microsoft_request_and_parse():
    s = FakeSession([{"translations": [{"text": "halo"}]}])
    p = MicrosoftProvider(api_key="K", region="r", session=s)
    assert p.translate("hello", "auto", "id") == "halo"
    assert "from" not in s.kwargs["params"]  # auto-detect omits `from`


def test_provider_without_key_is_unavailable():
    assert not GoogleCloudProvider(api_key="").is_available()


# --- keyless providers (fake HTTP, no network) ---------------------------------
from translator.providers import KeylessGoogleProvider, MyMemoryProvider  # noqa: E402
from translator.service import build_default_service, is_offline_result  # noqa: E402


class FakeGetSession:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.params = payload, error, None

    def get(self, url, params=None, timeout=None):
        assert timeout is not None  # every request must have a timeout
        self.params = params
        if self.error:
            raise self.error
        payload = self.payload

        class Resp:
            def raise_for_status(_): pass
            def json(_): return payload
        return Resp()


def test_keyless_google_joins_sentence_chunks():
    s = FakeGetSession([[["Halo dunia. ", "Hello world. ", None], ["Apa kabar?", "How are you?", None]],
                        None, "en"])
    out = KeylessGoogleProvider(session=s).translate("Hello world. How are you?", "auto", "id")
    assert out == "Halo dunia. Apa kabar?" and s.params["sl"] == "auto"


def test_keyless_network_error_becomes_translation_error():
    import requests
    p = KeylessGoogleProvider(session=FakeGetSession(error=requests.ConnectionError("down")))
    with pytest.raises(TranslationError):
        p.translate("hi", "en", "id")


def test_mymemory_parses_and_rejects_quota_warning():
    ok = FakeGetSession({"responseStatus": 200, "responseData": {"translatedText": "Halo"}})
    assert MyMemoryProvider(session=ok).translate("Hello", "en", "id") == "Halo"
    assert ok.params["langpair"] == "en|id"
    warn = FakeGetSession({"responseStatus": 200,
                           "responseData": {"translatedText": "MYMEMORY WARNING: YOU USED ALL"}})
    with pytest.raises(TranslationError):
        MyMemoryProvider(session=warn).translate("Hello", "en", "id")
    with pytest.raises(TranslationError):
        MyMemoryProvider(session=ok).translate("Hello", "auto", "id")


def test_offline_fallback_is_flagged_and_not_cached():
    svc = build_default_service(offline_only=True)
    first = svc.translate("hello", "en", "id")
    assert is_offline_result(first)
    assert not svc.translate("hello", "en", "id").cached


def test_offline_refuses_languages_it_does_not_have():
    with pytest.raises(TranslationError, match="Offline fallback only covers"):
        OfflineLexiconProvider().translate("hello", "en", "ja")
