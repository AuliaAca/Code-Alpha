"""Text cleaning for FAQ matching: lowercase -> tokenize -> drop stop words -> stem.

NLTK's tokenizer/stemmer classes are used but none of them needs a corpus
download, so the bot works offline and in CI. A compact stop-word list is
bundled for the same reason.
"""
from __future__ import annotations

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Question words and request verbs ("explain", "tell") carry no topic signal, so
# they are removed. Words like "not" are kept because they flip meaning.
STOP_WORDS = frozenset("""
a an the and or but if of to in on at by for with from into as is am are was were be been
being do does did doing have has had i me my we our you your it its this that these those
they them their he she his her can could should would will just so than then there here
what which who whom when where why how about any some all also very too up out over
explain tell describe show give please mean means
""".split())


class TextPreprocessor:
    """Turns raw text into a normalised string of stemmed tokens."""

    def __init__(self, stop_words: frozenset[str] = STOP_WORDS):
        self._tokenizer = RegexpTokenizer(r"[a-z0-9]+")
        self._stemmer = PorterStemmer()
        self._stop_words = stop_words

    def tokens(self, text: str) -> list[str]:
        words = self._tokenizer.tokenize(text.lower())
        return [self._stemmer.stem(w) for w in words if w not in self._stop_words]

    def __call__(self, text: str) -> str:
        return " ".join(self.tokens(text))
