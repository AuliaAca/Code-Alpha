import pytest

from faq_bot import FAQBot, FAQMatcher, TextPreprocessor


def test_preprocess_lowercases_stems_and_drops_stopwords():
    assert TextPreprocessor().tokens("How does the Tracking WORK?") == ["track", "work"]


@pytest.fixture(scope="module")
def matcher():
    return FAQMatcher()


@pytest.mark.parametrize("query,expected", [
    ("which languages does the translator support", "Which languages can the translator handle?"),
    ("how can i use my webcam", "Can I use my webcam?"),
    ("what does the temperature setting do", "What does temperature mean for generation?"),
    ("how do I install it", "How do I install the project?"),
    ("explain SORT tracker", "What is SORT?"),
])
def test_paraphrases_hit_the_right_faq(matcher, query, expected):
    assert matcher.top(query, 1)[0].question == expected


def test_empty_or_stopword_only_query_returns_nothing(matcher):
    assert matcher.top("the of and", 3) == []


def test_bot_small_talk_and_fallback():
    bot = FAQBot()
    assert "Hello" in bot.reply("hi there").text
    assert "don't know" in bot.reply("what is the capital of france").text


def test_bot_answers_confidently():
    reply = FAQBot().reply("How do I run the full demo?")
    assert "python main.py demo" in reply.text and reply.confidence >= 0.35
