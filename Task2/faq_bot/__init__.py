"""FAQ chatbot (CodeAlpha AI Task 2)."""
from .bot import FAQBot
from .matcher import FAQMatcher, Match
from .preprocess import TextPreprocessor

__all__ = ["FAQBot", "FAQMatcher", "Match", "TextPreprocessor"]
