"""TF-IDF + cosine similarity retrieval over a FAQ table."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import TextPreprocessor

QUESTION_WEIGHT = 0.75  # share of the score that comes from the question text
DEFAULT_FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.csv"


@dataclass(frozen=True)
class Match:
    question: str
    answer: str
    category: str
    score: float  # cosine similarity in [0, 1]


def load_faqs(path: Path = DEFAULT_FAQ_PATH) -> list[dict]:
    """Read the FAQ CSV (columns: category, question, answer)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No FAQs found in {path}")
    return rows


class FAQMatcher:
    def __init__(self, faqs: list[dict] | None = None,
                 preprocessor: TextPreprocessor | None = None):
        self._faqs = faqs or load_faqs()
        self._prep = preprocessor or TextPreprocessor()
        # One shared vocabulary, but questions and answers are scored separately:
        # users echo the *question* wording, so it gets the larger weight, while
        # the answer text still rescues indirectly phrased queries.
        questions = [self._prep(f["question"]) for f in self._faqs]
        answers = [self._prep(f["answer"]) for f in self._faqs]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._vectorizer.fit(questions + answers)
        self._q_matrix = self._vectorizer.transform(questions)
        self._a_matrix = self._vectorizer.transform(answers)

    def _scores(self, cleaned: list[str]) -> np.ndarray:
        vectors = self._vectorizer.transform(cleaned)
        return (QUESTION_WEIGHT * cosine_similarity(vectors, self._q_matrix)
                + (1 - QUESTION_WEIGHT) * cosine_similarity(vectors, self._a_matrix))

    def top(self, query: str, k: int = 3) -> List[Match]:
        """Return the ``k`` most similar FAQs, best first."""
        cleaned = self._prep(query)
        if not cleaned:
            return []
        scores = self._scores([cleaned]).ravel()
        best = np.argsort(scores)[::-1][:k]
        return [Match(self._faqs[i]["question"], self._faqs[i]["answer"],
                      self._faqs[i]["category"], float(scores[i])) for i in best]

    def similarity_matrix(self, queries: list[str]) -> np.ndarray:
        """Scores of every query against every FAQ (used by the visualisation)."""
        return self._scores([self._prep(q) for q in queries])

    @property
    def faqs(self) -> list[dict]:
        return self._faqs
