"""Turn MIDI/score data into token sequences an LSTM can learn.

Token format:  ``"64@1.0"`` = MIDI pitch 64 held for 1.0 quarter notes
(``"60.64.67@1.0"`` would be a chord). A rest is ``"rest@1.0"``. Encoding pitch and duration in one
token keeps the model a plain next-token predictor (KISS).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

DURATIONS = (0.5, 1.0, 2.0, 4.0)  # quarter-note lengths we allow
REST = "rest"
UNKNOWN = "<unk>"


def quantize_duration(quarter_length: float) -> float:
    """Snap a duration to the nearest allowed value (shrinks the vocabulary)."""
    return min(DURATIONS, key=lambda d: abs(d - float(quarter_length)))


def make_token(pitches: Sequence[int], duration: float) -> str:
    body = REST if not pitches else ".".join(str(p) for p in sorted(set(pitches)))
    return f"{body}@{quantize_duration(duration)}"


def parse_token(token: str) -> tuple[list[int], float]:
    """Inverse of :func:`make_token` -> (pitches, duration)."""
    body, dur = token.split("@")
    pitches = [] if body == REST else [int(p) for p in body.split(".")]
    return pitches, float(dur)


def transpose_to_c(score):
    """Transpose to C major / A minor so equal harmonies share equal tokens.

    Without this the same chord in 12 keys becomes 12 rare tokens and the model
    sees mostly ``<unk>``.
    """
    from music21 import interval, pitch

    key = score.analyze("key")
    target = pitch.Pitch("C" if key.mode == "major" else "A")
    return score.transpose(interval.Interval(key.tonic, target))


def pick_melody(score):
    """The part named "Soprano", else the part with the highest average pitch.

    ``parts[0]`` is not reliable: some scores list instruments before the voices.
    """
    parts = list(getattr(score, "parts", []))
    if not parts:
        return score
    for part in parts:
        if "soprano" in (part.partName or "").lower():
            return part

    def mean_pitch(part) -> float:
        pitches = [p.midi for n in part.flatten().notes for p in n.pitches]
        return sum(pitches) / len(pitches) if pitches else 0.0

    return max(parts, key=mean_pitch)


def score_to_tokens(score) -> List[str]:
    """Melody tokens from the top voice (soprano) of a music21 score.

    Modelling one voice keeps the vocabulary small (~100 tokens) so an LSTM can
    learn it from a few hundred pieces. Chords in the source are kept as chords.
    """
    score = transpose_to_c(score)
    tokens = []
    for element in pick_melody(score).flatten().notesAndRests:
        pitches = [] if element.isRest else [p.midi for p in element.pitches]
        tokens.append(make_token(pitches, element.duration.quarterLength))
    return tokens


def load_bach_corpus(max_pieces: int = 150) -> List[List[str]]:
    """Bach chorales bundled with music21 (no download needed)."""
    from music21 import corpus  # heavy import, keep it lazy

    pieces = []
    for path in list(corpus.getComposer("bach"))[:max_pieces]:
        try:
            pieces.append(score_to_tokens(corpus.parse(path)))
        except Exception:  # a few corpus files are not plain chorales; skip them
            continue
    return [p for p in pieces if len(p) > 16]


def load_midi_folder(folder: Path) -> List[List[str]]:
    """Use your own .mid files instead of the Bach corpus."""
    from music21 import converter

    files = sorted(Path(folder).glob("*.mid")) + sorted(Path(folder).glob("*.midi"))
    return [score_to_tokens(converter.parse(str(f))) for f in files]


class Vocabulary:
    """Bidirectional token <-> index map. Rare tokens collapse into ``<unk>``."""

    def __init__(self, tokens: Iterable[str]):
        self.itos: List[str] = [UNKNOWN] + sorted(set(tokens) - {UNKNOWN})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, pieces: Iterable[Sequence[str]], min_count: int = 2) -> "Vocabulary":
        counts = Counter(t for piece in pieces for t in piece)
        return cls(t for t, c in counts.items() if c >= min_count)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def __len__(self) -> int:
        return len(self.itos)

    def to_json(self) -> list[str]:
        return self.itos

    @classmethod
    def from_json(cls, itos: list[str]) -> "Vocabulary":
        vocab = cls([])
        vocab.itos = list(itos)
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        return vocab


def make_windows(pieces: Iterable[Sequence[int]], seq_len: int):
    """Sliding windows: input = seq_len tokens, target = the next token."""
    inputs, targets = [], []
    for piece in pieces:
        for i in range(len(piece) - seq_len):
            inputs.append(piece[i:i + seq_len])
            targets.append(piece[i + seq_len])
    return inputs, targets


def cache_corpus(pieces: List[List[str]], path: Path) -> None:
    path.write_text(json.dumps(pieces), encoding="utf-8")


def read_cached_corpus(path: Path) -> List[List[str]]:
    return json.loads(path.read_text(encoding="utf-8"))
