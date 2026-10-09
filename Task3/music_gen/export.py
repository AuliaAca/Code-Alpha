"""Write generated tokens to MIDI (music21) and to a WAV file (NumPy synth).

The WAV synth is a small additive synthesiser so audio works with no system
dependencies (no FluidSynth / sound-fonts needed).
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import parse_token

SAMPLE_RATE = 22050


def tokens_to_midi(tokens: Sequence[str], path: Path, bpm: int = 84) -> Path:
    from music21 import chord, note, stream, tempo

    part = stream.Part()
    part.append(tempo.MetronomeMark(number=bpm))
    for token in tokens:
        pitches, duration = parse_token(token)
        if not pitches:
            element = note.Rest()
        elif len(pitches) == 1:
            element = note.Note(pitches[0])
        else:
            element = chord.Chord(pitches)
        element.quarterLength = duration
        part.append(element)
    path.parent.mkdir(parents=True, exist_ok=True)
    part.write("midi", fp=str(path))
    return path


def _tone(freq: float, seconds: float) -> np.ndarray:
    """A soft organ-like tone: 3 harmonics with attack/release to avoid clicks."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    wave_ = sum(amp * np.sin(2 * np.pi * freq * k * t) for k, amp in ((1, 1.0), (2, 0.35), (3, 0.15)))
    envelope = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (seconds - t) / 0.08).clip(0)
    return wave_ * envelope


def tokens_to_wav(tokens: Sequence[str], path: Path, bpm: int = 84) -> Path:
    seconds_per_beat = 60.0 / bpm
    pieces = []
    for token in tokens:
        pitches, duration = parse_token(token)
        seconds = duration * seconds_per_beat
        if not pitches:
            pieces.append(np.zeros(int(SAMPLE_RATE * seconds)))
            continue
        freqs = [440.0 * 2 ** ((p - 69) / 12) for p in pitches]  # MIDI -> Hz
        pieces.append(sum(_tone(f, seconds) for f in freqs) / len(freqs) ** 0.7)
    audio = np.concatenate(pieces) if pieces else np.zeros(1)
    audio = 0.8 * audio / max(np.abs(audio).max(), 1e-9)  # normalise, leave headroom
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    return path
