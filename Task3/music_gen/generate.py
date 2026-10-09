"""Sampling new music from a trained checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .data import UNKNOWN, Vocabulary, make_token
from .model import MusicLSTM

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "bach_lstm.pt"


class MusicGenerator:
    def __init__(self, model: MusicLSTM, vocab: Vocabulary, seq_len: int):
        self.model, self.vocab, self.seq_len = model.eval(), vocab, seq_len

    @classmethod
    def load(cls, path: Path = DEFAULT_CHECKPOINT) -> "MusicGenerator":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg, vocab = ckpt["config"], Vocabulary.from_json(ckpt["vocab"])
        model = MusicLSTM(len(vocab), cfg["embed_dim"], cfg["hidden"], cfg["layers"], cfg["dropout"])
        model.load_state_dict(ckpt["state"])
        return cls(model, vocab, cfg["seq_len"])

    def seed_from_pitches(self, pitches: Sequence[int]) -> List[str]:
        """Build a seed phrase from single MIDI notes (used by the hub)."""
        return [make_token([p], 1.0) for p in pitches]

    @torch.no_grad()
    def generate(self, length: int = 64, temperature: float = 0.9, top_k: int = 12,
                 seed_tokens: Optional[Sequence[str]] = None, rng_seed: Optional[int] = None) -> List[str]:
        """Return ``length`` new tokens.

        temperature < 1 -> conservative, > 1 -> adventurous.
        top_k limits sampling to the k likeliest tokens (avoids nonsense picks).
        """
        gen = torch.Generator().manual_seed(rng_seed) if rng_seed is not None else None
        seed_ids = self.vocab.encode(seed_tokens or [])
        # Pad/trim the seed to exactly seq_len by repeating it (or using the most
        # common non-unknown token when no seed is given).
        if not seed_ids:
            seed_ids = [1]
        window = (seed_ids * self.seq_len)[-self.seq_len:]
        out: List[int] = []
        for _ in range(length):
            logits = self.model(torch.tensor([window]))[0] / max(temperature, 1e-3)
            logits[self.vocab.stoi[UNKNOWN]] = float("-inf")  # never emit <unk>
            values, indices = torch.topk(logits, min(top_k, len(logits)))
            probs = torch.softmax(values, dim=0)
            choice = indices[torch.multinomial(probs, 1, generator=gen)].item()
            out.append(choice)
            window = window[1:] + [choice]  # slide the context window
        return self.vocab.decode(out)
