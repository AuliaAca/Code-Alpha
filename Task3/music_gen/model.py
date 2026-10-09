"""Next-token LSTM: embedding -> stacked LSTM -> vocabulary logits."""
from __future__ import annotations

import torch
from torch import nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq) token ids -> (batch, vocab) logits for the next token."""
        out, _ = self.lstm(self.embed(x))
        return self.head(self.drop(out[:, -1]))  # only the last step predicts
