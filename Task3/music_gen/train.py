"""Training loop. Saves a self-contained checkpoint (weights + vocab + config)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Vocabulary, make_windows
from .model import MusicLSTM


@dataclass
class TrainConfig:
    seq_len: int = 24
    embed_dim: int = 64
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    epochs: int = 12
    batch_size: int = 128
    lr: float = 2e-3
    min_count: int = 2
    val_fraction: float = 0.1  # share of *pieces* held out (no window leakage)
    seed: int = 7


def _loader(pieces_ids: List[List[int]], cfg: TrainConfig, shuffle: bool) -> DataLoader:
    x, y = make_windows(pieces_ids, cfg.seq_len)
    data = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(data, batch_size=cfg.batch_size, shuffle=shuffle)


def _run_epoch(model, loader, loss_fn, optimizer=None) -> tuple[float, float]:
    """One pass over ``loader``. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilises LSTMs
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            count += len(yb)
    return total_loss / count, correct / count


def train(pieces: List[List[str]], cfg: TrainConfig, out_path: Path,
          progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Train on token pieces; returns the history dict that is also saved."""
    torch.manual_seed(cfg.seed)
    vocab = Vocabulary.build(pieces, cfg.min_count)
    ids = [vocab.encode(p) for p in pieces]
    n_val = max(1, int(len(ids) * cfg.val_fraction))
    train_loader = _loader(ids[:-n_val], cfg, shuffle=True)
    val_loader = _loader(ids[-n_val:], cfg, shuffle=False)

    model = MusicLSTM(len(vocab), cfg.embed_dim, cfg.hidden, cfg.layers, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": 0, "seconds": 0.0}
    best_val, best_state = float("inf"), None

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, _ = _run_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_loss < best_val:  # early-stopping style: keep the best weights, not the last
            best_val, history["best_epoch"] = va_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if progress:
            progress({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_acc": va_acc})
    history["seconds"] = time.time() - start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "vocab": vocab.to_json(), "config": asdict(cfg)}, out_path)
    out_path.with_suffix(".history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return history
