"""Path helpers so tasks can import each other without installing packages.

Each TaskN folder is a self-contained project that owns one Python package
(``translator``, ``faq_bot``, ``music_gen``, ``tracking``). The hub needs all
of them on ``sys.path``; this module is the single place that knows how.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIRS = [REPO_ROOT / f"Task{i}" for i in range(1, 5)]


def add_task_paths() -> None:
    """Prepend the repo root and every task folder to ``sys.path`` (idempotent)."""
    for path in [REPO_ROOT, *TASK_DIRS]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
