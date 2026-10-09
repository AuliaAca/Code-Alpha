"""Data contracts that connect the tasks.

Tasks never import each other's internals. They exchange these small,
dependency-free dataclasses instead, which keeps coupling low (Dependency
Inversion) and lets any task be swapped or tested in isolation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Detection:
    """One object found in one frame (Task4 detectors -> tracker)."""
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class TrackedObject:
    """A detection that has been given a persistent identity."""
    track_id: int
    label: str
    confidence: float
    box: Box

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class SceneReport:
    """Summary of a video (Task4 -> hub -> Task1 / Task3)."""
    frames: int = 0
    fps: float = 0.0
    # track_id -> label, only confirmed tracks
    tracks: Dict[int, str] = field(default_factory=dict)
    # track_id -> list of centre points over time
    trails: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    # mean speed of all tracks in pixels/frame (drives the music "energy")
    mean_speed: float = 0.0

    @property
    def label_counts(self) -> Dict[str, int]:
        """How many distinct objects of each label were tracked."""
        return dict(Counter(self.tracks.values()))


@dataclass(frozen=True)
class TranslationResult:
    """Output of the Task1 service."""
    text: str
    source: str
    target: str
    provider: str
    cached: bool = False


@dataclass(frozen=True)
class Reply:
    """A routed answer shown to the user by the hub."""
    text: str
    handled_by: str  # "Task1" | "Task2" | "Task3" | "Task4"
    artifact: str | None = None  # path of a produced file, if any
