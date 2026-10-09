"""Scene -> words -> music.

    Task4 (track video) --SceneReport--> Task1 (narrate in any language)
                                     \\-> Task3 (compose a melody from the scene)

The hub only talks to the tasks through their public entry points and the
dataclasses in ``common.contracts`` (Dependency Inversion): each task can be
replaced or tested alone.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import List, Optional, Tuple

from common.contracts import SceneReport, TranslationResult
from common.paths import REPO_ROOT, add_task_paths

add_task_paths()
from faq_bot import FAQBot  # noqa: E402
from music_gen.export import tokens_to_midi, tokens_to_wav  # noqa: E402
from music_gen.generate import DEFAULT_CHECKPOINT, MusicGenerator  # noqa: E402
from tracking import TrackingPipeline, build_detector  # noqa: E402
from translator import TranslationService, build_default_service  # noqa: E402

DEMO_VIDEO = REPO_ROOT / "Task4" / "samples" / "demo_scene.mp4"
OUTPUT_DIR = REPO_ROOT / "outputs"

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_PLURALS = {"person": "people", "bus": "buses"}
# Pentatonic C scale: any mix of these notes sounds consonant, so the seed
# phrase built from arbitrary objects still sounds musical.
_PENTATONIC = [60, 62, 64, 67, 69, 72, 74, 76, 79]


def describe_scene(report: SceneReport) -> str:
    """English sentence for a report, e.g. "The scene has one car and two people"."""
    parts = []
    for label, count in sorted(report.label_counts.items()):
        word = _NUMBER_WORDS.get(count, str(count))
        noun = label if count == 1 else _PLURALS.get(label, label + "s")
        parts.append(f"{word} {noun}")
    return "The scene has " + (" and ".join(parts) if parts else "no objects") + "."


def label_to_pitch(label: str) -> int:
    """Stable pitch for a label (crc32, unlike ``hash()``, is not randomised per run)."""
    return _PENTATONIC[zlib.crc32(label.encode()) % len(_PENTATONIC)]


class CodeLabHub:
    def __init__(self, translator: Optional[TranslationService] = None,
                 bot: Optional[FAQBot] = None, checkpoint: Path = DEFAULT_CHECKPOINT):
        self._translator, self._bot = translator, bot
        self._checkpoint, self._generator = checkpoint, None

    # -- lazily built collaborators (keeps start-up fast) ---------------------
    @property
    def translator(self) -> TranslationService:
        if self._translator is None:
            self._translator = build_default_service()
        return self._translator

    @property
    def bot(self) -> FAQBot:
        if self._bot is None:
            self._bot = FAQBot()
        return self._bot

    @property
    def generator(self) -> MusicGenerator:
        if self._generator is None:
            if not Path(self._checkpoint).exists():
                raise FileNotFoundError(
                    f"No trained model at {self._checkpoint}. Run: python Task3/cli.py train")
            self._generator = MusicGenerator.load(Path(self._checkpoint))
        return self._generator

    # -- the three links ------------------------------------------------------
    def track_scene(self, video: Path = DEMO_VIDEO, output: Optional[Path] = None,
                    detector: str = "auto", codec: str = "mp4v") -> SceneReport:
        """Task4: detect + track everything in a video.

        The synthetic demo scene is coloured shapes, so it always uses the motion
        detector with colour labels; real videos use YOLO when it is installed.
        """
        video = Path(video)
        is_demo = video.resolve() == DEMO_VIDEO.resolve()
        if is_demo and not video.exists():
            from make_demo_video import make_demo_video  # generated on demand
            make_demo_video(DEMO_VIDEO)
        chosen = build_detector("motion", color_labels=True) if is_demo else build_detector(detector)
        return TrackingPipeline(chosen).run(video, output=output, codec=codec)

    def narrate(self, report: SceneReport, target: str = "id") -> Tuple[str, TranslationResult]:
        """Task1: turn the report into a sentence and translate it."""
        english = describe_scene(report)
        return english, self.translator.translate(english, "en", target)

    def sonify(self, report: SceneReport, out_dir: Path = OUTPUT_DIR,
               name: str = "scene_music") -> Tuple[List[str], Path, Path]:
        """Task3: compose music shaped by the scene.

        * which objects were seen  -> the seed phrase (one note per object)
        * how fast they moved      -> sampling temperature (calm scene = safe melody)
        * how long the video is    -> piece length
        """
        seed_pitches = [label_to_pitch(label) for _, label in sorted(report.tracks.items())] or [60]
        temperature = min(1.3, max(0.6, 0.6 + report.mean_speed / 8))
        seconds = report.frames / report.fps if report.fps else 6.0
        length = int(min(96, max(24, seconds * 4)))
        rng_seed = zlib.crc32(",".join(sorted(report.tracks.values())).encode())  # reproducible
        gen = self.generator
        tokens = gen.generate(length, temperature, seed_tokens=gen.seed_from_pitches(seed_pitches),
                              rng_seed=rng_seed)
        out_dir = Path(out_dir)
        return (tokens, tokens_to_midi(tokens, out_dir / f"{name}.mid"),
                tokens_to_wav(tokens, out_dir / f"{name}.wav"))
