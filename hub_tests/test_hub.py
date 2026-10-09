"""Integration tests: the tasks working together through the hub."""
import wave

import pytest

from common.contracts import SceneReport
from hub import CodeLabHub, Router
from hub.pipeline import describe_scene, label_to_pitch
from translator import build_default_service


@pytest.fixture(scope="module")
def hub():
    # offline_only keeps the tests independent of the network
    return CodeLabHub(translator=build_default_service(offline_only=True))


def report(**overrides):
    base = dict(frames=90, fps=30.0, tracks={1: "car", 2: "person", 3: "person"}, mean_speed=3.0)
    return SceneReport(**{**base, **overrides})


def test_describe_scene_pluralises():
    assert describe_scene(report()) == "The scene has one car and two people."
    assert describe_scene(SceneReport()) == "The scene has no objects."


def test_label_to_pitch_is_stable_and_in_range():
    assert label_to_pitch("car") == label_to_pitch("car")
    assert 60 <= label_to_pitch("bus") <= 79


def test_narrate_uses_translator(hub):
    english, translated = hub.narrate(report(), "es")
    assert english.startswith("The scene has") and "escena" in translated.text.lower()


def test_sonify_is_reproducible_and_scene_dependent(hub, tmp_path):
    a, midi, wav = hub.sonify(report(), tmp_path, "a")
    b, _, _ = hub.sonify(report(), tmp_path, "b")
    other, _, _ = hub.sonify(report(tracks={1: "ball"}, mean_speed=9.0), tmp_path, "c")
    assert a == b                      # same scene -> same music
    assert a != other                  # different scene -> different music
    assert midi.exists() and wave.open(str(wav)).getnframes() > 0


def test_full_chain_video_to_words_to_music(hub, tmp_path):
    scene = hub.track_scene(output=tmp_path / "t.mp4")           # Task4
    assert scene.label_counts == {"car": 1, "bus": 1, "person": 1, "ball": 1}
    _, translated = hub.narrate(scene, "id")                      # Task1
    assert "mobil" in translated.text
    tokens, midi, _ = hub.sonify(scene, tmp_path)                 # Task3
    assert len(tokens) >= 24 and midi.exists()


@pytest.mark.parametrize("message,handler", [
    ("translate hello to spanish", "Task1"),
    ("translate 'thank you' to id", "Task1"),
    ("compose a calm melody", "Task3"),
    ("describe the scene in french", "Task4"),
    ("how do I install the project?", "Task2"),
])
def test_router_picks_the_right_task(hub, message, handler):
    assert Router(hub).handle(message).handled_by == handler


def test_router_translation_result(hub):
    assert Router(hub).handle("translate hello to spanish").text.startswith("hola")


def test_router_unknown_language(hub):
    assert "don't know the language" in Router(hub).handle("translate hello to klingon").text


def test_scene_narration_in_unsupported_offline_language_degrades_gracefully(hub):
    reply = Router(hub).handle("describe the scene in japanese")
    assert reply.handled_by == "Task4" and "translation failed" in reply.text
