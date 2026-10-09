import wave

import pytest

from music_gen.data import (Vocabulary, make_token, make_windows, parse_token,
                            quantize_duration)
from music_gen.export import tokens_to_midi, tokens_to_wav
from music_gen.generate import MusicGenerator
from music_gen.train import TrainConfig, train

# A tiny repeating "scale" corpus: easy for a small model to learn quickly.
SCALE = [make_token([60 + i], 1.0) for i in (0, 2, 4, 5, 7, 9, 11, 12)]
PIECES = [SCALE * 6 for _ in range(4)]


def test_quantize_and_token_roundtrip():
    assert quantize_duration(0.6) == 0.5 and quantize_duration(3.4) == 4.0
    assert parse_token(make_token([64, 60, 67], 1.0)) == ([60, 64, 67], 1.0)
    assert parse_token(make_token([], 2.0)) == ([], 2.0)


def test_vocab_encode_decode_and_unknown():
    vocab = Vocabulary.build([["a@1.0", "a@1.0", "b@1.0"]], min_count=2)
    assert vocab.encode(["a@1.0", "b@1.0"]) == [vocab.stoi["a@1.0"], 0]  # rare -> <unk>
    assert Vocabulary.from_json(vocab.to_json()).itos == vocab.itos


def test_windows_shape():
    x, y = make_windows([[1, 2, 3, 4, 5]], seq_len=3)
    assert x == [[1, 2, 3], [2, 3, 4]] and y == [4, 5]


@pytest.fixture(scope="module")
def generator(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "m.pt"
    cfg = TrainConfig(seq_len=8, hidden=32, embed_dim=16, epochs=15, min_count=1)
    history = train(PIECES, cfg, path)
    assert history["train_loss"][-1] < history["train_loss"][0]  # it learns
    return MusicGenerator.load(path)


def test_generation_length_and_determinism(generator):
    a = generator.generate(20, temperature=0.8, rng_seed=5)
    b = generator.generate(20, temperature=0.8, rng_seed=5)
    assert len(a) == 20 and a == b
    assert "<unk>" not in a


def test_generation_uses_seed_phrase(generator):
    seed = generator.seed_from_pitches([60, 62, 64])
    assert len(generator.generate(5, seed_tokens=seed, rng_seed=1)) == 5


def test_export_files(generator, tmp_path):
    tokens = generator.generate(12, rng_seed=2) + [make_token([], 1.0), make_token([60, 64], 2.0)]
    midi = tokens_to_midi(tokens, tmp_path / "a.mid")
    wav = tokens_to_wav(tokens, tmp_path / "a.wav")
    assert midi.read_bytes()[:4] == b"MThd"  # valid MIDI header
    with wave.open(str(wav)) as w:
        assert w.getnframes() > 0 and w.getnchannels() == 1
