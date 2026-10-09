"""Task3 command line.

    python Task3/cli.py train                       # learn from Bach chorales
    python Task3/cli.py generate --length 64        # write .mid + .wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.console import use_utf8_console  # noqa: E402
from common.paths import add_task_paths  # noqa: E402

use_utf8_console()
add_task_paths()
from music_gen.data import (cache_corpus, load_bach_corpus, load_midi_folder,  # noqa: E402
                            read_cached_corpus)
from music_gen.export import tokens_to_midi, tokens_to_wav  # noqa: E402
from music_gen.generate import DEFAULT_CHECKPOINT, MusicGenerator  # noqa: E402
from music_gen.train import TrainConfig, train  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS_CACHE = HERE / "models" / "corpus_tokens.json"


def cmd_train(args) -> None:
    if args.midi_dir:
        pieces = load_midi_folder(Path(args.midi_dir))
    elif CORPUS_CACHE.exists() and not args.refresh:
        pieces = read_cached_corpus(CORPUS_CACHE)  # skip the slow music21 parse
    else:
        pieces = load_bach_corpus(args.pieces)
        cache_corpus(pieces, CORPUS_CACHE)
    print(f"{len(pieces)} pieces, {sum(map(len, pieces))} tokens")
    cfg = TrainConfig(epochs=args.epochs)
    train(pieces, cfg, Path(args.out),
          progress=lambda m: print(f"epoch {m['epoch']:>2}  train {m['train_loss']:.3f}  "
                                   f"val {m['val_loss']:.3f}  acc {m['val_acc']:.1%}"))
    print(f"saved {args.out}")


def cmd_generate(args) -> None:
    gen = MusicGenerator.load(Path(args.checkpoint))
    tokens = gen.generate(args.length, args.temperature, args.top_k, rng_seed=args.seed)
    out = Path(args.out_dir)
    print("MIDI:", tokens_to_midi(tokens, out / f"{args.name}.mid"))
    print("WAV: ", tokens_to_wav(tokens, out / f"{args.name}.wav"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--pieces", type=int, default=150, help="Bach chorales to parse")
    t.add_argument("--midi-dir", help="train on your own .mid files instead")
    t.add_argument("--refresh", action="store_true", help="ignore the cached corpus")
    t.add_argument("--out", default=str(DEFAULT_CHECKPOINT))
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate")
    g.add_argument("--length", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=12)
    g.add_argument("--seed", type=int, default=None, help="rng seed for reproducible output")
    g.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    g.add_argument("--out-dir", default=str(HERE / "outputs"))
    g.add_argument("--name", default="generated")
    g.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
