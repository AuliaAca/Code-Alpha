"""CodeAlpha CodeLab entry point.

    python main.py demo [--lang id]   # video -> narration -> music, end to end
    python main.py chat               # one assistant routed across all tasks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.console import use_utf8_console
from common.paths import REPO_ROOT, add_task_paths

use_utf8_console()
add_task_paths()
from hub import CodeLabHub, Router  # noqa: E402
from hub.pipeline import DEMO_VIDEO, OUTPUT_DIR, describe_scene  # noqa: E402
from translator import TranslationError, is_offline_result  # noqa: E402


def cmd_demo(args) -> int:
    hub = CodeLabHub()
    out = Path(args.out)
    print("[Task4] tracking the demo scene ...")
    try:
        report = hub.track_scene(Path(args.video) if args.video else DEMO_VIDEO,
                                 output=out / "tracked.mp4", detector=args.detector)
    except (IOError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"        {report.frames} frames, objects: {report.label_counts}")

    print(f"[Task1] narrating in {args.lang!r} ...")
    try:
        english, translated = hub.narrate(report, args.lang)
        print(f"        {english}\n        -> {translated.text}   ({translated.provider})")
        if is_offline_result(translated):
            print("        note: offline word-by-word fallback (no internet translator reachable)")
        translation, provider = translated.text, translated.provider
    except TranslationError as exc:  # keep going: the music step does not need it
        english, translation, provider = describe_scene(report), None, None
        print(f"        {english}\n        translation skipped: {exc}")

    print("[Task3] composing music from the scene ...")
    tokens, midi, wav = hub.sonify(report, out)
    print(f"        {len(tokens)} notes -> {midi.name}, {wav.name}")

    summary = {"objects": report.label_counts, "english": english, "translation": translation,
               "translation_provider": provider, "notes": len(tokens),
               "mean_speed_px_per_frame": round(report.mean_speed, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"\nAll outputs are in {out}")
    return 0


def cmd_chat(_args) -> int:
    router = Router()
    print("CodeLab assistant. Try: 'translate hello to spanish', 'compose a melody', "
          "'describe the scene in french', or ask a question. 'exit' quits.")
    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if message.lower() in {"exit", "quit"}:
            return 0
        if message:
            reply = router.handle(message)
            print(f"[{reply.handled_by}] {reply.text}" + (f"\n         file: {reply.artifact}" if reply.artifact else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the connected pipeline")
    demo.add_argument("--lang", default="id", help="language code for the narration")
    demo.add_argument("--video", help="your own video (default: generated demo scene)")
    demo.add_argument("--detector", choices=["auto", "yolo", "motion"], default="auto",
                      help="detector for --video (the demo scene always uses motion)")
    demo.add_argument("--out", default=str(OUTPUT_DIR))
    demo.set_defaults(func=cmd_demo)
    sub.add_parser("chat", help="routed assistant").set_defaults(func=cmd_chat)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
