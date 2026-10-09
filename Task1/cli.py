"""Command line front-end:  python Task1/cli.py "hello world" -t id"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.console import use_utf8_console  # noqa: E402
from common.paths import add_task_paths  # noqa: E402

use_utf8_console()
add_task_paths()
from translator import TranslationError, build_default_service, is_offline_result  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate text from the terminal.")
    parser.add_argument("text", help="text to translate")
    parser.add_argument("-s", "--source", default="auto", help="source code or 'auto'")
    parser.add_argument("-t", "--target", required=True, help="target language code")
    parser.add_argument("--offline", action="store_true", help="use only the offline lexicon")
    args = parser.parse_args(argv)
    try:
        result = build_default_service(args.offline).translate(args.text, args.source, args.target)
    except TranslationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{result.text}\n[{result.provider}]")
    if is_offline_result(result):
        print("note: word-by-word offline fallback - no online translator was reachable",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
