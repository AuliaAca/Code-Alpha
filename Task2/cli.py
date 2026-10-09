"""Terminal chat:  python Task2/cli.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.console import use_utf8_console  # noqa: E402
from common.paths import add_task_paths  # noqa: E402

use_utf8_console()
add_task_paths()
from faq_bot import FAQBot  # noqa: E402


def main() -> None:
    bot = FAQBot()
    print("CodeLab FAQ bot. Type 'exit' to quit.")
    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        reply = bot.reply(message)
        print(f"bot > {reply.text}")
        if message.lower() in {"bye", "exit", "quit"}:
            break


if __name__ == "__main__":
    main()
