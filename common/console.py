"""Make terminal output safe on every OS.

On Windows, output that is redirected or piped uses the legacy code page
(e.g. cp1252) and crashes on characters such as "👋", "•" or Japanese text.
Switching the streams to UTF-8 with ``errors="replace"`` avoids that.
"""
from __future__ import annotations

import sys


def use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a text stream (e.g. under some IDEs)
            pass
