"""Simple settings file for playing without remembering command-line options."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Change only these values.
PLAYERS = 2  # 2 = heads-up, 3 = multiplayer
USE_RESOLVER = True
SHOW_PROBABILITIES = True
MODEL_CHOICE = "sample"  # sample | greedy


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "presine",
        "--players",
        str(PLAYERS),
        "--model-choice",
        MODEL_CHOICE,
    ]
    if not USE_RESOLVER:
        command.append("--no-resolver")
    if SHOW_PROBABILITIES:
        command.append("--solver-learn")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
