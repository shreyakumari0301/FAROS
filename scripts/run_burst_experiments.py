"""Run burst evaluation scripts in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def run(script: str, extra=None):
    cmd = [str(PY), str(ROOT / "scripts" / script)]
    if extra:
        cmd.extend(extra)
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    probe = "--probe" in sys.argv
    extra = ["--probe"] if probe else []
    run("train_burst_wait.py", extra)
    run("eval_burst_wait.py", extra)
    run("eval_burst_multiseed.py", extra)


if __name__ == "__main__":
    main()
