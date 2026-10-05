from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def to_local_path(path: Union[Path, str]) -> Path:
    """Map Windows paths to WSL mount paths when running under WSL."""
    p = str(path).replace("\\", "/")
    if os.name != "nt" and len(p) >= 3 and p[1] == ":" and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)
