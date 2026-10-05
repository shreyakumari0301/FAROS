"""I/O helpers for Phase-1 branch banks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_groups(path: str) -> List[dict]:
    p = Path(path)
    groups: List[dict] = []
    if p.is_dir():
        shards = sorted(p.glob("instance_*.jsonl"))
        if not shards:
            shards = sorted(p.glob("*.jsonl"))
        for shard in shards:
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        groups.append(json.loads(line))
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    groups.append(json.loads(line))
    return groups
