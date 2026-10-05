"""Frozen burst instance bank — same worlds for training labels and tour eval."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List

from env.dynamic_cvrp import generate_burst_instance

ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / "data" / "processed" / "burst_instances_v1"
CUSTOMERS = 200
CAPACITY = 80.0
HORIZON = 40.0
DEFAULT_EPISODES = 20


def bank_path(seed: int) -> Path:
    return BANK_DIR / f"seed_{seed}.pkl"


def generate_bank(seed: int, episodes: int = DEFAULT_EPISODES) -> List[Dict]:
    rng = random.Random(seed)
    return [
        generate_burst_instance(CUSTOMERS, CAPACITY, HORIZON, rng) for _ in range(episodes)
    ]


def save_bank(seed: int, instances: List[Dict]) -> Path:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    path = bank_path(seed)
    with open(path, "wb") as handle:
        pickle.dump(instances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "seed": seed,
        "episodes": len(instances),
        "customers": CUSTOMERS,
        "capacity": CAPACITY,
        "horizon": HORIZON,
        "generator": "generate_burst_instance",
    }
    with open(BANK_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return path


def load_bank(seed: int) -> List[Dict]:
    path = bank_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"missing instance bank: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_or_create_bank(seed: int, episodes: int = DEFAULT_EPISODES) -> List[Dict]:
    path = bank_path(seed)
    if path.exists():
        instances = load_bank(seed)
        if len(instances) >= episodes:
            return instances[:episodes]
    instances = generate_bank(seed, episodes)
    save_bank(seed, instances)
    return instances


def ensure_banks(seeds, episodes: int = DEFAULT_EPISODES) -> None:
    for seed in seeds:
        load_or_create_bank(seed, episodes)
        print(f"bank ok seed {seed} episodes {episodes} -> {bank_path(seed)}", flush=True)
