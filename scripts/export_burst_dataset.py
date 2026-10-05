"""Export cross-burst decision groups for later neural training (visible features only)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import generate_burst_instance
from scripts.train_burst_value import FEATURE_NAMES
from scripts.train_burst_wait import MIN_SEP, TOP_K, collect_instance_wait

ACTION_NAMES = ("go", "wait")


def record_from_group(episode: int, index: int, feat, cost, inst_meta):
    actions = []
    for row, c in zip(feat, cost):
        actions.append(
            {
                "features": {FEATURE_NAMES[i]: float(row[i]) for i in range(len(FEATURE_NAMES))},
                "mean_completion_cost": float(c),
            }
        )
    return {
        "episode": episode,
        "decision_index": index,
        "capacity": inst_meta["capacity"],
        "horizon": inst_meta["horizon"],
        "n_customers": inst_meta["n_customers"],
        "actions": actions,
        "best_index": int(cost.argmin()),
        "myopic_index": 0,
    }


def export_seed(seed: int, episodes: int, futures: int, out_dir: Path):
    rng = random.Random(seed)
    customers, horizon, capacity = 200, 40.0, 80.0
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": seed,
        "episodes": episodes,
        "futures": futures,
        "top_k": TOP_K,
        "min_sep": MIN_SEP,
        "feature_names": list(FEATURE_NAMES),
        "wait_feature_note": "Last action may be wait; feature vector uses wait_features()",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    for split_name, ep_range in (
        ("train", range(0, min(12, episodes))),
        ("val", range(12, min(16, episodes))),
        ("test", range(16, min(20, episodes))),
    ):
        path = out_dir / f"seed_{seed}_{split_name}.jsonl"
        count = 0
        with open(path, "w", encoding="utf-8") as handle:
            for episode in ep_range:
                inst = generate_burst_instance(customers, capacity, horizon, rng)
                groups = collect_instance_wait(inst, TOP_K, futures, rng, MIN_SEP)
                inst_meta = {
                    "capacity": capacity,
                    "horizon": horizon,
                    "n_customers": customers,
                }
                for index, (feat, cost) in enumerate(groups):
                    row = record_from_group(episode, index, feat, cost, inst_meta)
                    handle.write(json.dumps(row) + "\n")
                    count += 1
        print(f"seed {seed} {split_name}: {count} decisions -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", type=str, default="", help="comma-separated, overrides --seed")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--futures", type=int, default=20)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        args.episodes = 8
        args.futures = 8
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]
    base = Path(ROOT) / "data" / "processed" / "burst_decisions_v1"
    for seed in seeds:
        export_seed(seed, args.episodes, args.futures, base / f"seed_{seed}")


if __name__ == "__main__":
    main()
