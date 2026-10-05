"""Full burst tours across multiple seeds (train 0-11, val 12-15, test 16-19)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.burst_dispatch import (
    HORIZON,
    learned_choice_factory,
    nearest_choice,
    oracle_cross_burst_factory,
    rollout_tour,
    train_burst_model,
)

DEFAULT_SEEDS = (1234, 1235, 1236)


def eval_seed(seed: int, futures: int = 20):
    instances, model, apply = train_burst_model(seed, futures=futures)
    learn = learned_choice_factory(model, apply)
    oracle_cb = oracle_cross_burst_factory(HORIZON)
    results = {}
    for name, choose, cross in (
        ("nearest", nearest_choice, None),
        ("learned", learn, None),
        ("wait_oracle_at_cross_burst", nearest_choice, oracle_cb),
    ):
        costs = []
        for episode in range(16, 20):
            cost, _, _, _ = rollout_tour(
                instances[episode], choose, choose_at_cross=cross, horizon=HORIZON
            )
            costs.append(cost)
            print(f"seed {seed} {name} ep {episode}: {cost:.4f}", flush=True)
        results[name] = {
            "costs": [float(c) for c in costs],
            "mean_cost": float(np.mean(costs)),
        }
    base = results["nearest"]["mean_cost"]
    for name in ("learned", "wait_oracle_at_cross_burst"):
        results[name]["gap_vs_nearest"] = base - results[name]["mean_cost"]
    return {"seed": seed, "policies": results}


def aggregate(per_seed):
    names = ("nearest", "learned", "wait_oracle_at_cross_burst")
    summary = {}
    for name in names:
        means = [row["policies"][name]["mean_cost"] for row in per_seed]
        gaps = [row["policies"][name].get("gap_vs_nearest", 0.0) for row in per_seed]
        summary[name] = {
            "mean_cost_per_seed": means,
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "mean_gap_vs_nearest": float(np.mean(gaps)),
        }
    return summary


def main():
    probe = "--probe" in sys.argv
    seeds = (1234,) if probe else DEFAULT_SEEDS
    per_seed = [eval_seed(seed, futures=8 if probe else 20) for seed in seeds]
    report = {
        "seeds": list(seeds),
        "test_episodes": [16, 17, 18, 19],
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    out = Path(ROOT) / "outputs" / "logs" / "burst_tours_multiseed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
