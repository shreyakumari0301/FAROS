"""Realized tour cost of three dispatchers on held-out burst instances.

The learned scorer is trained on episodes 0-11, early-stopped on 12-15, and
never sees episodes 16-19. Depot returns and waits stay the nearest-feasible
rule. Only the choice among known feasible customers changes. The tour uses
the instance's true reveal times.
"""

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
    density_choice,
    learned_choice_factory,
    nearest_choice,
    rollout_tour,
    train_burst_model,
)


def summarize(rows):
    costs = np.array([row["cost"] for row in rows], dtype=np.float64)
    return {
        "episodes": [row["episode"] for row in rows],
        "costs": [float(c) for c in costs],
        "mean_cost": float(costs.mean()),
        "mean_go_decisions": float(np.mean([row["goes"] for row in rows])),
        "mean_deviations_from_nearest": float(np.mean([row["differs"] for row in rows])),
    }


def main():
    seed = 1234
    instances, model, apply = train_burst_model(seed)
    policies = {
        "nearest": nearest_choice,
        "density": density_choice,
        "learned": learned_choice_factory(model, apply),
    }
    report = {"seed": seed, "test_episodes": [16, 17, 18, 19], "policies": {}}
    for name, choose in policies.items():
        rows = []
        for episode in range(16, 20):
            cost, goes, differs, _cross = rollout_tour(instances[episode], choose)
            rows.append({"episode": episode, "cost": cost, "goes": goes, "differs": differs})
            print(f"{name} episode {episode}: tour {cost:.4f} deviations {differs}/{goes}", flush=True)
        report["policies"][name] = summarize(rows)
    nearest = report["policies"]["nearest"]["mean_cost"]
    for name in ("density", "learned"):
        report["policies"][name]["gap_vs_nearest"] = nearest - report["policies"][name]["mean_cost"]
    out = Path(ROOT) / "outputs" / "logs" / "burst_tours.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
