"""Wait vs serve at cross-burst states, plus oracle-augmented full tours."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.burst_dispatch import (
    HORIZON,
    collect_cross_burst_states,
    first_action_costs,
    learned_choice_factory,
    nearest_choice,
    oracle_cross_burst_factory,
    rollout_tour,
    train_burst_model,
)


def summarize_state_costs(rows):
    keys = ("nearest", "learned", "wait", "oracle")
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        if vals:
            out[f"mean_total_cost_{key}"] = float(np.mean(vals))
    if rows:
        out["mean_savings_oracle_vs_nearest"] = float(
            np.mean([row["nearest"] - row["oracle"] for row in rows])
        )
        waits = [row for row in rows if "wait" in row]
        if waits:
            out["mean_savings_wait_vs_nearest"] = float(
                np.mean([row["nearest"] - row["wait"] for row in waits])
            )
            out["fraction_wait_beats_nearest"] = float(
                np.mean([row["wait"] < row["nearest"] - 1e-9 for row in waits])
            )
        picks = Counter(row["oracle_pick"] for row in rows)
        out["oracle_pick_counts"] = dict(picks)
        out["fraction_oracle_chooses_wait"] = float(
            picks.get("wait", 0) / len(rows)
        )
    out["decisions"] = len(rows)
    return out


def main():
    probe = "--probe" in sys.argv
    seed = 1234
    instances, model, apply = train_burst_model(seed, futures=8 if probe else 20, episodes=8 if probe else 20)
    test_eps = [7] if probe else [16, 17, 18, 19]
    learn = learned_choice_factory(model, apply)
    oracle_cb = oracle_cross_burst_factory(HORIZON)

    state_rows = []
    for episode in test_eps:
        snaps = collect_cross_burst_states(instances[episode], HORIZON)
        for snap in snaps:
            costs = first_action_costs(instances[episode], snap, model, apply, HORIZON)
            costs["episode"] = episode
            state_rows.append(costs)
        print(f"episode {episode}: {len(snaps)} cross-burst states", flush=True)

    state_summary = summarize_state_costs(state_rows)

    tour_policies = {
        "nearest": (nearest_choice, None),
        "learned": (learn, None),
        "wait_oracle_at_cross_burst": (nearest_choice, oracle_cb),
    }
    tours = {}
    for name, (choose, cross) in tour_policies.items():
        rows = []
        for episode in test_eps:
            cost, goes, differs, cross_n = rollout_tour(
                instances[episode], choose, choose_at_cross=cross, horizon=HORIZON
            )
            rows.append(
                {
                    "episode": episode,
                    "cost": cost,
                    "goes": goes,
                    "differs": differs,
                    "cross_burst_decisions": cross_n,
                }
            )
            print(f"{name} episode {episode}: tour {cost:.4f}", flush=True)
        costs = np.array([r["cost"] for r in rows])
        tours[name] = {
            "mean_cost": float(costs.mean()),
            "costs": [float(c) for c in costs],
            "mean_cross_burst_decisions": float(np.mean([r["cross_burst_decisions"] for r in rows])),
        }
    nearest_mean = tours["nearest"]["mean_cost"]
    for name in ("learned", "wait_oracle_at_cross_burst"):
        tours[name]["gap_vs_nearest"] = nearest_mean - tours[name]["mean_cost"]

    report = {
        "seed": seed,
        "test_episodes": test_eps,
        "cross_burst_first_action": state_summary,
        "full_tours": tours,
        "note": (
            "First-action costs are total tour cost after one deviating move at a "
            "cross-burst state on the nearest trajectory, then nearest-feasible. "
            "Oracle may wait or go to a separated candidate. "
            "wait_oracle_at_cross_burst uses that oracle only at cross-burst stops."
        ),
    }
    out = Path(ROOT) / "outputs" / "logs" / "burst_wait.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
