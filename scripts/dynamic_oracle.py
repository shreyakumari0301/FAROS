"""Dynamic CVRP oracle headroom test. No learned model.

At each stop with at least two feasible known customers, score the K nearest
by the mean nearest-feasible completion cost over M resampled futures of
customers not yet revealed. The realized day is only used to choose which
states the frozen dispatcher actually visits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import (
    apply_action,
    batched_rollout_cost,
    feasible_customers,
    generate_instance,
    initial_state,
    policy_action,
    rank_nearest,
    rollout_cost,
    self_test,
)


def score_candidates(inst: dict, state: dict, candidates: list, n_futures: int, rng: random.Random) -> np.ndarray:
    """Mean completion cost of each candidate under resampled later arrivals."""
    k = len(candidates)
    horizon = float(inst["horizon"])
    decision_time = float(state["time"])
    unknown = inst["reveal"] > decision_time + 1e-9
    unknown[0] = False
    n_unknown = int(unknown.sum())
    span = max(horizon - decision_time, 0.0)

    posts = [apply_action(inst, state, "go", node) for node in candidates]
    batch = k * n_futures
    n_nodes = inst["demands"].shape[0]
    reveal = np.repeat(inst["reveal"][None, :], batch, axis=0)
    if n_unknown and span > 0:
        draws = np.array(
            [rng.random() for _ in range(batch * n_unknown)],
            dtype=np.float64,
        ).reshape(batch, n_unknown)
        reveal[:, unknown] = decision_time + span * draws

    visited = np.stack([post["visited"] for post in posts], axis=0)
    visited = np.repeat(visited, n_futures, axis=0)
    current = np.repeat(np.array([int(post["current"]) for post in posts], dtype=np.int64), n_futures)
    time = np.repeat(np.array([float(post["time"]) for post in posts]), n_futures)
    remaining = np.repeat(np.array([float(post["remaining_capacity"]) for post in posts]), n_futures)
    cost = np.repeat(np.array([float(post["cost"]) for post in posts]), n_futures)

    finished = batched_rollout_cost(
        inst["dist"],
        inst["demands"],
        reveal,
        visited,
        current,
        time,
        remaining,
        cost,
        float(inst["capacity"]),
    )
    return finished.reshape(k, n_futures).mean(axis=1)


def run_instance(inst: dict, top_k: int, n_futures: int, rng: random.Random) -> list:
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    rows = []
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            ranked = rank_nearest(inst, state, feasible_customers(inst, state))
            candidates = ranked[:top_k]
            if len(candidates) >= 2:
                means = score_candidates(inst, state, candidates, n_futures, rng)
                best = int(np.argmin(means))
                rows.append(
                    {
                        "time": float(state["time"]),
                        "myopic_node": int(candidates[0]),
                        "best_node": int(candidates[best]),
                        "myopic_cost": float(means[0]),
                        "best_cost": float(means[best]),
                        "delta": float(means[0] - means[best]),
                        "best_rank": best,
                        "n_candidates": len(candidates),
                    }
                )
            state = apply_action(inst, state, "go", ranked[0])
            if bool(state["visited"][1:].all()):
                break
            continue
        state = apply_action(inst, state, kind, node)
    else:
        raise RuntimeError("instance exceeded event limit")
    final = rollout_cost(inst, initial_state(inst["demands"].shape[0], float(inst["capacity"])))
    return rows, final


def _slice_stats(rows: list) -> dict:
    delta = np.array([row["delta"] for row in rows], dtype=np.float64)
    ranks = np.array([row["best_rank"] for row in rows], dtype=np.int64)
    return {
        "decisions": int(delta.size),
        "rank_best_fraction": {
            str(rank): float(np.mean(ranks == rank)) for rank in range(int(ranks.max()) + 1)
        },
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_p90": float(np.percentile(delta, 90)),
        "delta_max": float(delta.max()),
        "fraction_gt_0": float(np.mean(delta > 1e-8)),
        "fraction_gt_0.01": float(np.mean(delta > 0.01)),
        "fraction_gt_0.05": float(np.mean(delta > 0.05)),
        "fraction_gt_0.1": float(np.mean(delta > 0.1)),
    }


def summarize(rows: list, horizon: float) -> dict:
    if not rows:
        raise RuntimeError("no branching decisions")
    summary = _slice_stats(rows)
    open_rows = [row for row in rows if row["time"] < horizon - 1e-8]
    closed_rows = [row for row in rows if row["time"] >= horizon - 1e-8]
    summary["while_arrivals_remain"] = _slice_stats(open_rows) if open_rows else {"decisions": 0}
    summary["after_all_revealed"] = _slice_stats(closed_rows) if closed_rows else {"decisions": 0}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--capacity", type=float, default=80.0)
    parser.add_argument("--horizon", type=float, default=40.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--futures", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rng = random.Random(args.seed)
    rows = []
    finals = []
    for episode in range(args.instances):
        inst = generate_instance(args.customers, args.capacity, args.horizon, rng)
        episode_rows, final_cost = run_instance(inst, args.top_k, args.futures, rng)
        for row in episode_rows:
            row["episode"] = episode
        rows.extend(episode_rows)
        finals.append(final_cost)
        print(
            f"episode {episode}: decisions={len(episode_rows)} "
            f"myopic_cost={final_cost:.3f}",
            flush=True,
        )

    summary = summarize(rows, args.horizon)
    summary.update(
        {
            "instances": args.instances,
            "customers": args.customers,
            "capacity": args.capacity,
            "horizon": args.horizon,
            "top_k": args.top_k,
            "futures": args.futures,
            "seed": args.seed,
            "myopic_cost_mean": float(np.mean(finals)),
            "note": (
                "Horizon is 40, not 1. Euclidean travel on CVRP200 lasts much longer "
                "than 1, so reveal times are spread across the operating day. "
                "T=1 would reveal every customer after the first unit of travel."
            ),
        }
    )
    out = Path(ROOT) / "outputs" / "logs" / "dynamic_oracle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "decisions": rows}, handle)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
