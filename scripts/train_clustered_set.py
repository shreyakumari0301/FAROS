"""Set encoder on clustered dynamic CVRP.

Future customers keep their cluster locations. Reveal times of customers that
are still hidden are resampled inside that cluster's window. The model sees
only customers already revealed, including the time they appeared.
"""

from __future__ import annotations

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
    generate_clustered_instance,
    initial_state,
    policy_action,
    rank_nearest,
)
from scripts.train_arrival_value import evaluate
from scripts.train_set_encoder import DeepSet, prediction_gradient, standardize

GLOBAL_NAMES = (
    "time_fraction",
    "capacity_slack",
    "unknown_fraction",
    "current_x",
    "current_y",
    "depot_distance",
    "immediate_cost",
)


def candidate_inputs(inst: dict, post: dict, immediate: float):
    n_customers = inst["demands"].shape[0] - 1
    capacity = float(inst["capacity"])
    horizon = float(inst["horizon"])
    time = float(post["time"])
    current = int(post["current"])
    known = inst["reveal"] <= time + 1e-9
    unvisited = ~post["visited"]
    active = known & unvisited
    active[0] = False
    nodes = np.stack(
        [
            inst["coords"][1:, 0],
            inst["coords"][1:, 1],
            inst["demands"][1:] / capacity,
            inst["dist"][current, 1:],
            inst["dist"][0, 1:],
            inst["reveal"][1:] / horizon,
        ],
        axis=1,
    )
    mask = active[1:]
    nodes = nodes * mask[:, None]
    n_unknown = int((unvisited[1:] & ~known[1:]).sum())
    xy = inst["coords"][current]
    context = np.array(
        [
            time / horizon,
            float(post["remaining_capacity"]) / capacity,
            n_unknown / n_customers,
            float(xy[0]),
            float(xy[1]),
            float(inst["dist"][current, 0]),
            immediate,
        ],
        dtype=np.float64,
    )
    return nodes.astype(np.float64), mask, context


def score_candidates(inst: dict, state: dict, candidates: list, n_futures: int, rng: random.Random) -> np.ndarray:
    decision_time = float(state["time"])
    posts = [apply_action(inst, state, "go", node) for node in candidates]
    k = len(candidates)
    batch = k * n_futures
    reveal = np.repeat(inst["reveal"][None, :], batch, axis=0)
    unknown = np.flatnonzero(inst["reveal"] > decision_time + 1e-9)
    unknown = unknown[unknown != 0]
    if unknown.size:
        low = np.maximum(inst["window_start"][unknown], decision_time)
        high = inst["window_start"][unknown] + inst["window_width"][unknown]
        span = np.maximum(high - low, 0.0)
        draws = np.array(
            [rng.random() for _ in range(batch * int(unknown.size))],
            dtype=np.float64,
        ).reshape(batch, unknown.size)
        sampled = low + draws * span
        reveal[:, unknown] = np.where(span > 0.0, sampled, decision_time)
    visited = np.repeat(np.stack([post["visited"] for post in posts]), n_futures, axis=0)
    current = np.repeat([int(post["current"]) for post in posts], n_futures)
    time = np.repeat([float(post["time"]) for post in posts], n_futures)
    remaining = np.repeat([float(post["remaining_capacity"]) for post in posts], n_futures)
    cost = np.repeat([float(post["cost"]) for post in posts], n_futures)
    finished = batched_rollout_cost(
        inst["dist"], inst["demands"], reveal, visited, current, time, remaining, cost, float(inst["capacity"])
    )
    return finished.reshape(k, n_futures).mean(axis=1)


def collect_instance(inst: dict, top_k: int, n_futures: int, rng: random.Random):
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    horizon = float(inst["horizon"])
    groups = []
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, _node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            ranked = rank_nearest(inst, state, feasible_customers(inst, state))
            candidates = ranked[:top_k]
            if len(candidates) >= 2 and float(state["time"]) < horizon - 1e-8:
                before = float(state["cost"])
                posts = [apply_action(inst, state, "go", action) for action in candidates]
                packed = [candidate_inputs(inst, post, float(post["cost"]) - before) for post in posts]
                costs = score_candidates(inst, state, candidates, n_futures, rng)
                groups.append((packed, costs))
            state = apply_action(inst, state, "go", ranked[0])
            if bool(state["visited"][1:].all()):
                break
            continue
        state = apply_action(inst, state, kind, 0)
    else:
        raise RuntimeError("instance exceeded event limit")
    return groups


def pad_groups(groups, width, n_customers):
    nodes = np.zeros((len(groups), width, n_customers, 6), dtype=np.float64)
    mask = np.zeros((len(groups), width, n_customers), dtype=bool)
    context = np.zeros((len(groups), width, len(GLOBAL_NAMES)), dtype=np.float64)
    costs = np.zeros((len(groups), width), dtype=np.float64)
    valid = np.zeros((len(groups), width), dtype=bool)
    for index, (packed, cost) in enumerate(groups):
        for candidate, (node, node_mask, ctx) in enumerate(packed):
            nodes[index, candidate] = node
            mask[index, candidate] = node_mask
            context[index, candidate] = ctx
        costs[index, : len(packed)] = cost
        valid[index, : len(packed)] = True
    return nodes, mask, context, costs, valid


def recent_centroid_choice(nodes, mask, context, valid):
    """Pick the candidate closest to the latest revealed known customers."""
    choice = np.zeros(nodes.shape[0], dtype=np.int64)
    for group in range(nodes.shape[0]):
        best_score = np.inf
        best = 0
        for candidate in range(nodes.shape[1]):
            if not valid[group, candidate]:
                continue
            active = mask[group, candidate]
            if not active.any():
                score = np.inf
            else:
                revealed = nodes[group, candidate, active]
                cutoff = np.quantile(revealed[:, 5], 0.7)
                recent = revealed[revealed[:, 5] >= cutoff - 1e-12]
                centroid = recent[:, :2].mean(axis=0)
                vehicle = context[group, candidate, 3:5]
                score = float(np.linalg.norm(vehicle - centroid))
            if score < best_score:
                best_score = score
                best = candidate
        choice[group] = best
    return choice


def choice_metrics(choice, costs, valid, label):
    masked = np.where(valid, costs, np.inf)
    rows = np.arange(costs.shape[0])
    myopic = costs[:, 0]
    chosen = costs[rows, choice]
    oracle = masked.min(axis=1)
    return {
        "method": label,
        "mean_cost": float(chosen.mean()),
        "myopic_mean_cost": float(myopic.mean()),
        "oracle_mean_cost": float(oracle.mean()),
        "mean_gap_vs_myopic": float((myopic - chosen).mean()),
        "top1_accuracy": float(np.mean(choice == np.argmin(masked, axis=1))),
    }


def train_model(nodes, mask, context, target, valid, seed, epochs, hidden):
    model = DeepSet(random.Random(seed), node_dim=6, global_dim=7, hidden=hidden)
    order = list(range(nodes.shape[0]))
    rng = random.Random(seed + 1)
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            index = np.array(order[start:start + 64])
            pred, cache = model.forward(nodes[index], mask[index], context[index])
            grad = prediction_gradient(pred, target[index], valid[index])
            model.adam(model.backward(cache, grad), 1e-3, 1e-4)
    return model


def run_setting(sigma, n_clusters, seed=1234):
    instances, customers, horizon, capacity = 20, 200, 40.0, 80.0
    top_k, futures, train_count = 5, 20, 16
    rng = random.Random(seed)
    grouped = []
    for episode in range(instances):
        inst = generate_clustered_instance(
            customers, capacity, horizon, rng, n_clusters=n_clusters, sigma=sigma
        )
        groups = collect_instance(inst, top_k, futures, rng)
        grouped.append(groups)
        print(
            f"sigma={sigma} clusters={n_clusters} episode {episode}: {len(groups)} decisions",
            flush=True,
        )
    train_groups = [group for episode in grouped[:train_count] for group in episode]
    test_groups = [group for episode in grouped[train_count:] for group in episode]
    train = pad_groups(train_groups, top_k, customers)
    test = pad_groups(test_groups, top_k, customers)
    heuristic = recent_centroid_choice(test[0], test[1], test[2], test[4])
    heuristic_metrics = choice_metrics(heuristic, test[3], test[4], "recent_centroid")
    print(json.dumps(heuristic_metrics), flush=True)
    (train_nodes, train_mask, train_ctx, train_y, train_valid), (
        test_nodes, test_mask, test_ctx, _test_y, test_valid,
    ) = standardize(train, test)
    model = train_model(train_nodes, train_mask, train_ctx, train_y, train_valid, seed, epochs=40, hidden=64)
    pred, _ = model.forward(test_nodes, test_mask, test_ctx)
    metrics = evaluate(pred, test[3], test[4])
    train_pred, _ = model.forward(train_nodes, train_mask, train_ctx)
    train_metrics = evaluate(train_pred, train[3], train[4])
    metrics.update(
        {
            "sigma": sigma,
            "n_clusters": n_clusters,
            "train_decisions": len(train_groups),
            "train_model_mean_cost": train_metrics["model_mean_cost"],
            "train_myopic_mean_cost": train_metrics["myopic_mean_cost"],
            "heuristic": heuristic_metrics,
            "beats_myopic": bool(
                metrics["mean_model_gap"] > max(0.02, 0.15 * metrics["mean_oracle_gap"])
            ),
        }
    )
    return metrics


def main():
    tried = []
    for sigma, n_clusters in ((0.04, 4), (0.02, 4), (0.015, 8)):
        metrics = run_setting(sigma, n_clusters)
        tried.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        out = Path(ROOT) / "outputs" / "logs" / "clustered_set_encoder.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"tried": tried, "best": metrics}, handle, indent=2)
        if metrics["beats_myopic"]:
            print("BEAT myopic nearest-feasible", flush=True)
            return
    print("DID NOT beat myopic on these clustered settings", flush=True)


if __name__ == "__main__":
    main()
