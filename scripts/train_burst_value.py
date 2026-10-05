"""Beat nearest-feasible when the alternative is a different burst.

Candidates are spatially separated known customers, so the choice is between
a nearby leftover and a customer from a wave that just started. Features use
only customers already revealed.
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
    generate_burst_instance,
    initial_state,
    policy_action,
    rank_nearest,
)
from scripts.train_arrival_value import MLP, batch_loss, evaluate

FEATURE_NAMES = (
    "immediate_cost",
    "near_known",
    "near_open",
    "served_reveal",
    "near_mean_reveal",
    "near_max_reveal",
    "time_fraction",
    "capacity_slack",
    "depot_distance",
    "nearest_other_open",
    "served_isolation",
)
RADIUS = 0.12


def diverse_candidates(inst, state, top_k, min_sep):
    ranked = rank_nearest(inst, state, feasible_customers(inst, state))
    if len(ranked) < 2:
        return ranked
    chosen = [ranked[0]]
    for node in ranked[1:]:
        if all(float(inst["dist"][node, other]) >= min_sep for other in chosen):
            chosen.append(node)
        if len(chosen) == top_k:
            break
    if len(chosen) < 2:
        return ranked[:top_k]
    return chosen


def burst_features(inst, post, immediate, served):
    horizon = float(inst["horizon"])
    capacity = float(inst["capacity"])
    current = int(post["current"])
    time = float(post["time"])
    known = inst["reveal"] <= time + 1e-9
    known[0] = False
    near = known & (inst["dist"][current] <= RADIUS)
    unvisited = ~post["visited"]
    unvisited[0] = False
    near_open = near & unvisited
    open_known = np.flatnonzero(known & unvisited)
    if near.any():
        rel = inst["reveal"][near]
        mean_reveal = float(rel.mean()) / horizon
        max_reveal = float(rel.max()) / horizon
    else:
        mean_reveal = max_reveal = 0.0
    if open_known.size:
        nearest_other = float(inst["dist"][current, open_known].min())
    else:
        nearest_other = 1.0
    others = known.copy()
    others[served] = False
    if others.any():
        isolation = float(inst["dist"][served, np.flatnonzero(others)].min())
    else:
        isolation = 1.0
    return np.array(
        [
            immediate,
            float(near.sum()),
            float(near_open.sum()),
            float(inst["reveal"][served]) / horizon,
            mean_reveal,
            max_reveal,
            time / horizon,
            float(post["remaining_capacity"]) / capacity,
            float(inst["dist"][current, 0]),
            nearest_other,
            isolation,
        ],
        dtype=np.float64,
    )


def score_candidates(inst, state, candidates, n_futures, rng):
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


def collect_instance(inst, top_k, n_futures, rng, min_sep):
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    horizon = float(inst["horizon"])
    groups = []
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            candidates = diverse_candidates(inst, state, top_k, min_sep)
            separated = len(candidates) >= 2 and float(inst["dist"][candidates[0], candidates[1]]) >= min_sep
            if separated and float(state["time"]) < horizon - 1e-8:
                before = float(state["cost"])
                feats = []
                for action in candidates:
                    post = apply_action(inst, state, "go", action)
                    feats.append(burst_features(inst, post, float(post["cost"]) - before, action))
                costs = score_candidates(inst, state, candidates, n_futures, rng)
                groups.append((np.stack(feats), costs))
            state = apply_action(inst, state, "go", rank_nearest(inst, state, feasible_customers(inst, state))[0])
            if bool(state["visited"][1:].all()):
                break
            continue
        state = apply_action(inst, state, kind, node)
    else:
        raise RuntimeError("instance exceeded event limit")
    return groups


def pad_groups(groups, width):
    features = np.zeros((len(groups), width, len(FEATURE_NAMES)), dtype=np.float64)
    costs = np.zeros((len(groups), width), dtype=np.float64)
    valid = np.zeros((len(groups), width), dtype=bool)
    for index, (feat, cost) in enumerate(groups):
        k = feat.shape[0]
        features[index, :k] = feat
        costs[index, :k] = cost
        valid[index, :k] = True
    return features, costs, valid


def standardize_fit(source):
    flat = source[0][source[2]]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    y_mean = float(source[1][source[2]].mean())
    y_std = float(source[1][source[2]].std()) or 1.0

    def apply(pack):
        x, y, v = pack
        return (x - mean) / std, (y - y_mean) / y_std, v

    return apply


def _grad(pred, target, valid):
    _, diff, better = batch_loss(pred, target, valid)
    grad = np.where(np.abs(diff) < 1.0, diff, np.sign(diff))
    grad = grad / max(float(valid.sum()), 1.0)
    if better.any():
        logit = pred[:, None, :] - pred[:, :, None]
        d_logit = -1.0 / (1.0 + np.exp(np.clip(logit, -40.0, 40.0)))
        d_logit = np.where(better, d_logit, 0.0)
        pair_count = better.sum(axis=(1, 2)).clip(min=1).astype(np.float64)
        d_logit /= pair_count[:, None, None]
        d_logit /= pred.shape[0]
        grad = grad + d_logit.sum(axis=1) - d_logit.sum(axis=2)
    return grad


def train_early_stop(train, val, seed, epochs=80):
    """Fit on earlier episodes. Keep the weights with the best validation cost."""
    apply = standardize_fit(train)
    train_x, train_y, train_v = apply(train)
    val_x, _val_y, val_v = apply(val)
    model = MLP(train_x.shape[-1], 32, random.Random(seed))
    order = list(range(train_x.shape[0]))
    rng = random.Random(seed + 1)
    best_cost = np.inf
    best_params = [p.copy() for p in model.params]
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 128):
            index = np.array(order[start:start + 128])
            pred, cache = model.forward(train_x[index])
            grad = _grad(pred, train_y[index], train_v[index])
            model.adam(model.backward(cache, grad), 1e-3, 1e-3)
        pred, _ = model.forward(val_x)
        cost = evaluate(pred, val[1], val[2])["model_mean_cost"]
        if cost < best_cost:
            best_cost = cost
            best_params = [p.copy() for p in model.params]
    for param, saved in zip(model.params, best_params):
        param[...] = saved
    return model, apply


def neighborhood_choice(features, valid):
    """Prefer the candidate sitting in the densest revealed neighborhood."""
    score = features[:, :, 2] - features[:, :, 10]
    score = np.where(valid, score, -np.inf)
    return np.argmax(score, axis=1)


def choice_metrics(choice, costs, valid):
    rows = np.arange(costs.shape[0])
    myopic = costs[:, 0]
    chosen = costs[rows, choice]
    oracle = np.where(valid, costs, np.inf).min(axis=1)
    return {
        "mean_cost": float(chosen.mean()),
        "myopic_mean_cost": float(myopic.mean()),
        "oracle_mean_cost": float(oracle.mean()),
        "mean_gap_vs_myopic": float((myopic - chosen).mean()),
    }


def run(instances, futures, seed=1234):
    customers, horizon, capacity = 200, 40.0, 80.0
    top_k, min_sep = 4, 0.25
    rng = random.Random(seed)
    grouped = []
    for episode in range(instances):
        inst = generate_burst_instance(customers, capacity, horizon, rng)
        groups = collect_instance(inst, top_k, futures, rng, min_sep)
        grouped.append(groups)
        print(f"episode {episode}: {len(groups)} cross-burst decisions", flush=True)
    if instances <= 4:
        split = [0, max(1, instances // 2), instances]
    else:
        split = [0, 12, 16, instances]
    train_groups = [group for episode in grouped[split[0]:split[1]] for group in episode]
    val_groups = [group for episode in grouped[split[1]:split[2]] for group in episode]
    test_groups = [group for episode in grouped[split[-2]:] for group in episode] if instances > 4 else val_groups
    if instances <= 4:
        test_groups = [group for episode in grouped[split[1]:] for group in episode]
        val_groups = test_groups
    train = pad_groups(train_groups, top_k)
    val = pad_groups(val_groups, top_k)
    test = pad_groups(test_groups, top_k)
    heuristic = choice_metrics(neighborhood_choice(test[0], test[2]), test[1], test[2])
    print("neighborhood heuristic", json.dumps(heuristic), flush=True)
    model, apply = train_early_stop(train, val, seed)
    test_x, _test_y, _test_v = apply(test)
    pred, _ = model.forward(test_x)
    metrics = evaluate(pred, test[1], test[2])
    train_x, _train_y, _train_v = apply(train)
    train_pred, _ = model.forward(train_x)
    train_metrics = evaluate(train_pred, train[1], train[2])
    metrics.update(
        {
            "train_decisions": len(train_groups),
            "val_decisions": len(val_groups),
            "train_model_mean_cost": train_metrics["model_mean_cost"],
            "train_myopic_mean_cost": train_metrics["myopic_mean_cost"],
            "train_oracle_mean_cost": train_metrics["oracle_mean_cost"],
            "neighborhood_heuristic": heuristic,
            "beats_myopic": bool(
                metrics["mean_model_gap"] > max(0.02, 0.15 * max(metrics["mean_oracle_gap"], 1e-8))
            ),
        }
    )
    return metrics


def main():
    probe = "--probe" in sys.argv
    metrics = run(instances=4 if probe else 20, futures=8 if probe else 20)
    print(json.dumps(metrics, indent=2), flush=True)
    if probe:
        return
    out = Path(ROOT) / "outputs" / "logs" / "burst_value.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("BEAT myopic" if metrics["beats_myopic"] else "DID NOT beat myopic", flush=True)


if __name__ == "__main__":
    main()
