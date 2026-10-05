"""Predict expected completion cost from the exact post-action state.

The target is the mean nearest-feasible cost under resampled future arrivals.
Hidden customer locations and demands are not model inputs. Decisions taken
after every customer has already appeared are excluded.
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
    feasible_customers,
    generate_instance,
    initial_state,
    policy_action,
    rank_nearest,
)
from scripts.dynamic_oracle import score_candidates

FEATURE_NAMES = (
    "time_fraction",
    "capacity_slack",
    "current_x",
    "current_y",
    "depot_distance",
    "known_remaining_fraction",
    "unknown_fraction",
    "known_demand_over_capacity",
    "mean_dist_known",
    "min_dist_known",
    "mean_depot_dist_known",
    "immediate_cost",
)


def post_features(inst: dict, post: dict, immediate: float) -> np.ndarray:
    n_customers = inst["demands"].shape[0] - 1
    capacity = float(inst["capacity"])
    horizon = float(inst["horizon"])
    time = float(post["time"])
    current = int(post["current"])
    known = inst["reveal"] <= time + 1e-9
    known[0] = False
    unvisited = ~post["visited"]
    unvisited[0] = False
    known_left = np.flatnonzero(known & unvisited)
    n_unknown = int((unvisited & ~known).sum())
    if known_left.size:
        dist_here = inst["dist"][current, known_left]
        mean_dist = float(dist_here.mean())
        min_dist = float(dist_here.min())
        mean_depot = float(inst["dist"][0, known_left].mean())
        known_demand = float(inst["demands"][known_left].sum())
    else:
        mean_dist = min_dist = mean_depot = known_demand = 0.0
    xy = inst["coords"][current]
    return np.array(
        [
            time / horizon,
            float(post["remaining_capacity"]) / capacity,
            float(xy[0]),
            float(xy[1]),
            float(inst["dist"][current, 0]),
            known_left.size / n_customers,
            n_unknown / n_customers,
            known_demand / capacity,
            mean_dist,
            min_dist,
            mean_depot,
            immediate,
        ],
        dtype=np.float64,
    )


def collect_instance(inst: dict, top_k: int, n_futures: int, rng: random.Random):
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    groups = []
    horizon = float(inst["horizon"])
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            ranked = rank_nearest(inst, state, feasible_customers(inst, state))
            candidates = ranked[:top_k]
            if len(candidates) >= 2 and float(state["time"]) < horizon - 1e-8:
                before = float(state["cost"])
                posts = [apply_action(inst, state, "go", action) for action in candidates]
                features = np.stack(
                    [
                        post_features(inst, post, float(post["cost"]) - before)
                        for post in posts
                    ]
                )
                costs = score_candidates(inst, state, candidates, n_futures, rng)
                groups.append((features, costs))
            state = apply_action(inst, state, "go", ranked[0])
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


def standardize(train, test):
    flat = train[0][train[2]]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    train_x = (train[0] - mean) / std
    test_x = (test[0] - mean) / std
    y_mean = float(train[1][train[2]].mean())
    y_std = float(train[1][train[2]].std())
    if y_std < 1e-8:
        y_std = 1.0
    return train_x, (train[1] - y_mean) / y_std, test_x, (test[1] - y_mean) / y_std, mean, std


class MLP:
    def __init__(self, n_in: int, hidden: int, rng: random.Random):
        self.w1 = _randn(rng, (n_in, hidden), 1.0 / np.sqrt(n_in))
        self.b1 = np.zeros(hidden)
        self.w2 = _randn(rng, (hidden, hidden), 1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = _randn(rng, (hidden, 1), 1.0 / np.sqrt(hidden))
        self.b3 = np.zeros(1)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.moment = [np.zeros_like(p) for p in self.params]
        self.velocity = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def forward(self, x):
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        pred = (h2 @ self.w3 + self.b3).reshape(x.shape[:-1])
        cache = (x, h1, h2)
        return pred, cache

    def backward(self, cache, grad):
        x, h1, h2 = cache
        g = grad.reshape(*grad.shape, 1)
        gw3 = h2.reshape(-1, h2.shape[-1]).T @ g.reshape(-1, 1)
        gb3 = g.reshape(-1, 1).sum(axis=0)
        gh2 = (g @ self.w3.T).reshape(h2.shape)
        gh2 *= h2 > 0
        flat_h1 = h1.reshape(-1, h1.shape[-1])
        flat_gh2 = gh2.reshape(-1, gh2.shape[-1])
        gw2 = flat_h1.T @ flat_gh2
        gb2 = flat_gh2.sum(axis=0)
        gh1 = (gh2 @ self.w2.T)
        gh1 *= h1 > 0
        flat_x = x.reshape(-1, x.shape[-1])
        flat_gh1 = gh1.reshape(-1, gh1.shape[-1])
        gw1 = flat_x.T @ flat_gh1
        gb1 = flat_gh1.sum(axis=0)
        return [gw1, gb1, gw2, gb2, gw3, gb3]

    def adam(self, grads, lr, weight_decay):
        self.step += 1
        b1, b2 = 0.9, 0.999
        for param, grad, moment, velocity in zip(self.params, grads, self.moment, self.velocity):
            grad = grad + weight_decay * param
            moment *= b1
            moment += (1 - b1) * grad
            velocity *= b2
            velocity += (1 - b2) * (grad * grad)
            moment_hat = moment / (1 - b1 ** self.step)
            velocity_hat = velocity / (1 - b2 ** self.step)
            param -= lr * moment_hat / (np.sqrt(velocity_hat) + 1e-8)


def _randn(rng: random.Random, shape, scale):
    values = [rng.gauss(0.0, scale) for _ in range(int(np.prod(shape)))]
    return np.array(values, dtype=np.float64).reshape(shape)


def batch_loss(pred, target, valid):
    diff = (pred - target) * valid
    abs_diff = np.abs(diff)
    huber = np.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    count = np.maximum(valid.sum(), 1.0)
    value = huber.sum() / count
    # Lower predicted cost should match lower target cost.
    gap = target[:, :, None] - target[:, None, :]
    better = (gap < -1e-6) & valid[:, :, None] & valid[:, None, :]
    logit = pred[:, None, :] - pred[:, :, None]
    rank = np.zeros(pred.shape[0], dtype=np.float64)
    if better.any():
        soft = np.logaddexp(0.0, -logit)
        pair_count = better.sum(axis=(1, 2)).clip(min=1)
        rank = (soft * better).sum(axis=(1, 2)) / pair_count
    return value + rank.mean(), diff, better


def train_model(features, target, valid, seed, epochs=30, batch_size=128, lr=1e-3):
    rng = random.Random(seed)
    model = MLP(features.shape[-1], 64, rng)
    n = features.shape[0]
    order = list(range(n))
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            index = np.array(order[start:start + batch_size])
            pred, cache = model.forward(features[index])
            _, diff, better = batch_loss(pred, target[index], valid[index])
            grad = np.where(np.abs(diff) < 1.0, diff, np.sign(diff))
            grad = grad / max(float(valid[index].sum()), 1.0)
            if better.any():
                # d/d(pred_j - pred_i) of softplus(-(pred_j-pred_i)) = -sigmoid(-(pred_j-pred_i))
                logit = pred[:, None, :] - pred[:, :, None]
                d_logit = -1.0 / (1.0 + np.exp(logit))
                d_logit = np.where(better, d_logit, 0.0)
                pair_count = better.sum(axis=(1, 2)).clip(min=1).astype(np.float64)
                d_logit /= pair_count[:, None, None]
                d_logit /= pred.shape[0]
                grad += d_logit.sum(axis=1) - d_logit.sum(axis=2)
            grads = model.backward(cache, grad)
            model.adam(grads, lr, 1e-4)
    return model


def evaluate(pred, costs, valid):
    masked_pred = np.where(valid, pred, np.inf)
    masked_cost = np.where(valid, costs, np.inf)
    model_index = np.argmin(masked_pred, axis=1)
    oracle_index = np.argmin(masked_cost, axis=1)
    rows = np.arange(costs.shape[0])
    myopic = costs[:, 0]
    chosen = costs[rows, model_index]
    oracle = masked_cost[rows, oracle_index]
    oracle_gap = myopic - oracle
    model_gap = myopic - chosen
    usable = oracle_gap > 1e-8
    recovered = np.zeros(costs.shape[0], dtype=np.float64)
    recovered[usable] = model_gap[usable] / oracle_gap[usable]
    return {
        "decisions": int(costs.shape[0]),
        "myopic_mean_cost": float(myopic.mean()),
        "model_mean_cost": float(chosen.mean()),
        "oracle_mean_cost": float(oracle.mean()),
        "mean_oracle_gap": float(oracle_gap.mean()),
        "mean_model_gap": float(model_gap.mean()),
        "gap_recovered": float(recovered[usable].mean()) if usable.any() else 0.0,
        "top1_accuracy": float(np.mean(model_index == oracle_index)),
        "myopic_top1_accuracy": float(np.mean(oracle_index == 0)),
    }


def main():
    instances = 20
    customers = 200
    capacity = 80.0
    horizon = 40.0
    top_k = 5
    futures = 20
    seed = 1234
    train_count = 16
    rng = random.Random(seed)
    grouped = []
    for episode in range(instances):
        inst = generate_instance(customers, capacity, horizon, rng)
        groups = collect_instance(inst, top_k, futures, rng)
        grouped.append(groups)
        print(f"collected episode {episode}: {len(groups)} decisions", flush=True)

    train_groups = [group for episode in grouped[:train_count] for group in episode]
    test_groups = [group for episode in grouped[train_count:] for group in episode]
    train = pad_groups(train_groups, top_k)
    test = pad_groups(test_groups, top_k)
    train_x, train_y, test_x, _, _, _ = standardize(train, test)
    model = train_model(train_x, train_y, train[2], seed)
    pred, _ = model.forward(test_x)
    # Predictions are in standardized cost units; argmin is unchanged.
    metrics = evaluate(pred, test[1], test[2])
    metrics.update(
        {
            "train_decisions": len(train_groups),
            "test_instances": instances - train_count,
            "features": list(FEATURE_NAMES),
            "note": "Inputs are the exact post-action state of known customers only. Target is mean completion cost over 20 futures.",
        }
    )
    out = Path(ROOT) / "outputs" / "logs" / "arrival_value.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
