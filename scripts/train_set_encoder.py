"""Score each candidate from the exact set of known customers after the action.

Hidden customers are masked out. The target is the mean completion cost over
future-arrival samples. Decisions after every customer has appeared are dropped.
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
from scripts.train_arrival_value import batch_loss, evaluate

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
        ],
        axis=1,
    )
    mask = active[1:].astype(bool)
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
                packed = [
                    candidate_inputs(inst, post, float(post["cost"]) - before)
                    for post in posts
                ]
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
    nodes = np.zeros((len(groups), width, n_customers, 5), dtype=np.float64)
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


def _randn(rng, shape, scale):
    values = [rng.gauss(0.0, scale) for _ in range(int(np.prod(shape)))]
    return np.array(values, dtype=np.float64).reshape(shape)


class DeepSet:
    def __init__(self, rng: random.Random, node_dim=5, global_dim=7, hidden=32):
        self.w_phi = _randn(rng, (node_dim, hidden), 1.0 / np.sqrt(node_dim))
        self.b_phi = np.zeros(hidden)
        self.w2 = _randn(rng, (hidden + global_dim, hidden), 1.0 / np.sqrt(hidden + global_dim))
        self.b2 = np.zeros(hidden)
        self.w3 = _randn(rng, (hidden, 1), 1.0 / np.sqrt(hidden))
        self.b3 = np.zeros(1)
        self.params = [self.w_phi, self.b_phi, self.w2, self.b2, self.w3, self.b3]
        self.moment = [np.zeros_like(p) for p in self.params]
        self.velocity = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def forward(self, nodes, mask, context):
        hidden = np.maximum(nodes @ self.w_phi + self.b_phi, 0.0)
        denom = np.maximum(mask.sum(axis=2, keepdims=True), 1.0)
        pooled = (hidden * mask[..., None]).sum(axis=2) / denom
        joined = np.concatenate([pooled, context], axis=-1)
        second = np.maximum(joined @ self.w2 + self.b2, 0.0)
        pred = (second @ self.w3 + self.b3).reshape(second.shape[:2])
        return pred, (nodes, mask, hidden, denom, joined, second)

    def backward(self, cache, grad):
        nodes, mask, hidden, denom, joined, second = cache
        g = grad.reshape(*grad.shape, 1)
        gw3 = second.reshape(-1, second.shape[-1]).T @ g.reshape(-1, 1)
        gb3 = g.reshape(-1, 1).sum(axis=0)
        d_second = (g @ self.w3.T).reshape(second.shape)
        d_second *= second > 0
        flat_joined = joined.reshape(-1, joined.shape[-1])
        flat_second = d_second.reshape(-1, d_second.shape[-1])
        gw2 = flat_joined.T @ flat_second
        gb2 = flat_second.sum(axis=0)
        d_joined = (d_second @ self.w2.T).reshape(joined.shape)
        d_pooled = d_joined[..., : hidden.shape[-1]]
        scale = d_pooled / denom.reshape(d_pooled.shape[0], d_pooled.shape[1], 1)
        d_hidden = scale[:, :, None, :] * mask[..., None]
        d_hidden *= hidden > 0
        flat_nodes = nodes.reshape(-1, nodes.shape[-1])
        flat_hidden = d_hidden.reshape(-1, d_hidden.shape[-1])
        gw_phi = flat_nodes.T @ flat_hidden
        gb_phi = flat_hidden.sum(axis=0)
        return [gw_phi, gb_phi, gw2, gb2, gw3, gb3]

    def adam(self, grads, lr, weight_decay):
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for param, grad, moment, velocity in zip(self.params, grads, self.moment, self.velocity):
            grad = grad + weight_decay * param
            moment *= beta1
            moment += (1.0 - beta1) * grad
            velocity *= beta2
            velocity += (1.0 - beta2) * (grad * grad)
            moment_hat = moment / (1.0 - beta1 ** self.step)
            velocity_hat = velocity / (1.0 - beta2 ** self.step)
            param -= lr * moment_hat / (np.sqrt(velocity_hat) + 1e-8)


def standardize(train, test):
    nodes, mask, context, costs, valid = train
    observed = nodes[mask]
    node_mean = observed.mean(axis=0)
    node_std = observed.std(axis=0)
    node_std[node_std < 1e-8] = 1.0
    ctx_rows = context[valid]
    ctx_mean = ctx_rows.mean(axis=0)
    ctx_std = ctx_rows.std(axis=0)
    ctx_std[ctx_std < 1e-8] = 1.0
    y_mean = float(costs[valid].mean())
    y_std = float(costs[valid].std())
    if y_std < 1e-8:
        y_std = 1.0

    def apply(pack):
        n, m, c, y, v = pack
        scaled_nodes = (n - node_mean) * m[..., None] / node_std
        scaled_ctx = (c - ctx_mean) / ctx_std
        return scaled_nodes, m, scaled_ctx, (y - y_mean) / y_std, v

    return apply(train), apply(test)


def prediction_gradient(pred, target, valid):
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


def train_model(nodes, mask, context, target, valid, seed, epochs=25, batch_size=64, lr=1e-3):
    model = DeepSet(random.Random(seed))
    order = list(range(nodes.shape[0]))
    rng = random.Random(seed + 1)
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            index = np.array(order[start:start + batch_size])
            pred, cache = model.forward(nodes[index], mask[index], context[index])
            grad = prediction_gradient(pred, target[index], valid[index])
            model.adam(model.backward(cache, grad), lr, 1e-4)
    return model


def check_gradients():
    rng = random.Random(0)
    model = DeepSet(rng, hidden=4)
    nodes = _randn(rng, (2, 2, 3, 5), 0.1)
    mask = np.array(
        [[[1, 1, 0], [1, 0, 0]], [[1, 1, 1], [0, 1, 0]]],
        dtype=bool,
    )
    context = _randn(rng, (2, 2, 7), 0.1)
    pred, cache = model.forward(nodes, mask, context)
    grad = np.ones_like(pred)
    grads = model.backward(cache, grad)
    epsilon = 1e-5
    numerical = np.zeros_like(model.w3)
    for index in np.ndindex(model.w3.shape):
        model.w3[index] += epsilon
        plus, _ = model.forward(nodes, mask, context)
        model.w3[index] -= 2 * epsilon
        minus, _ = model.forward(nodes, mask, context)
        model.w3[index] += epsilon
        numerical[index] = (plus - minus).sum() / (2 * epsilon)
    error = float(np.max(np.abs(grads[4] - numerical)))
    if error > 1e-4:
        raise AssertionError(f"set-encoder gradient error {error}")
    print(f"set-encoder gradient check: OK ({error:.2e})")


def main():
    if "--check" in sys.argv:
        check_gradients()
        return
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
    train = pad_groups(train_groups, top_k, customers)
    test = pad_groups(test_groups, top_k, customers)
    (train_nodes, train_mask, train_ctx, train_y, train_valid), (
        test_nodes, test_mask, test_ctx, _test_y, test_valid,
    ) = standardize(train, test)
    model = train_model(train_nodes, train_mask, train_ctx, train_y, train_valid, seed)
    pred, _ = model.forward(test_nodes, test_mask, test_ctx)
    metrics = evaluate(pred, test[3], test[4])
    metrics.update(
        {
            "train_decisions": len(train_groups),
            "test_instances": instances - train_count,
            "global_features": list(GLOBAL_NAMES),
            "node_features": ["x", "y", "demand", "dist_to_vehicle", "dist_to_depot"],
            "pass_mean_cost_below_myopic": bool(metrics["model_mean_cost"] < metrics["myopic_mean_cost"]),
            "note": "Each candidate is the exact post-action set of known unvisited customers. Hidden customers are masked.",
        }
    )
    out = Path(ROOT) / "outputs" / "logs" / "set_encoder.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
