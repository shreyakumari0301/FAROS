"""Train ranker from exported burst JSONL; evaluate one-step on test split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_arrival_value import MLP, batch_loss, evaluate
from scripts.train_burst_value import FEATURE_NAMES


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def to_arrays(rows):
    groups_x = []
    groups_y = []
    groups_v = []
    for row in rows:
        actions = row["actions"]
        k = len(actions)
        width = max(k, 1)
        x = np.zeros((width, len(FEATURE_NAMES)), dtype=np.float64)
        y = np.zeros(width, dtype=np.float64)
        v = np.zeros(width, dtype=bool)
        for index, action in enumerate(actions):
            feat = action["features"]
            x[index] = [feat[name] for name in FEATURE_NAMES]
            y[index] = action["mean_completion_cost"]
            v[index] = True
        groups_x.append(x)
        groups_y.append(y)
        groups_v.append(v)
    max_w = max(g.shape[0] for g in groups_x)
    n = len(groups_x)
    X = np.zeros((n, max_w, len(FEATURE_NAMES)))
    Y = np.zeros((n, max_w))
    V = np.zeros((n, max_w), dtype=bool)
    for i, (x, y, v) in enumerate(zip(groups_x, groups_y, groups_v)):
        X[i, : x.shape[0]] = x
        Y[i, : y.shape[0]] = y
        V[i, : v.shape[0]] = v
    return X, Y, V


def train_ranker(X, Y, V, seed, epochs=60):
    flat = X[V]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    y_mean = float(Y[V].mean())
    y_std = float(Y[V].std()) or 1.0
    Xn = (X - mean) / std
    Yn = (Y - y_mean) / y_std
    import random

    model = MLP(X.shape[-1], 64, random.Random(seed))
    order = list(range(X.shape[0]))
    rng = random.Random(seed + 1)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 128):
            idx = np.array(order[start : start + 128])
            pred, cache = model.forward(Xn[idx])
            _, diff, better = batch_loss(pred, Yn[idx], V[idx])
            grad = np.where(np.abs(diff) < 1.0, diff, np.sign(diff))
            grad = grad / max(float(V[idx].sum()), 1.0)
            if better.any():
                logit = pred[:, None, :] - pred[:, :, None]
                d_logit = -1.0 / (1.0 + np.exp(np.clip(logit, -40.0, 40.0)))
                d_logit = np.where(better, d_logit, 0.0)
                pair_count = better.sum(axis=(1, 2)).clip(min=1).astype(np.float64)
                d_logit /= pair_count[:, None, None]
                d_logit /= pred.shape[0]
                grad = grad + d_logit.sum(axis=1) - d_logit.sum(axis=2)
            model.adam(model.backward(cache, grad), 1e-3, 1e-4)

    def apply_pack(pack):
        x, y, v = pack
        return (x - mean) / std, (y - y_mean) / y_std, v

    return model, apply_pack


def main():
    base = Path(ROOT) / "data" / "processed" / "burst_decisions_v1" / "seed_1234"
    train = load_jsonl(base / "seed_1234_train.jsonl")
    test = load_jsonl(base / "seed_1234_test.jsonl")
    tr = to_arrays(train)
    te = to_arrays(test)
    model, apply = train_ranker(tr[0], tr[1], tr[2], 1234)
    Xn, Yn, V = apply(te)
    pred, _ = model.forward(Xn)
    metrics = evaluate(pred, te[1], te[2])
    out = Path(ROOT) / "outputs" / "logs" / "burst_export_ranker.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
