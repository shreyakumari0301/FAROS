"""Train wait + go scorer at cross-burst states; evaluate one-step and full tours."""

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
from scripts.burst_dispatch import (
    HORIZON,
    TOP_K,
    MIN_SEP,
    has_future_reveal,
    is_cross_burst,
    learned_choice_factory,
    nearest_choice,
    rollout_tour,
)
from scripts.train_arrival_value import evaluate
from scripts.train_burst_value import (
    FEATURE_NAMES,
    RADIUS,
    burst_features,
    diverse_candidates,
    pad_groups,
    train_early_stop,
)

MAX_ACTIONS = TOP_K + 1


def wait_features(inst: dict, state: dict) -> np.ndarray:
    horizon = float(inst["horizon"])
    capacity = float(inst["capacity"])
    current = int(state["current"])
    time = float(state["time"])
    n_customers = inst["demands"].shape[0] - 1
    known = inst["reveal"] <= time + 1e-9
    known[0] = False
    near = known & (inst["dist"][current] <= RADIUS)
    unvisited = ~state["visited"]
    unvisited[0] = False
    near_open = near & unvisited
    if near.any():
        rel = inst["reveal"][near]
        mean_reveal = float(rel.mean()) / horizon
        max_reveal = float(rel.max()) / horizon
    else:
        mean_reveal = max_reveal = 0.0
    open_known = np.flatnonzero(known & unvisited)
    nearest_other = (
        float(inst["dist"][current, open_known].min()) if open_known.size else 1.0
    )
    future = [
        float(inst["reveal"][node])
        for node in range(1, inst["demands"].shape[0])
        if not state["visited"][node] and float(inst["reveal"][node]) > time + 1e-9
    ]
    next_reveal = (min(future) / horizon) if future else (time / horizon)
    n_unknown = int((unvisited & ~known).sum())
    return np.array(
        [
            0.0,
            float(near.sum()),
            float(near_open.sum()),
            next_reveal,
            mean_reveal,
            max_reveal,
            time / horizon,
            float(state["remaining_capacity"]) / capacity,
            float(inst["dist"][current, 0]),
            nearest_other,
            n_unknown / n_customers,
        ],
        dtype=np.float64,
    )


from scripts.burst_scoring import score_actions  # noqa: E402

def collect_instance_wait(inst, top_k, n_futures, rng, min_sep):
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    horizon = float(inst["horizon"])
    groups = []
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            candidates = diverse_candidates(inst, state, top_k, min_sep)
            separated = (
                len(candidates) >= 2
                and float(inst["dist"][candidates[0], candidates[1]]) >= min_sep
            )
            if separated and float(state["time"]) < horizon - 1e-8:
                before = float(state["cost"])
                actions = [("go", n) for n in candidates]
                feats = []
                for action in candidates:
                    post = apply_action(inst, state, "go", action)
                    feats.append(burst_features(inst, post, float(post["cost"]) - before, action))
                if has_future_reveal(inst, state):
                    actions.append(("wait", 0))
                    feats.append(wait_features(inst, state))
                costs = score_actions(inst, state, actions, n_futures, rng)
                groups.append((np.stack(feats), costs))
            state = apply_action(
                inst, state, "go", rank_nearest(inst, state, feasible_customers(inst, state))[0]
            )
            if bool(state["visited"][1:].all()):
                break
            continue
        state = apply_action(inst, state, kind, node)
    else:
        raise RuntimeError("instance exceeded event limit")
    return groups


def learned_wait_cross_factory(model, apply):
    def choose(inst: dict, state: dict, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        candidates = diverse_candidates(inst, state, TOP_K, MIN_SEP)
        before = float(state["cost"])
        feats = []
        meta = []
        for node in candidates:
            post = apply_action(inst, state, "go", node)
            feats.append(burst_features(inst, post, float(post["cost"]) - before, node))
            meta.append(("go", node))
        if has_future_reveal(inst, state):
            feats.append(wait_features(inst, state))
            meta.append(("wait", 0))
        features = np.stack(feats)[None, :, :]
        width = features.shape[1]
        dummy_y = np.zeros((1, width), dtype=np.float64)
        dummy_v = np.zeros((1, MAX_ACTIONS), dtype=bool)
        dummy_v[0, :width] = True
        padded = np.zeros((1, MAX_ACTIONS, len(FEATURE_NAMES)), dtype=np.float64)
        padded[0, :width] = features[0]
        scaled, _, _ = apply((padded, dummy_y, dummy_v))
        pred, _ = model.forward(scaled)
        index = int(np.argmin(pred[0, :width]))
        kind, node = meta[index]
        if kind == "wait":
            return ("wait", 0)
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def go_only_groups(groups):
    """Drop the optional wait slot so the go-only model sees width TOP_K."""
    out = []
    for feat, cost in groups:
        if feat.shape[0] > TOP_K:
            out.append((feat[:TOP_K], cost[:TOP_K]))
        else:
            out.append((feat, cost))
    return out


def split_episodes(episodes: int):
    if episodes <= 8:
        return list(range(0, 4)), list(range(4, 6)), [7]
    return list(range(0, 12)), list(range(12, 16)), list(range(16, 20))


def train_for_seed(seed: int, futures: int, episodes: int, use_instance_bank: bool = False):
    customers, horizon, capacity = 200, 40.0, 80.0
    train_eps, val_eps, test_eps = split_episodes(episodes)
    grouped = []
    if use_instance_bank:
        from scripts.burst_instances import load_or_create_bank

        instances = load_or_create_bank(seed, episodes)
        label_rng = random.Random(seed + 5000)
        for episode in range(episodes):
            groups = collect_instance_wait(instances[episode], TOP_K, futures, label_rng, MIN_SEP)
            grouped.append(groups)
    else:
        rng = random.Random(seed)
        instances = []
        for episode in range(episodes):
            inst = generate_burst_instance(customers, capacity, horizon, rng)
            instances.append(inst)
            groups = collect_instance_wait(inst, TOP_K, futures, rng, MIN_SEP)
            grouped.append(groups)
        print(f"seed {seed} episode {episode}: {len(groups)} wait+go decisions", flush=True)
    train = pad_groups([g for ep in train_eps for g in grouped[ep]], MAX_ACTIONS)
    val = pad_groups([g for ep in val_eps for g in grouped[ep]], MAX_ACTIONS)
    go_train = pad_groups(go_only_groups([g for ep in train_eps for g in grouped[ep]]), TOP_K)
    go_val = pad_groups(go_only_groups([g for ep in val_eps for g in grouped[ep]]), TOP_K)
    wait_model, wait_apply = train_early_stop(train, val, seed)
    go_model, go_apply = train_early_stop(go_train, go_val, seed + 10000)
    return instances, grouped, test_eps, wait_model, wait_apply, go_model, go_apply


def eval_seed(seed: int, futures: int, episodes: int, use_instance_bank: bool = True):
    instances, grouped, test_eps, wait_model, wait_apply, go_model, go_apply = train_for_seed(
        seed, futures, episodes, use_instance_bank=use_instance_bank
    )
    learn_go = learned_choice_factory(go_model, go_apply)
    learn_wait = learned_wait_cross_factory(wait_model, wait_apply)
    test_groups = [g for ep in test_eps for g in grouped[ep]]
    test = pad_groups(test_groups, MAX_ACTIONS)
    scaled, _, _ = wait_apply(test)
    pred, _ = wait_model.forward(scaled)
    one_step = evaluate(pred, test[1], test[2])

    tours = {}
    for name, choose, cross in (
        ("nearest", nearest_choice, None),
        ("learned_go", learn_go, None),
        ("learned_wait_cross", nearest_choice, learn_wait),
    ):
        costs = []
        waits = 0
        for episode in test_eps:
            cost, _goes, _diff, cross_n = rollout_tour(
                instances[episode], choose, choose_at_cross=cross, horizon=HORIZON
            )
            costs.append(cost)
            waits += cross_n if name == "learned_wait_cross" else 0
            print(f"seed {seed} {name} ep {episode}: {cost:.4f}", flush=True)
        tours[name] = {
            "mean_cost": float(np.mean(costs)),
            "costs": [float(c) for c in costs],
            "cross_burst_uses": waits if name == "learned_wait_cross" else None,
        }
    base = tours["nearest"]["mean_cost"]
    for name in ("learned_go", "learned_wait_cross"):
        tours[name]["gap_vs_nearest"] = base - tours[name]["mean_cost"]
    return {"seed": seed, "one_step": one_step, "tours": tours}


def main():
    probe = "--probe" in sys.argv
    seeds = (1234,) if probe else (1234, 1235, 1236)
    futures = 8 if probe else 20
    episodes = 8 if probe else 20
    per_seed = [eval_seed(seed, futures, episodes, use_instance_bank=True) for seed in seeds]
    test_eps = split_episodes(episodes)[2]
    report = {"seeds": list(seeds), "test_episodes": test_eps, "per_seed": per_seed}
    if len(per_seed) > 1:
        names = ("nearest", "learned_go", "learned_wait_cross")
        report["aggregate"] = {}
        for name in names:
            means = [row["tours"][name]["mean_cost"] for row in per_seed]
            entry = {
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means)),
            }
            if name != "nearest":
                gaps = [row["tours"][name]["gap_vs_nearest"] for row in per_seed]
                entry["mean_gap_vs_nearest"] = float(np.mean(gaps))
            report["aggregate"][name] = entry
    out = Path(ROOT) / "outputs" / "logs" / "burst_learned_wait.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
