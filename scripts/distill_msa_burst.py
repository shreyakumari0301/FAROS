"""Distill MSA scenario consensus into a fast cross-burst policy (novel deployable line).

Teacher: at each cross-burst stop, argmin mean completion cost over M resampled futures.
Student: MLP on visible per-action features only; one forward pass at deploy time.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import (
    apply_action,
    feasible_customers,
    generate_burst_instance,
    initial_state,
    policy_action,
    rank_nearest,
)
from scripts.burst_dispatch import (
    HORIZON,
    MIN_SEP,
    TOP_K,
    has_future_reveal,
    is_cross_burst,
    nearest_choice,
    rollout_tour,
)
from scripts.burst_scoring import score_actions
from scripts.run_paper_baselines import _cross_actions, scenario_cross_factory
from scripts.train_arrival_value import MLP, batch_loss
from scripts.train_burst_value import burst_features, diverse_candidates
from scripts.train_burst_wait import split_episodes, wait_features

MAX_ACTIONS = TOP_K + 1
FEAT_DIM = 11


def action_feature_matrix(inst, state):
    before = float(state["cost"])
    actions, _ = _cross_actions(inst, state)
    feats = []
    for kind, node in actions:
        if kind == "wait":
            feats.append(wait_features(inst, state))
        else:
            post = apply_action(inst, state, "go", int(node))
            feats.append(burst_features(inst, post, float(post["cost"]) - before, int(node)))
    return actions, np.stack(feats)


def collect_teacher_on_instance(inst, teacher_rng, n_futures, min_sep):
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    horizon = float(inst["horizon"])
    groups = []
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        if kind == "go":
            if (
                is_cross_burst(inst, state, horizon)
                and float(state["time"]) < horizon - 1e-8
            ):
                actions, feats = action_feature_matrix(inst, state)
                costs = score_actions(inst, state, actions, n_futures, teacher_rng)
                groups.append((feats, costs, int(np.argmin(costs)), actions))
            state = apply_action(
                inst, state, "go", rank_nearest(inst, state, feasible_customers(inst, state))[0]
            )
            if bool(state["visited"][1:].all()):
                break
            continue
        state = apply_action(inst, state, kind, node)
    return groups


def pad_teacher_groups(groups, width):
    n = len(groups)
    x = np.zeros((n, width, FEAT_DIM), dtype=np.float64)
    y = np.zeros((n, width), dtype=np.float64)
    v = np.zeros((n, width), dtype=bool)
    label = np.zeros(n, dtype=np.int64)
    for i, (feat, cost, best, _actions) in enumerate(groups):
        k = feat.shape[0]
        x[i, :k] = feat
        y[i, :k] = cost
        v[i, :k] = True
        label[i] = best
    return x, y, v, label


def standardize_train_val(train, val):
    x, y, v, lab = train
    flat = x[v]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    y_mean = float(y[v].mean())
    y_std = float(y[v].std()) or 1.0

    def apply(pack):
        a, b, c, d = pack
        return (a - mean) / std, (b - y_mean) / y_std, c, d

    return apply((x, y, v, lab)), apply(val), (mean, std, y_mean, y_std)


def train_distilled(train, val, seed, epochs=80):
    x, y, v, lab = train
    vx, vy, vv, vlab = val
    model = MLP(FEAT_DIM, 64, random.Random(seed))
    order = list(range(x.shape[0]))
    rng = random.Random(seed + 1)
    best_acc = -1.0
    best = [p.copy() for p in model.params]
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            idx = np.array(order[start : start + 64])
            pred, cache = model.forward(x[idx])
            # Ranking on scaled costs
            _, diff, better = batch_loss(pred, y[idx], v[idx])
            grad = np.where(np.abs(diff) < 1.0, diff, np.sign(diff))
            grad = grad / max(float(v[idx].sum()), 1.0)
            if better.any():
                logit = pred[:, None, :] - pred[:, :, None]
                d_logit = -1.0 / (1.0 + np.exp(np.clip(logit, -40.0, 40.0)))
                d_logit = np.where(better, d_logit, 0.0)
                pair_count = better.sum(axis=(1, 2)).clip(min=1).astype(np.float64)
                d_logit /= pair_count[:, None, None]
                d_logit /= pred.shape[0]
                grad = grad + d_logit.sum(axis=1) - d_logit.sum(axis=2)
            # Classification: match teacher argmin on valid slots
            masked_pred = np.where(v[idx], pred, -1e9)
            rows = np.arange(pred.shape[0])
            chosen = masked_pred[rows, lab[idx]]
            logsum = np.log(np.exp(masked_pred - masked_pred.max(axis=1, keepdims=True)).sum(axis=1))
            stable = masked_pred - masked_pred.max(axis=1, keepdims=True)
            logsum = np.log(np.exp(stable).sum(axis=1)) + masked_pred.max(axis=1)
            ce = chosen - logsum
            ce_grad = np.exp(stable) / np.exp(stable).sum(axis=1, keepdims=True)
            ce_grad[rows, lab[idx]] -= 1.0
            ce_grad = np.where(v[idx], ce_grad, 0.0) / max(len(idx), 1)
            grad = grad + 2.0 * ce_grad
            model.adam(model.backward(cache, grad), 1e-3, 1e-4)
        vp, _ = model.forward(vx)
        acc = float(np.mean(np.argmax(np.where(vv, vp, -np.inf), axis=1) == vlab))
        if acc > best_acc:
            best_acc = acc
            best = [p.copy() for p in model.params]
    for p, s in zip(model.params, best):
        p[...] = s
    return model, best_acc


def distilled_factory(model, stats):
    mean, std, _y_mean, _y_std = stats

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        actions, feats = action_feature_matrix(inst, state)
        width = feats.shape[0]
        padded = np.zeros((1, MAX_ACTIONS, FEAT_DIM))
        padded[0, :width] = (feats - mean) / std
        pred, _ = model.forward(padded)
        pick = int(np.argmin(pred[0, :width]))
        kind, node = actions[pick]
        if kind == "wait":
            return ("wait", 0)
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def rollout_with_stats(inst, choose_go, cross_cb, count_mode=None, m_futures=20, rng=None):
    cross = 0
    rollouts = 0
    forwards = 0
    t0 = time.perf_counter()
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        if bool(state["visited"][1:].all()):
            from scripts.burst_dispatch import finish_tour

            return (
                float(finish_tour(inst, state)["cost"]),
                cross,
                rollouts,
                forwards,
                time.perf_counter() - t0,
            )
        feas = feasible_customers(inst, state)
        if feas:
            if cross_cb is not None and is_cross_burst(inst, state, HORIZON):
                cross += 1
                if count_mode == "msa":
                    actions, _ = _cross_actions(inst, state)
                    rollouts += m_futures * len(actions)
                elif count_mode == "student":
                    forwards += 1
                kind, node = cross_cb(inst, state, feas)
                if kind == "wait":
                    state = apply_action(inst, state, "wait", 0)
                    continue
                node = int(node)
                if node not in feas:
                    node = rank_nearest(inst, state, feas)[0]
            else:
                node = int(choose_go(inst, state, feas))
            state = apply_action(inst, state, "go", node)
            continue
        kind, node = policy_action(inst, state)
        if kind == "done":
            from scripts.burst_dispatch import finish_tour

            return (
                float(finish_tour(inst, state)["cost"]),
                cross,
                rollouts,
                forwards,
                time.perf_counter() - t0,
            )
        state = apply_action(inst, state, kind, node)
    raise RuntimeError("rollout exceeded events")


def eval_seed(seed, episodes, n_futures, probe=False):
    customers, horizon, capacity = 200, 40.0, 80.0
    train_eps, val_eps, test_eps = split_episodes(episodes)
    rng = random.Random(seed)
    teacher_rng = random.Random(seed + 5000)
    instances = [generate_burst_instance(customers, capacity, horizon, rng) for _ in range(episodes)]
    train_g = []
    val_g = []
    for ep in train_eps:
        train_g.extend(collect_teacher_on_instance(instances[ep], teacher_rng, n_futures, MIN_SEP))
    for ep in val_eps:
        val_g.extend(collect_teacher_on_instance(instances[ep], teacher_rng, n_futures, MIN_SEP))
    train = pad_teacher_groups(train_g, MAX_ACTIONS)
    val = pad_teacher_groups(val_g, MAX_ACTIONS)
    tr, va, stats = standardize_train_val(train, val)
    model, val_acc = train_distilled(tr, va, seed)
    distilled_cb = distilled_factory(model, stats)
    scenario_rng = random.Random(seed + 9000)
    scenario_cb = scenario_cross_factory(scenario_rng, n_futures)
    rows = {}
    for name, cross, mode in (
        ("nearest", None, None),
        ("msa_teacher", scenario_cb, "msa"),
        ("distilled_student", distilled_cb, "student"),
    ):
        costs, crosses, roll, fwd, secs = [], [], [], [], []
        for ep in test_eps:
            if name == "nearest":
                cost, c, r, f, s = rollout_with_stats(instances[ep], nearest_choice, None)
            else:
                cost, c, r, f, s = rollout_with_stats(
                    instances[ep],
                    nearest_choice,
                    cross_cb=cross,
                    count_mode=mode,
                    m_futures=n_futures,
                    rng=scenario_rng,
                )
            costs.append(cost)
            crosses.append(c)
            roll.append(r)
            fwd.append(f)
            secs.append(s)
            print(f"seed {seed} {name} ep {ep}: {cost:.4f}", flush=True)
        rows[name] = {
            "mean_cost": float(np.mean(costs)),
            "costs": costs,
            "mean_cross_burst": float(np.mean(crosses)),
            "mean_scenario_rollouts": float(np.mean(roll)),
            "mean_student_forwards": float(np.mean(fwd)),
            "mean_seconds": float(np.mean(secs)),
        }
    base = rows["nearest"]["mean_cost"]
    for name in ("msa_teacher", "distilled_student"):
        rows[name]["gap_vs_nearest"] = base - rows[name]["mean_cost"]
        rows[name]["gap_vs_msa"] = rows["msa_teacher"]["mean_cost"] - rows[name]["mean_cost"]
    rows["distilled_student"]["val_teacher_top1"] = val_acc
    return {"seed": seed, "test_episodes": test_eps, "teacher_futures": n_futures, "policies": rows}


def main():
    probe = "--probe" in sys.argv
    seeds = (1234,) if probe else (1234, 1235, 1236)
    episodes = 8 if probe else 20
    futures = 8 if probe else 20
    per_seed = [eval_seed(s, episodes, futures, probe) for s in seeds]
    report = {
        "method": "Amortized MSA distillation at cross-burst forks",
        "novelty_note": (
            "Deploy-time student matches Bent-style scenario teacher using only "
            "visible features; no online Monte Carlo rollouts."
        ),
        "seeds": list(seeds),
        "per_seed": per_seed,
    }
    if len(per_seed) > 1:
        report["aggregate"] = {}
        for name in ("nearest", "msa_teacher", "distilled_student"):
            report["aggregate"][name] = {
                "mean_tour_cost": float(
                    np.mean([r["policies"][name]["mean_cost"] for r in per_seed])
                ),
                "mean_gap_vs_nearest": float(
                    np.mean([r["policies"][name].get("gap_vs_nearest", 0) for r in per_seed])
                ),
            }
        report["aggregate"]["distilled_student"]["mean_gap_vs_msa"] = float(
            np.mean([r["policies"]["distilled_student"]["gap_vs_msa"] for r in per_seed])
        )
        report["aggregate"]["msa_teacher"]["mean_scenario_rollouts"] = float(
            np.mean([r["policies"]["msa_teacher"]["mean_scenario_rollouts"] for r in per_seed])
        )
        report["aggregate"]["distilled_student"]["mean_student_forwards"] = float(
            np.mean([r["policies"]["distilled_student"]["mean_student_forwards"] for r in per_seed])
        )
    out = Path(ROOT) / "outputs" / "logs" / "distill_msa_burst.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
