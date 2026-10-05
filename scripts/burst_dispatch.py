"""Shared burst-instance dispatchers and tour rollouts."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from env.dynamic_cvrp import (
    apply_action,
    feasible_customers,
    generate_burst_instance,
    initial_state,
    policy_action,
    rank_nearest,
)
from scripts.train_burst_value import (
    burst_features,
    collect_instance,
    diverse_candidates,
    pad_groups,
    train_early_stop,
)

TOP_K = 4
MIN_SEP = 0.25
HORIZON = 40.0
ChooseGo = Callable[[dict, dict, List[int]], int]


def _copy_state(snap: dict) -> dict:
    return {
        "visited": snap["visited"].copy(),
        "current": snap["current"],
        "time": snap["time"],
        "remaining_capacity": snap["remaining_capacity"],
        "cost": snap["cost"],
    }


def finish_tour(inst: dict, state: dict) -> dict:
    if bool(state["visited"][1:].all()) and int(state["current"]) != 0:
        state = apply_action(inst, state, "depot", 0)
    if not bool(state["visited"][1:].all()):
        raise RuntimeError("tour ended with unserved customers")
    return state


def has_future_reveal(inst: dict, state: dict) -> bool:
    time = float(state["time"])
    for node in range(1, inst["demands"].shape[0]):
        if state["visited"][node]:
            continue
        if float(inst["reveal"][node]) > time + 1e-9:
            return True
    return False


def is_cross_burst(inst: dict, state: dict, horizon: float) -> bool:
    if float(state["time"]) >= horizon - 1e-8:
        return False
    candidates = diverse_candidates(inst, state, TOP_K, MIN_SEP)
    if len(candidates) < 2:
        return False
    return float(inst["dist"][candidates[0], candidates[1]]) >= MIN_SEP


def complete_from(inst: dict, state: dict, choose_go: ChooseGo) -> float:
    state = {
        "visited": state["visited"].copy(),
        "current": state["current"],
        "time": state["time"],
        "remaining_capacity": state["remaining_capacity"],
        "cost": state["cost"],
    }
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        if bool(state["visited"][1:].all()):
            return float(finish_tour(inst, state)["cost"])
        feas = feasible_customers(inst, state)
        if feas:
            node = int(choose_go(inst, state, feas))
            if node not in feas:
                raise RuntimeError(f"infeasible go to {node}")
            state = apply_action(inst, state, "go", node)
            continue
        kind, node = policy_action(inst, state)
        if kind == "done":
            return float(finish_tour(inst, state)["cost"])
        if kind == "go":
            raise RuntimeError("unexpected go without feasible customers")
        state = apply_action(inst, state, kind, node)
    raise RuntimeError("complete_from exceeded event limit")


def rollout_tour(
    inst: dict,
    choose_go: ChooseGo,
    choose_at_cross: Optional[Callable[[dict, dict, List[int]], Tuple[str, int]]] = None,
    horizon: float = 40.0,
) -> Tuple[float, int, int, int]:
    """Full tour from depot. choose_at_cross may return ('wait', 0) or ('go', node)."""
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    goes = 0
    differs = 0
    cross = 0
    for _ in range(inst["demands"].shape[0] * 4 + 8):
        if bool(state["visited"][1:].all()):
            return float(finish_tour(inst, state)["cost"]), goes, differs, cross
        feas = feasible_customers(inst, state)
        if feas:
            nearest = rank_nearest(inst, state, feas)[0]
            if choose_at_cross is not None and is_cross_burst(inst, state, horizon):
                cross += 1
                kind, node = choose_at_cross(inst, state, feas)
                if kind == "wait":
                    if not has_future_reveal(inst, state):
                        node = nearest
                    else:
                        state = apply_action(inst, state, "wait", 0)
                        continue
                else:
                    node = int(node)
                    if node not in feas:
                        node = nearest
            else:
                node = int(choose_go(inst, state, feas))
            differs += int(node != nearest)
            goes += 1
            state = apply_action(inst, state, "go", node)
            continue
        kind, node = policy_action(inst, state)
        if kind == "done":
            return float(finish_tour(inst, state)["cost"]), goes, differs, cross
        if kind == "go":
            raise RuntimeError("unexpected go without feasible customers")
        state = apply_action(inst, state, kind, node)
    raise RuntimeError("rollout_tour exceeded event limit")


def nearest_choice(inst: dict, state: dict, feas: List[int]) -> int:
    return rank_nearest(inst, state, feas)[0]


def density_choice(inst: dict, state: dict, feas: List[int]) -> int:
    before = float(state["cost"])
    best_node = feas[0]
    best_key = (-np.inf, -np.inf)
    for node in feas:
        post = apply_action(inst, state, "go", node)
        feat = burst_features(inst, post, float(post["cost"]) - before, node)
        score = float(feat[2] - feat[10])
        closeness = -float(inst["dist"][int(state["current"]), node])
        key = (score, closeness)
        if key > best_key:
            best_key = key
            best_node = node
    return best_node


def learned_choice_factory(model, apply):
    def choose(inst: dict, state: dict, feas: List[int]) -> int:
        candidates = diverse_candidates(inst, state, TOP_K, MIN_SEP)
        separated = (
            len(candidates) >= 2
            and float(inst["dist"][candidates[0], candidates[1]]) >= MIN_SEP
        )
        if not separated:
            return rank_nearest(inst, state, feas)[0]
        before = float(state["cost"])
        feats = []
        for node in candidates:
            post = apply_action(inst, state, "go", node)
            feats.append(burst_features(inst, post, float(post["cost"]) - before, node))
        features = np.stack(feats)[None, :, :]
        dummy_y = np.zeros(features.shape[:2], dtype=np.float64)
        dummy_v = np.ones(features.shape[:2], dtype=bool)
        scaled, _, _ = apply((features, dummy_y, dummy_v))
        pred, _ = model.forward(scaled)
        pick = candidates[int(np.argmin(pred[0, : len(candidates)]))]
        return pick if pick in feas else rank_nearest(inst, state, feas)[0]

    return choose


def oracle_cross_burst_factory(horizon: float = 40.0):
    def choose(inst: dict, state: dict, feas: List[int]) -> Tuple[str, int]:
        candidates = diverse_candidates(inst, state, TOP_K, MIN_SEP)
        options: List[Tuple[str, int]] = [("go", node) for node in candidates]
        if has_future_reveal(inst, state):
            options.append(("wait", 0))
        best_cost = np.inf
        best = ("go", rank_nearest(inst, state, feas)[0])
        for kind, node in options:
            if kind == "wait":
                post = apply_action(inst, state, "wait", 0)
            else:
                post = apply_action(inst, state, "go", int(node))
            cost = complete_from(inst, post, nearest_choice)
            if cost < best_cost:
                best_cost = cost
                best = (kind, int(node) if kind == "go" else 0)
        return best

    return choose


def collect_cross_burst_states(
    inst: dict, horizon: float = 40.0
) -> List[dict]:
    """States visited by nearest tour where cross-burst choice matters."""
    snapshots = []
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))

    def record():
        if is_cross_burst(inst, state, horizon):
            snapshots.append(
                {
                    "visited": state["visited"].copy(),
                    "current": state["current"],
                    "time": state["time"],
                    "remaining_capacity": state["remaining_capacity"],
                    "cost": state["cost"],
                }
            )

    for _ in range(inst["demands"].shape[0] * 4 + 8):
        if bool(state["visited"][1:].all()):
            break
        feas = feasible_customers(inst, state)
        if feas:
            record()
            state = apply_action(inst, state, "go", rank_nearest(inst, state, feas)[0])
            continue
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        state = apply_action(inst, state, kind, node)
    return snapshots


def first_action_costs(inst: dict, snap: dict, model, apply, horizon: float = HORIZON) -> Dict[str, float]:
    learn = learned_choice_factory(model, apply)
    base = _copy_state(snap)
    feas = feasible_customers(inst, base)
    nearest = rank_nearest(inst, base, feas)[0]
    out = {
        "nearest": complete_from(
            inst, apply_action(inst, _copy_state(snap), "go", nearest), nearest_choice
        ),
    }
    out["learned"] = complete_from(
        inst,
        apply_action(inst, _copy_state(snap), "go", learn(inst, base, feas)),
        nearest_choice,
    )
    if has_future_reveal(inst, base):
        out["wait"] = complete_from(
            inst, apply_action(inst, _copy_state(snap), "wait", 0), nearest_choice
        )
    candidates = diverse_candidates(inst, base, TOP_K, MIN_SEP)
    options: List[Tuple[str, int]] = [("go", n) for n in candidates]
    if has_future_reveal(inst, base):
        options.append(("wait", 0))
    best = np.inf
    oracle_pick = "nearest"
    for kind, node in options:
        state = _copy_state(snap)
        if kind == "wait":
            post = apply_action(inst, state, "wait", 0)
            label = "wait"
        else:
            post = apply_action(inst, state, "go", int(node))
            label = f"go_{node}"
        cost = complete_from(inst, post, nearest_choice)
        if cost < best:
            best = cost
            oracle_pick = label
    out["oracle"] = best
    out["oracle_pick"] = oracle_pick
    return out


def train_burst_model(seed: int, futures: int = 20, episodes: int = 20):
    customers, horizon, capacity = 200, 40.0, 80.0
    rng = random.Random(seed)
    grouped = []
    instances = []
    for episode in range(episodes):
        inst = generate_burst_instance(customers, capacity, horizon, rng)
        instances.append(inst)
        grouped.append(collect_instance(inst, TOP_K, futures, rng, MIN_SEP))
    train = pad_groups([g for episode in grouped[:12] for g in episode], TOP_K)
    val = pad_groups([g for episode in grouped[12:16] for g in episode], TOP_K)
    model, apply = train_early_stop(train, val, seed)
    return instances, model, apply
