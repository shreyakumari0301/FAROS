"""Literature-aligned baselines for the burst dynamic CVRP paper line.

Policies (frozen nearest-feasible except at cross-burst forks):
  nearest          — myopic insertion
  msa_scenario     — Bent-style: argmin mean completion cost over M resampled futures
  wait_first       — Mitrovic/Laporte spirit: wait if a same-window customer is still hidden
  learned_wait     — small MLP trained on wait+go labels (optional, per-seed train)
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

from env.dynamic_cvrp import apply_action, generate_burst_instance, rank_nearest
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
from scripts.train_burst_value import diverse_candidates
from scripts.train_burst_wait import (
    learned_wait_cross_factory,
    split_episodes,
    train_for_seed,
)

WINDOW_TOL = 0.5
IMMINENT_HORIZON = 4.0


def _cross_actions(inst, state):
    candidates = diverse_candidates(inst, state, TOP_K, MIN_SEP)
    actions = [("go", int(n)) for n in candidates]
    if has_future_reveal(inst, state):
        actions.append(("wait", 0))
    return actions, candidates


def scenario_cross_factory(rng: random.Random, n_futures: int):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        actions, _ = _cross_actions(inst, state)
        costs = score_actions(inst, state, actions, n_futures, rng)
        kind, node = actions[int(np.argmin(costs))]
        if kind == "wait":
            return ("wait", 0)
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def wait_first_cross_factory():
    """Wait when the newest visible customer's wave still has unrevealed members.

    Uses only the a priori wave window, never true reveal times.
    """

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        if not has_future_reveal(inst, state):
            return ("go", rank_nearest(inst, state, feas)[0])
        time = float(state["time"])
        known = inst["reveal"] <= time + 1e-9
        unvisited = ~state["visited"]
        visible = np.flatnonzero(known & unvisited)
        visible = visible[visible != 0]
        if visible.size == 0:
            return ("go", rank_nearest(inst, state, feas)[0])
        anchor = int(visible[np.argmax(inst["reveal"][visible])])
        w_start = float(inst["window_start"][anchor])
        hidden = []
        for node in range(1, inst["demands"].shape[0]):
            if state["visited"][node] or known[node]:
                continue
            if abs(float(inst["window_start"][node]) - w_start) <= WINDOW_TOL:
                hidden.append(node)
        if hidden:
            # Non-anticipative: expected first reveal of k hidden same-wave customers.
            low = max(w_start, time)
            high = w_start + float(inst["window_width"][hidden[0]])
            next_t = low + max(high - low, 0.0) / (len(hidden) + 1)
            if next_t - time <= IMMINENT_HORIZON:
                return ("wait", 0)
        return ("go", rank_nearest(inst, state, feas)[0])

    return choose


def eval_policies(seed: int, episodes: int, futures: int, train_learned: bool):
    customers, horizon, capacity = 200, 40.0, 80.0
    _train_eps, _val_eps, test_eps = split_episodes(episodes)
    from scripts.burst_instances import load_or_create_bank

    instances = load_or_create_bank(seed, episodes)
    scenario_rng = random.Random(seed + 9000)
    policies = {
        "nearest": (nearest_choice, None),
        "msa_scenario": (nearest_choice, scenario_cross_factory(scenario_rng, futures)),
        "wait_first": (nearest_choice, wait_first_cross_factory()),
    }
    learned = None
    if train_learned:
        _inst, _grp, _te, wait_model, wait_apply, _go_m, _go_a = train_for_seed(
            seed, futures, episodes, use_instance_bank=True
        )
        policies["learned_wait_cross"] = (
            nearest_choice,
            learned_wait_cross_factory(wait_model, wait_apply),
        )
    rows = {}
    for name, (choose, cross) in policies.items():
        costs = []
        for ep in test_eps:
            cost, _, _, _ = rollout_tour(instances[ep], choose, choose_at_cross=cross, horizon=HORIZON)
            costs.append(cost)
            print(f"seed {seed} {name} ep {ep}: {cost:.4f}", flush=True)
        rows[name] = {
            "mean_cost": float(np.mean(costs)),
            "costs": [float(c) for c in costs],
        }
    base = rows["nearest"]["mean_cost"]
    for name in rows:
        if name != "nearest":
            rows[name]["gap_vs_nearest"] = base - rows[name]["mean_cost"]
    return {"seed": seed, "test_episodes": test_eps, "policies": rows}


def main():
    probe = "--probe" in sys.argv
    quick = "--quick" in sys.argv
    seeds = (1234,) if probe else (1234, 1235, 1236)
    episodes = 8 if probe else 20
    futures = 8 if (probe or quick) else 20
    train_learned = not quick and not probe and "--no-learned" not in sys.argv
    if quick:
        train_learned = "--no-learned" not in sys.argv
        seeds = (1234,)
    per_seed = [eval_policies(s, episodes, futures, train_learned) for s in seeds]
    report = {
        "seeds": list(seeds),
        "futures_per_decision": futures,
        "train_learned": train_learned,
        "per_seed": per_seed,
        "positioning": (
            "msa_scenario = Bent-style scenario consensus on resampled cluster reveals; "
            "wait_first = SS-VRPTW-CR / Mitrovic wait-first heuristic; "
            "learned_wait_cross = rollout-supervised MLP."
        ),
    }
    if len(per_seed) > 1:
        names = set(per_seed[0]["policies"].keys())
        report["aggregate"] = {}
        for name in sorted(names):
            means = [row["policies"][name]["mean_cost"] for row in per_seed]
            entry = {"mean_of_means": float(np.mean(means)), "std_of_means": float(np.std(means))}
            if name != "nearest":
                gaps = [row["policies"][name]["gap_vs_nearest"] for row in per_seed]
                entry["mean_gap_vs_nearest"] = float(np.mean(gaps))
            report["aggregate"][name] = entry
    out = Path(ROOT) / "outputs" / "logs" / "paper_baselines_burst.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
