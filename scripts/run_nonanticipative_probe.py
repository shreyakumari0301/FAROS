"""Re-run timing-screen policies with a non-anticipative wait-first rule.

The original wait-first rule reads the true reveal time of hidden same-wave
customers. This variant uses only the a priori wave window: with k hidden
customers of one wave, the expected first reveal is low + span / (k + 1).
Writes outputs/logs/burst_nonanticipative_probe.json; does not touch the
main benchmark log.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import rank_nearest
from scripts.burst_dispatch import HORIZON, has_future_reveal, is_cross_burst, nearest_choice, rollout_tour
from scripts.burst_instances import DEFAULT_EPISODES, load_or_create_bank
from scripts.run_burst_benchmark import (
    DEFAULT_SEEDS,
    SCENARIO_RNG_OFFSET,
    hybrid_factory_stats,
    lazy_factory_stats,
    scenario_cross_factory_stats,
)
from scripts.run_paper_baselines import IMMINENT_HORIZON, WINDOW_TOL
from scripts.train_burst_wait import split_episodes

FUTURES = 20
LOG = Path(ROOT) / "outputs" / "logs"


def wait_first_expected_factory():
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON) or not has_future_reveal(inst, state):
            return ("go", rank_nearest(inst, state, feas)[0])
        time = float(state["time"])
        known = inst["reveal"] <= time + 1e-9
        visible = np.flatnonzero(known & ~state["visited"])
        visible = visible[visible != 0]
        if visible.size == 0:
            return ("go", rank_nearest(inst, state, feas)[0])
        anchor = int(visible[np.argmax(inst["reveal"][visible])])
        w_start = float(inst["window_start"][anchor])
        hidden = [
            node
            for node in range(1, inst["demands"].shape[0])
            if not state["visited"][node]
            and not known[node]
            and abs(float(inst["window_start"][node]) - w_start) <= WINDOW_TOL
        ]
        if hidden:
            low = max(w_start, time)
            high = w_start + float(inst["window_width"][hidden[0]])
            expected_next = low + max(high - low, 0.0) / (len(hidden) + 1)
            if expected_next - time <= IMMINENT_HORIZON:
                return ("wait", 0)
        return ("go", rank_nearest(inst, state, feas)[0])

    return choose


def sign_test(base, method):
    diff = [b - m for b, m in zip(base, method)]
    wins = sum(d > 1e-9 for d in diff)
    losses = sum(d < -1e-9 for d in diff)
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"wins": wins, "losses": losses, "ties": len(diff) - n, "p_two_sided": p, "mean_delta": float(np.mean(diff))}


def main():
    _tr, _va, test_eps = split_episodes(DEFAULT_EPISODES)
    full = json.loads((LOG / "burst_benchmark_full.json").read_text())
    saved = {row["seed"]: row["policies"] for row in full["per_seed"]}
    costs = {name: [] for name in ("nearest", "msa_scenario", "wait_first_na", "hybrid_na", "lazy_na")}
    calls = {"hybrid_na": [], "lazy_na": []}
    for seed in DEFAULT_SEEDS:
        instances = load_or_create_bank(seed, DEFAULT_EPISODES)
        wf = wait_first_expected_factory()
        hybrid_stats, lazy_stats, msa_stats = {}, {}, {}
        policies = {
            "wait_first_na": wf,
            "hybrid_na": hybrid_factory_stats(
                wf, random.Random(seed + SCENARIO_RNG_OFFSET), FUTURES, hybrid_stats
            ),
            "lazy_na": lazy_factory_stats(
                wf,
                scenario_cross_factory_stats(random.Random(seed + SCENARIO_RNG_OFFSET), FUTURES, msa_stats),
                lazy_stats,
            ),
        }
        for name, cross in policies.items():
            for ep in test_eps:
                cost, _, _, _ = rollout_tour(instances[ep], nearest_choice, choose_at_cross=cross, horizon=HORIZON)
                costs[name].append(float(cost))
                print(f"seed {seed} {name} ep {ep}: {cost:.4f}", flush=True)
        calls["hybrid_na"].append(hybrid_stats.get("msa_calls", 0))
        calls["lazy_na"].append(lazy_stats.get("full_msa_invocations", 0))
        costs["nearest"] += saved[seed]["nearest"]["costs"]
        costs["msa_scenario"] += saved[seed]["msa_scenario"]["costs"]

    out = {
        "seeds": list(DEFAULT_SEEDS),
        "test_episodes": list(test_eps),
        "futures": FUTURES,
        "mean_cost": {k: float(np.mean(v)) for k, v in costs.items()},
        "mean_msa_calls_per_seed": {k: float(np.mean(v)) for k, v in calls.items()},
        "vs_nearest": {k: sign_test(costs["nearest"], costs[k]) for k in ("wait_first_na", "hybrid_na", "lazy_na")},
        "vs_msa": {k: sign_test(costs["msa_scenario"], costs[k]) for k in ("wait_first_na", "hybrid_na", "lazy_na")},
        "costs": costs,
    }
    path = LOG / "burst_nonanticipative_probe.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("mean_cost", "mean_msa_calls_per_seed", "vs_nearest", "vs_msa")}, indent=2))


if __name__ == "__main__":
    main()
