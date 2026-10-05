"""Novel deployable policies: hybrid recourse + lazy scenario activation."""

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

from env.dynamic_cvrp import rank_nearest
from scripts.burst_instances import load_or_create_bank
from scripts.burst_dispatch import HORIZON, MIN_SEP, TOP_K, is_cross_burst, nearest_choice, rollout_tour
from scripts.burst_scoring import score_actions
from scripts.run_paper_baselines import (
    _cross_actions,
    scenario_cross_factory,
    wait_first_cross_factory,
)
from scripts.train_burst_wait import split_episodes


def hybrid_wait_msa_go_factory(wait_cb, scenario_rng, n_futures: int):
    """SS-VRPTW-CR-style split: wait-first timing; MSA only among go candidates."""

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        wf = wait_cb(inst, state, feas)
        if wf[0] == "wait":
            return ("wait", 0)
        all_actions, _ = _cross_actions(inst, state)
        go_actions = [a for a in all_actions if a[0] == "go"]
        if len(go_actions) <= 1:
            return ("go", rank_nearest(inst, state, feas)[0])
        costs = score_actions(inst, state, go_actions, n_futures, scenario_rng)
        kind, node = go_actions[int(np.argmin(costs))]
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def lazy_msa_factory(wait_cb, scenario_cb, scenario_rng, n_futures: int):
    """Run full MSA only when wait-first and nearest disagree on wait vs go."""

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        nearest = rank_nearest(inst, state, feas)[0]
        wf = wait_cb(inst, state, feas)
        if wf[0] == "wait":
            return wf
        if wf == ("go", nearest):
            return scenario_cb(inst, state, feas)
        return wf

    return choose


def eval_seed(seed: int, episodes: int, futures: int):
    _tr, _va, test_eps = split_episodes(episodes)
    s_rng = random.Random(seed + 7000)
    instances = load_or_create_bank(seed, episodes)
    wf = wait_first_cross_factory()
    msa = scenario_cross_factory(s_rng, futures)
    policies = {
        "nearest": (nearest_choice, None),
        "msa_full": (nearest_choice, msa),
        "wait_first": (nearest_choice, wf),
        "hybrid_wait_msa_go": (
            nearest_choice,
            hybrid_wait_msa_go_factory(wf, s_rng, futures),
        ),
        "lazy_msa": (nearest_choice, lazy_msa_factory(wf, msa, s_rng, futures)),
    }
    out = {}
    for name, (choose, cross) in policies.items():
        costs = []
        for ep in test_eps:
            cost, _, _, _ = rollout_tour(instances[ep], choose, choose_at_cross=cross, horizon=HORIZON)
            costs.append(float(cost))
            print(f"seed {seed} {name} ep {ep}: {cost:.4f}", flush=True)
        out[name] = {"mean_cost": float(np.mean(costs)), "costs": costs}
    base = out["nearest"]["mean_cost"]
    msa_base = out["msa_full"]["mean_cost"]
    for name in out:
        if name != "nearest":
            out[name]["gap_vs_nearest"] = base - out[name]["mean_cost"]
        if name not in ("nearest", "msa_full"):
            out[name]["gap_vs_msa_full"] = msa_base - out[name]["mean_cost"]
    return {"seed": seed, "policies": out}


def main():
    probe = "--probe" in sys.argv
    seeds = (1234,) if probe else (1234, 1235, 1236)
    episodes = 8 if probe else 20
    futures = 8 if probe else 20
    per_seed = [eval_seed(s, episodes, futures) for s in seeds]
    report = {
        "title": "Hybrid wait-first + selective MSA (amortized scenario recourse)",
        "novelty": (
            "Decouple timing (wait-first, zero rollouts) from routing (MSA over go-actions only). "
            "Lazy MSA runs full scenario only when wait-first deviates from nearest."
        ),
        "per_seed": per_seed,
    }
    if len(per_seed) > 1:
        names = list(per_seed[0]["policies"].keys())
        report["aggregate"] = {}
        for name in names:
            entry = {
                "mean_cost": float(np.mean([r["policies"][name]["mean_cost"] for r in per_seed])),
            }
            if name != "nearest":
                entry["mean_gap_vs_nearest"] = float(
                    np.mean([r["policies"][name]["gap_vs_nearest"] for r in per_seed])
                )
                entry["mean_gap_vs_msa_full"] = float(
                    np.mean([r["policies"][name].get("gap_vs_msa_full", 0) for r in per_seed])
                )
            report["aggregate"][name] = entry
    out = Path(ROOT) / "outputs" / "logs" / "novel_burst.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
