"""Level A: FAROS with a learned arrival world model vs. the oracle sampler.

Customer locations are known a priori (as in the main benchmark); only
request times are uncertain. Per seed the world model is fit on training
days (episodes 0-11) and drives both the waiting rule and the imagined
futures on test days (16-19). The oracle variants sample from the true
wave windows. All variants score actions on common futures unless
``--independent`` is given. Writes outputs/logs/burst_world_model_a*.json.

  python scripts/run_world_model_benchmark.py                 # 10 seeds
  python scripts/run_world_model_benchmark.py --flat-prior    # ablation
  python scripts/run_world_model_benchmark.py --probe         # 1 seed
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import math
import random
import sys
import time as wall
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import rank_nearest
from models.arrival_world_model import ArrivalWorldModel
from scripts.burst_dispatch import HORIZON, has_future_reveal, is_cross_burst, nearest_choice, rollout_tour
from scripts.burst_instances import DEFAULT_EPISODES, load_or_create_bank
from scripts.run_burst_benchmark import (
    DEFAULT_SEEDS,
    SCENARIO_RNG_OFFSET,
    hybrid_factory_stats,
    lazy_factory_stats,
    scenario_cross_factory_stats,
)
from scripts.run_paper_baselines import IMMINENT_HORIZON, wait_first_cross_factory
from scripts.train_burst_wait import split_episodes

FUTURES = 20
WORKERS = 5
LOG = Path(ROOT) / "outputs" / "logs"
POLICIES = ("sampling", "wait_rule", "faros_go", "faros_all")


def wm_wait_factory(world: ArrivalWorldModel, diagnostics: dict | None = None):
    """Wait while the world model expects the newest wave to reveal soon."""

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
        if diagnostics is not None:
            diagnostics.setdefault("start_abs_error", []).extend(world.start_error(inst, time))
        next_t = world.expected_next_reveal(inst, anchor, time)
        if next_t is not None and next_t - time <= IMMINENT_HORIZON:
            return ("wait", 0)
        return ("go", rank_nearest(inst, state, feas)[0])

    return choose


def sign_test(base, method):
    """Paired comparison; positive mean_delta means ``method`` is cheaper than ``base``."""
    diff = [float(b) - float(m) for b, m in zip(base, method)]
    wins = int(sum(d > 1e-9 for d in diff))
    losses = int(sum(d < -1e-9 for d in diff))
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    rng = np.random.default_rng(42)
    arr = np.asarray(diff)
    boot = arr[rng.integers(0, arr.size, (2000, arr.size))].mean(axis=1)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(diff) - n,
        "p_two_sided": p,
        "mean_delta": float(arr.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def build_policies(seed, wait_cb, sampler, common, stats):
    def rng():
        return random.Random(seed + SCENARIO_RNG_OFFSET)

    return {
        "sampling": scenario_cross_factory_stats(rng(), FUTURES, stats["sampling"], sampler, common),
        "wait_rule": wait_cb,
        "faros_go": hybrid_factory_stats(wait_cb, rng(), FUTURES, stats["faros_go"], sampler, common),
        "faros_all": lazy_factory_stats(
            wait_cb,
            scenario_cross_factory_stats(rng(), FUTURES, stats["faros_all_msa"], sampler, common),
            stats["faros_all"],
        ),
    }


def run_seed(args):
    seed, flat, common = args
    train_eps, _val_eps, test_eps = split_episodes(DEFAULT_EPISODES)
    instances = load_or_create_bank(seed, DEFAULT_EPISODES)
    world = ArrivalWorldModel.fit([instances[ep] for ep in train_eps], flat_prior=flat)
    diagnostics: dict = {}
    row = {
        "seed": seed,
        "world_model": {"width": world.width, "fitted_waves": world.fitted_waves},
        "costs": {},
        "seconds": {},
        "scoring_stats": {},
    }
    groups = {
        "oracle": (wait_first_cross_factory(), None),
        "wm": (wm_wait_factory(world, diagnostics), world.imagine),
    }
    row["costs"]["nearest"] = [
        float(rollout_tour(instances[ep], nearest_choice, horizon=HORIZON)[0]) for ep in test_eps
    ]
    for prefix, (wait_cb, sampler) in groups.items():
        stats = {k: {} for k in ("sampling", "faros_go", "faros_all", "faros_all_msa")}
        for name, cross in build_policies(seed, wait_cb, sampler, common, stats).items():
            started = wall.perf_counter()
            run = []
            for ep in test_eps:
                cost, _, _, _ = rollout_tour(instances[ep], nearest_choice, choose_at_cross=cross, horizon=HORIZON)
                run.append(float(cost))
            row["costs"][f"{prefix}_{name}"] = run
            row["seconds"][f"{prefix}_{name}"] = wall.perf_counter() - started
            print(f"seed {seed} {prefix}_{name}: {np.mean(run):.4f}", flush=True)
        row["scoring_stats"][prefix] = stats
    errors = diagnostics.get("start_abs_error", [])
    row["start_abs_error_mean"] = float(np.mean(errors)) if errors else None
    return row


def main():
    probe = "--probe" in sys.argv
    flat = "--flat-prior" in sys.argv
    common = "--independent" not in sys.argv
    seeds = DEFAULT_SEEDS[:1] if probe else DEFAULT_SEEDS
    with Pool(min(WORKERS, len(seeds))) as pool:
        per_seed = pool.map(run_seed, [(seed, flat, common) for seed in seeds])
    names = ["nearest"] + [f"{p}_{k}" for p in ("oracle", "wm") for k in POLICIES]
    costs = {name: [c for row in per_seed for c in row["costs"][name]] for name in names}
    paired = {}
    for prefix in ("oracle", "wm"):
        for k in POLICIES:
            paired[f"{prefix}_{k} vs nearest"] = sign_test(costs["nearest"], costs[f"{prefix}_{k}"])
        for k in ("wait_rule", "faros_go", "faros_all"):
            paired[f"{prefix}_{k} vs {prefix}_sampling"] = sign_test(costs[f"{prefix}_sampling"], costs[f"{prefix}_{k}"])
        for k in ("faros_go", "faros_all"):
            paired[f"{prefix}_{k} vs {prefix}_wait_rule"] = sign_test(costs[f"{prefix}_wait_rule"], costs[f"{prefix}_{k}"])
        paired[f"{prefix}_faros_all vs {prefix}_faros_go"] = sign_test(costs[f"{prefix}_faros_go"], costs[f"{prefix}_faros_all"])
    for k in POLICIES:
        paired[f"wm_{k} vs oracle_{k}"] = sign_test(costs[f"oracle_{k}"], costs[f"wm_{k}"])
    out = {
        "level": "A",
        "common_random_numbers": common,
        "flat_prior": flat,
        "seeds": list(seeds),
        "train_episodes": split_episodes(DEFAULT_EPISODES)[0],
        "test_episodes": split_episodes(DEFAULT_EPISODES)[2],
        "futures": FUTURES,
        "mean_cost": {k: float(np.mean(v)) for k, v in costs.items()},
        "sd_cost": {k: float(np.std(v, ddof=1)) for k, v in costs.items()},
        "paired": paired,
        "per_seed": per_seed,
        "costs": costs,
    }
    suffix = ("_flat" if flat else "") + ("" if common else "_independent") + ("_probe" if probe else "")
    path = LOG / f"burst_world_model_a{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["mean_cost"], indent=2))


if __name__ == "__main__":
    main()
