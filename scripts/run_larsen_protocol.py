"""External validation: the Larsen et al. (2002) partially dynamic protocol.

Single vehicle, uniform customers, n_adv ~ U{0..E} advance requests and
Poisson immediate requests over an 8-hour day, route length objective,
no capacity, no deadline. E{n}=40 and 15 test instances per degree of
dynamism in {0, 5, ..., 100}% (315 per region, the instance count Bent and
Van Hentenryck (2003) used on this model); regions 10x10 km (horizon 32)
and 40x40 km (horizon 8). Service times are not modelled.

Larsen's policies serve requests without strategic idling, so every
policy here only chooses which visible customer to serve next:
nearest neighbour and first-come-first-served (Larsen's heuristics),
rollout over known customers, FAROS-Go with the learned Poisson world
model, and FAROS-Go with the realized future (clairvoyant bound).
Writes outputs/logs/larsen_protocol.json.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import generate_larsen_instance
from models.poisson_world_model import PoissonWorldModel
from scripts.deadline_protocol import ClairvoyantWorld, sampling_policy, simulate
from scripts.run_world_model_b import BlindWorld
from scripts.run_world_model_benchmark import sign_test

NO_DEADLINE = float("inf")

EXPECTED_TOTAL = 40
PER_LEVEL = 15
LEVELS = tuple(range(0, 101, 5))
TRAIN_DAYS = 200
REGIONS = {"M1_10km": 32.0, "M2_40km": 8.0}
BANK = Path(ROOT) / "data" / "processed" / "larsen_v1"
LOG = Path(ROOT) / "outputs" / "logs"


def stratified_bank(rng: random.Random, horizon: float):
    counts = {level: 0 for level in LEVELS}
    bank = []
    while any(c < PER_LEVEL for c in counts.values()):
        inst = generate_larsen_instance(rng, EXPECTED_TOTAL, horizon)
        n = inst["demands"].shape[0] - 1
        level = round(100 * inst["dod"])
        if n < 2 or level not in counts or abs(100 * inst["dod"] - level) > 1e-9:
            continue
        if counts[level] < PER_LEVEL:
            counts[level] += 1
            inst["dod_level"] = level
            bank.append(inst)
    return sorted(bank, key=lambda inst: inst["dod_level"])


def load_banks(region: str, horizon: float):
    path = BANK / f"{region}.pkl"
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    train_rng = random.Random(f"{region}-train")
    train = [generate_larsen_instance(train_rng, EXPECTED_TOTAL, horizon) for _ in range(TRAIN_DAYS)]
    test = stratified_bank(random.Random(f"{region}-test"), horizon)
    BANK.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump((train, test), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train, test


def summarize(costs, levels):
    base = np.array(costs["nearest"])
    out = {}
    for name, values in costs.items():
        values = np.array(values)
        gain = 100.0 * (base - values) / base
        out[name] = {
            "mean_cost": float(values.mean()),
            "mean_gain_pct_vs_nearest": float(gain.mean()),
            "gain_pct_by_band": {
                band: float(gain[(levels >= lo) & (levels <= hi)].mean())
                for band, (lo, hi) in {"dod_0_30": (0, 30), "dod_35_65": (35, 65), "dod_70_100": (70, 100)}.items()
            },
        }
        if name != "nearest":
            out[name]["vs_nearest"] = sign_test(list(base), list(values))
    out["faros_go vs rollout_known"] = sign_test(costs["rollout_known"], costs["faros_go"])
    out["faros_go_clairvoyant vs faros_go"] = sign_test(costs["faros_go"], costs["faros_go_clairvoyant"])
    return out


def fcfs_policy(inst, state, feas):
    """First come, first served; requests with equal times (advance requests) go nearest first."""
    current = int(state["current"])
    return ("go", min(feas, key=lambda node: (float(inst["reveal"][node]), float(inst["dist"][current, node]))))


def dynamism(inst):
    """Degree of dynamism and Larsen's effective degree of dynamism."""
    reveal = inst["reveal"][1:]
    return float((reveal > 1e-9).mean()), float((reveal / inst["horizon"]).mean())


def run_region(args):
    region, horizon = args
    train, test = load_banks(region, horizon)
    world = PoissonWorldModel.fit(train)
    policies = {
        "nearest": None,
        "fcfs": fcfs_policy,
        "rollout_known": sampling_policy(BlindWorld(world), random.Random(1), {}, 1, NO_DEADLINE, go_only=True),
        "faros_go": sampling_policy(world, random.Random(2), {}, t_return=NO_DEADLINE, go_only=True),
        "faros_go_clairvoyant": sampling_policy(
            ClairvoyantWorld(world), random.Random(3), {}, 1, NO_DEADLINE, go_only=True
        ),
    }
    costs = {name: [] for name in policies}
    response = {name: [] for name in policies}
    for index, inst in enumerate(test):
        for name, policy in policies.items():
            result = simulate(inst, policy, NO_DEADLINE)
            if result["waits"]:
                raise RuntimeError(f"{name} idled strategically; outside the Larsen protocol")
            costs[name].append(result["cost"])
            response[name].append(result["mean_response"])
        if index % 21 == 0:
            print(region, index, {k: round(v[-1], 3) for k, v in costs.items()}, flush=True)
    levels = np.array([inst["dod_level"] for inst in test])
    dod, edod = zip(*(dynamism(inst) for inst in test))
    return region, {
        "horizon": horizon,
        "world_model": {"expected_total": world.expected_total},
        "dod_mean": float(np.mean(dod)),
        "edod_mean": float(np.mean(edod)),
        "summary": summarize(costs, levels),
        "mean_response": {k: float(np.mean(v)) for k, v in response.items()},
        "dod_levels": levels.tolist(),
        "costs": costs,
    }


def main():
    report = {"expected_total": EXPECTED_TOTAL, "per_level": PER_LEVEL, "levels": list(LEVELS), "regions": {}}
    with Pool(len(REGIONS)) as pool:
        for region, result in pool.map(run_region, list(REGIONS.items())):
            report["regions"][region] = result
            print(region, json.dumps(result["summary"], indent=1), flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "larsen_protocol.json").write_text(
        json.dumps(report, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    )


if __name__ == "__main__":
    main()
