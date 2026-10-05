"""Level B: FAROS when future customer locations are unknown.

Policies observe only revealed requests. A generative world model, fit on
training days (episodes 0-11), imagines the unseen customers (location,
demand, reveal time) for both the waiting rule and the sampled futures.
Test days 16-19; actions are scored on common futures. Full-information
results come from the Level A log (scripts/analyze_world_model.py pairs
them). Writes outputs/logs/burst_world_model_b*.json.

  python scripts/run_world_model_b.py                     # learned world model
  python scripts/run_world_model_b.py --spatial uniform   # ablation
  python scripts/run_world_model_b.py --probe             # 1 seed
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import random
import sys
import time as wall
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import apply_action, batched_rollout_cost_coords, rank_nearest
from models.request_world_model import RequestWorldModel
from scripts.burst_dispatch import HORIZON, MIN_SEP, TOP_K, is_cross_burst, nearest_choice, rollout_tour
from scripts.burst_instances import DEFAULT_EPISODES, load_or_create_bank
from scripts.run_burst_benchmark import DEFAULT_SEEDS, SCENARIO_RNG_OFFSET
from scripts.run_paper_baselines import IMMINENT_HORIZON
from scripts.run_world_model_benchmark import sign_test
from scripts.train_burst_value import diverse_candidates
from scripts.train_burst_wait import split_episodes

FUTURES = 20
WORKERS = 5
LOG = Path(ROOT) / "outputs" / "logs"
POLICIES = ("sampling", "wait_rule", "faros_go", "faros_all")


def observe(world: RequestWorldModel, inst: dict, time: float):
    """Revealed customers only: node list (depot first) and grouped waves."""
    if hasattr(world, "bind"):
        world.bind(inst)
    revealed = np.flatnonzero(inst["reveal"] <= time + 1e-9)
    revealed = revealed[revealed != 0]
    nodes = np.concatenate([[0], revealed]).astype(np.int64)
    _labels, waves = world.observe(inst["coords"][revealed], inst["reveal"][revealed], time)
    wave_of = {}
    for index, wave in enumerate(waves):
        wave["nodes"] = revealed[wave["members"]]
        for member in wave["members"]:
            wave_of[int(revealed[member])] = index
    return nodes, waves, wave_of


def expects_more(world, waves) -> bool:
    return world.expected_hidden(waves) >= 0.5


def score_imagined(world, inst, state, actions, n_futures, rng, stats, common=True, return_time=False):
    """Mean completion cost per action over imagined futures (shared across actions if ``common``).

    With ``return_time`` also returns the depot return times, shape [actions, futures].
    """
    stats["msa_calls"] = int(stats.get("msa_calls", 0)) + 1
    stats["futures_scored"] = int(stats.get("futures_scored", 0)) + len(actions) * n_futures
    time = float(state["time"])
    nodes, waves, _ = observe(world, inst, time)
    position = {int(node): i for i, node in enumerate(nodes)}
    posts = [dict(state) if kind == "wait" else apply_action(inst, state, "go", int(node)) for kind, node in actions]
    batch = len(posts) * n_futures
    futures = world.imagine(waves, time, n_futures if common else batch, rng)
    if common:
        futures = futures * len(posts)
    n_obs = nodes.size
    width = n_obs + max(f[0].shape[0] for f in futures)
    coords = np.zeros((batch, width, 2))
    demands = np.zeros((batch, width))
    reveal = np.zeros((batch, width))
    visited = np.ones((batch, width), dtype=bool)
    coords[:, :n_obs] = inst["coords"][nodes]
    demands[:, :n_obs] = inst["demands"][nodes]
    reveal[:, :n_obs] = inst["reveal"][nodes]
    current = np.zeros(batch, dtype=np.int64)
    clock = np.zeros(batch)
    remaining = np.zeros(batch)
    cost = np.zeros(batch)
    for row, (xy, dem, rev) in enumerate(futures):
        post = posts[row // n_futures]
        m = xy.shape[0]
        coords[row, n_obs : n_obs + m] = xy
        demands[row, n_obs : n_obs + m] = dem
        reveal[row, n_obs : n_obs + m] = rev
        visited[row, :n_obs] = post["visited"][nodes]
        visited[row, n_obs : n_obs + m] = False
        current[row] = position[int(post["current"])]
        clock[row] = float(post["time"])
        if actions[row // n_futures][0] == "wait":
            pending = rev[rev > time + 1e-9]
            clock[row] = float(pending.min()) if pending.size else time
        remaining[row] = float(post["remaining_capacity"])
        cost[row] = float(post["cost"])
    finished, finish_time = batched_rollout_cost_coords(
        coords, demands, reveal, visited, current, clock, remaining, cost, float(inst["capacity"]), return_time=True
    )
    means = finished.reshape(len(posts), n_futures).mean(axis=1)
    if return_time:
        return means, finish_time.reshape(len(posts), n_futures)
    return means


def cross_actions(world, inst, state):
    actions = [("go", int(n)) for n in diverse_candidates(inst, state, TOP_K, MIN_SEP)]
    _nodes, waves, _ = observe(world, inst, float(state["time"]))
    if expects_more(world, waves):
        actions.append(("wait", 0))
    return actions


def wait_rule_factory(world):
    def choose(inst, state, feas):
        nearest = ("go", rank_nearest(inst, state, feas)[0])
        if not is_cross_burst(inst, state, HORIZON):
            return nearest
        time = float(state["time"])
        nodes, waves, wave_of = observe(world, inst, time)
        if not expects_more(world, waves):
            return nearest
        visible = [int(n) for n in nodes[1:] if not state["visited"][n]]
        if not visible:
            return nearest
        anchor = max(visible, key=lambda n: float(inst["reveal"][n]))
        next_t = world.expected_next_reveal(waves[wave_of[anchor]], time)
        if next_t is not None and next_t - time <= IMMINENT_HORIZON:
            return ("wait", 0)
        return nearest

    return choose


def sampling_factory(world, rng, stats):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        actions = cross_actions(world, inst, state)
        costs = score_imagined(world, inst, state, actions, FUTURES, rng, stats)
        kind, node = actions[int(np.argmin(costs))]
        if kind == "wait":
            return ("wait", 0)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def faros_go_factory(world, rng, stats):
    wait_cb = wait_rule_factory(world)

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        if wait_cb(inst, state, feas)[0] == "wait":
            return ("wait", 0)
        go_actions = [a for a in cross_actions(world, inst, state) if a[0] == "go"]
        if len(go_actions) <= 1:
            return ("go", rank_nearest(inst, state, feas)[0])
        costs = score_imagined(world, inst, state, go_actions, FUTURES, rng, stats)
        node = go_actions[int(np.argmin(costs))][1]
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def faros_all_factory(world, rng, stats):
    wait_cb = wait_rule_factory(world)
    sample_cb = sampling_factory(world, rng, stats)

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        if wait_cb(inst, state, feas)[0] == "wait":
            return ("wait", 0)
        return sample_cb(inst, state, feas)

    return choose


class BlindWorld:
    """Wraps a world model but imagines nothing: rollout over known customers only."""

    def __init__(self, world):
        self.world = world

    def observe(self, coords, reveal, time=0.0):
        return self.world.observe(coords, reveal, time)

    def expected_hidden(self, waves) -> float:
        return 0.0

    def expected_next_reveal(self, wave, time):
        return None

    def imagine(self, waves, time, batch, rng):
        return [(np.zeros((0, 2)), np.zeros(0), np.zeros(0)) for _ in range(batch)]


def run_seed(args):
    seed, spatial = args
    train_eps, _val_eps, test_eps = split_episodes(DEFAULT_EPISODES)
    instances = load_or_create_bank(seed, DEFAULT_EPISODES)
    world = RequestWorldModel.fit([instances[ep] for ep in train_eps], spatial=spatial)
    stats = {k: {} for k in ("sampling", "faros_go", "faros_all", "rollout_known")}

    def rng():
        return random.Random(seed + SCENARIO_RNG_OFFSET)

    policies = {
        "sampling": sampling_factory(world, rng(), stats["sampling"]),
        "wait_rule": wait_rule_factory(world),
        "faros_go": faros_go_factory(world, rng(), stats["faros_go"]),
        "faros_all": faros_all_factory(world, rng(), stats["faros_all"]),
        "rollout_known": faros_go_factory(BlindWorld(world), rng(), stats["rollout_known"]),
    }
    row = {
        "seed": seed,
        "world_model": {
            "width": world.width,
            "spread": world.spread,
            "min_gap": world.min_gap,
            "mean_size": float(world.sizes.mean()),
            "waves_per_day": float(world.waves_per_day.mean()),
        },
        "costs": {"nearest": [float(rollout_tour(instances[ep], nearest_choice, horizon=HORIZON)[0]) for ep in test_eps]},
        "seconds": {},
        "scoring_stats": stats,
    }
    for name, cross in policies.items():
        started = wall.perf_counter()
        run = []
        for ep in test_eps:
            cost, _, _, _ = rollout_tour(instances[ep], nearest_choice, choose_at_cross=cross, horizon=HORIZON)
            run.append(float(cost))
        row["costs"][f"wmb_{name}"] = run
        row["seconds"][f"wmb_{name}"] = wall.perf_counter() - started
        print(f"seed {seed} wmb_{name}: {np.mean(run):.4f}", flush=True)
    return row


def main():
    probe = "--probe" in sys.argv
    spatial = sys.argv[sys.argv.index("--spatial") + 1] if "--spatial" in sys.argv else "learned"
    seeds = DEFAULT_SEEDS[:1] if probe else DEFAULT_SEEDS
    train_eps, _val_eps, test_eps = split_episodes(DEFAULT_EPISODES)
    with Pool(min(WORKERS, len(seeds))) as pool:
        per_seed = pool.map(run_seed, [(seed, spatial) for seed in seeds])
    names = ["nearest"] + [f"wmb_{k}" for k in (*POLICIES, "rollout_known")]
    costs = {name: [c for row in per_seed for c in row["costs"][name]] for name in names}
    paired = {f"wmb_{k} vs nearest": sign_test(costs["nearest"], costs[f"wmb_{k}"]) for k in (*POLICIES, "rollout_known")}
    for k in ("wait_rule", "faros_go", "faros_all"):
        paired[f"wmb_{k} vs wmb_sampling"] = sign_test(costs["wmb_sampling"], costs[f"wmb_{k}"])
    paired["wmb_faros_go vs wmb_wait_rule"] = sign_test(costs["wmb_wait_rule"], costs["wmb_faros_go"])
    paired["wmb_faros_all vs wmb_wait_rule"] = sign_test(costs["wmb_wait_rule"], costs["wmb_faros_all"])
    paired["wmb_faros_go vs wmb_rollout_known"] = sign_test(costs["wmb_rollout_known"], costs["wmb_faros_go"])
    out = {
        "level": "B",
        "common_random_numbers": True,
        "spatial": spatial,
        "seeds": list(seeds),
        "train_episodes": list(train_eps),
        "test_episodes": list(test_eps),
        "futures": FUTURES,
        "mean_cost": {k: float(np.mean(v)) for k, v in costs.items()},
        "sd_cost": {k: float(np.std(v, ddof=1)) for k, v in costs.items()},
        "paired": paired,
        "per_seed": per_seed,
        "costs": costs,
    }
    suffix = "" if spatial == "learned" else f"_{spatial}"
    suffix += "_probe" if probe else ""
    path = LOG / f"burst_world_model_b{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("mean_cost", "paired")}, indent=2))


if __name__ == "__main__":
    main()
