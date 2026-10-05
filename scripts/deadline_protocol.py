"""Main experiment: burst dynamic CVRP with a hard return-to-depot deadline.

Problem: minimise travel distance subject to returning to the depot by
T_RETURN after serving every customer. Requests are revealed by
T_ARRIVAL = 5 * (4 - 1) + 8 = 23 by construction of the generator.
T_RETURN = 40 is the existing horizon: immediate nearest dispatch returns
by 34.22 on all 200 bank days (feasible by construction, checked below),
while waiting for every request misses 40 on 78% of days.

Information: the planner sees only revealed requests. Futures come from a
world object; all sampling planners score actions on common futures and
treat an action as admissible only if every sampled future returns by
T_RETURN.

Worlds (future knowledge, weakest to strongest):
  known      -- nothing imagined (rollout over known customers)
  uniform    -- learned timing/counts, locations uniform in the region
  learned    -- learned generative request world model
  oracle     -- today's true locations and windows, reveal times sampled
  clairvoyant-- today's realized future requests (not deployable)

Writes outputs/logs/deadline_main.json.
  python scripts/deadline_protocol.py [--probe]
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

from env.dynamic_cvrp import apply_action, feasible_customers, initial_state, policy_action, rank_nearest
from models.request_world_model import RequestWorldModel
from scripts.burst_dispatch import HORIZON, has_future_reveal, is_cross_burst
from scripts.burst_instances import DEFAULT_EPISODES, load_or_create_bank
from scripts.run_burst_benchmark import DEFAULT_SEEDS, SCENARIO_RNG_OFFSET
from scripts.run_paper_baselines import IMMINENT_HORIZON, WINDOW_TOL
from scripts.run_world_model_b import BlindWorld, cross_actions, expects_more, observe, score_imagined
from scripts.run_world_model_benchmark import sign_test
from scripts.train_burst_wait import split_episodes

T_ARRIVAL = 23.0
T_RETURN = 40.0
FUTURES = 20
WORKERS = 10
LOG = Path(ROOT) / "outputs" / "logs"


# ---- worlds with access to today's truth ------------------------------------


class _TruthWorld:
    """Groups waves like ``base`` but imagines from the bound instance."""

    def __init__(self, base):
        self.base = base
        self.inst = None
        self.time = 0.0

    def bind(self, inst):
        self.inst = inst

    def observe(self, coords, reveal, time=0.0):
        self.time = float(time)
        return self.base.observe(coords, reveal, time)

    def _hidden(self):
        hidden = np.flatnonzero(self.inst["reveal"] > self.time + 1e-9)
        return hidden[hidden != 0]

    def expected_hidden(self, waves) -> float:
        return float(self._hidden().size)

    def _same_wave(self, wave):
        start = float(self.inst["window_start"][int(wave["nodes"][0])])
        hidden = self._hidden()
        return hidden[np.abs(self.inst["window_start"][hidden] - start) <= WINDOW_TOL], start


class OracleWorld(_TruthWorld):
    """True locations and windows of today's customers; reveal times sampled."""

    def expected_next_reveal(self, wave, time):
        hidden, start = self._same_wave(wave)
        if hidden.size == 0:
            return None
        low = max(start, time)
        high = start + float(self.inst["window_width"][hidden[0]])
        return low + max(high - low, 0.0) / (hidden.size + 1)

    def imagine(self, waves, time, batch, rng):
        gen = np.random.default_rng(rng.getrandbits(63))
        hidden = self._hidden()
        low = np.maximum(self.inst["window_start"][hidden], time)
        high = self.inst["window_start"][hidden] + self.inst["window_width"][hidden]
        xy, dem = self.inst["coords"][hidden], self.inst["demands"][hidden]
        return [(xy, dem, low + gen.random(hidden.size) * np.maximum(high - low, 0.0)) for _ in range(batch)]


class ClairvoyantWorld(_TruthWorld):
    """Today's realized future requests (upper bound, not deployable)."""

    def expected_next_reveal(self, wave, time):
        if "window_start" not in self.inst:
            return None
        hidden, _start = self._same_wave(wave)
        return float(self.inst["reveal"][hidden].min()) if hidden.size else None

    def imagine(self, waves, time, batch, rng):
        hidden = self._hidden()
        future = (self.inst["coords"][hidden], self.inst["demands"][hidden], self.inst["reveal"][hidden])
        return [future] * batch


# ---- deadline-aware decisions -----------------------------------------------


def _nearest(inst, state, feas):
    return ("go", rank_nearest(inst, state, feas)[0])


def _pick(actions, means, finish, t_return):
    """Cheapest action whose every sampled future returns by the deadline."""
    on_time = finish <= t_return + 1e-9
    safe = on_time.all(axis=1)
    if safe.any():
        return actions[int(np.argmin(np.where(safe, means, np.inf)))]
    share = on_time.mean(axis=1)
    return actions[int(np.argmin(np.where(share == share.max(), means, np.inf)))]


def _wait_safe(world, inst, state, futures, rng, stats, t_return):
    _means, finish = score_imagined(world, inst, state, [("wait", 0)], futures, rng, stats, return_time=True)
    return bool((finish <= t_return + 1e-9).all())


def _rule_wants_wait(world, inst, state, time):
    nodes, waves, wave_of = observe(world, inst, time)
    if not expects_more(world, waves):
        return False
    visible = [int(n) for n in nodes[1:] if not state["visited"][n]]
    if not visible:
        return False
    anchor = max(visible, key=lambda n: float(inst["reveal"][n]))
    next_t = world.expected_next_reveal(waves[wave_of[anchor]], time)
    return next_t is not None and next_t - time <= IMMINENT_HORIZON


def wait_all_policy(world):
    """Diagnostic: idle whenever more requests are expected; ignores the deadline."""

    def choose(inst, state, feas):
        _nodes, waves, _ = observe(world, inst, float(state["time"]))
        return ("wait", 0) if expects_more(world, waves) else _nearest(inst, state, feas)

    return choose


def latest_departure_policy(world, rng, stats, futures=FUTURES, t_return=T_RETURN, margin=0.0):
    """Idle as long as every imagined future still returns ``margin`` before the deadline."""

    def choose(inst, state, feas):
        _nodes, waves, _ = observe(world, inst, float(state["time"]))
        if expects_more(world, waves) and _wait_safe(world, inst, state, futures, rng, stats, t_return - margin):
            return ("wait", 0)
        return _nearest(inst, state, feas)

    return choose


def wait_rule_policy(world, rng, stats, futures=FUTURES, t_return=T_RETURN):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return _nearest(inst, state, feas)
        if _rule_wants_wait(world, inst, state, float(state["time"])) and _wait_safe(
            world, inst, state, futures, rng, stats, t_return
        ):
            return ("wait", 0)
        return _nearest(inst, state, feas)

    return choose


def sampling_policy(world, rng, stats, futures=FUTURES, t_return=T_RETURN, go_only=False):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return _nearest(inst, state, feas)
        actions = cross_actions(world, inst, state)
        if go_only:
            actions = [a for a in actions if a[0] == "go"]
        if len(actions) <= 1:
            return _nearest(inst, state, feas)
        means, finish = score_imagined(world, inst, state, actions, futures, rng, stats, return_time=True)
        kind, node = _pick(actions, means, finish, t_return)
        if kind == "wait":
            return ("wait", 0)
        return ("go", node) if node in feas else _nearest(inst, state, feas)

    return choose


def faros_policy(world, rng, stats, futures=FUTURES, t_return=T_RETURN, go_only=False):
    """FAROS-Go (go_only) / FAROS-All: guarded wave rule first, then sampling."""
    sample = sampling_policy(world, rng, stats, futures, t_return, go_only)

    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return _nearest(inst, state, feas)
        if _rule_wants_wait(world, inst, state, float(state["time"])) and _wait_safe(
            world, inst, state, futures, rng, stats, t_return
        ):
            return ("wait", 0)
        return sample(inst, state, feas)

    return choose


# ---- simulation ---------------------------------------------------------------


def simulate(inst, policy=None, t_return=T_RETURN):
    """Run one day; ``policy`` is consulted whenever a customer is feasible."""
    state = initial_state(inst["demands"].shape[0], float(inst["capacity"]))
    response, waits = [], 0
    for _ in range(inst["demands"].shape[0] * 8 + 16):
        if bool(state["visited"][1:].all()):
            break
        feas = feasible_customers(inst, state)
        if feas:
            kind, node = policy(inst, state, feas) if policy is not None else _nearest(inst, state, feas)
            if kind == "wait" and has_future_reveal(inst, state):
                waits += 1
                state = apply_action(inst, state, "wait", 0)
                continue
            node = int(node) if kind == "go" and int(node) in feas else rank_nearest(inst, state, feas)[0]
            arrival = float(state["time"]) + float(inst["dist"][int(state["current"]), node])
            response.append(arrival - float(inst["reveal"][node]))
            state = apply_action(inst, state, "go", node)
            continue
        kind, node = policy_action(inst, state)
        if kind == "done":
            break
        state = apply_action(inst, state, kind, node)
    if int(state["current"]) != 0:
        state = apply_action(inst, state, "depot", 0)
    finish = float(state["time"])
    return {
        "cost": float(state["cost"]),
        "finish": finish,
        "feasible": finish <= t_return + 1e-9,
        "mean_response": float(np.mean(response)) if response else 0.0,
        "waits": waits,
    }


# ---- experiment ----------------------------------------------------------------

PRIMARY = (
    ("faros_all", "nearest"),
    ("faros_all", "latest_departure"),
    ("faros_all", "sampling_learned"),
    ("faros_go", "latest_departure"),
    ("latest_departure_clairvoyant", "faros_all"),
    ("faros_all", "faros_all_uniform"),
    ("faros_go", "sampling_learned"),
    ("faros_go", "rollout_known"),
    ("sampling_learned", "rollout_known"),
    ("sampling_learned", "sampling_uniform"),
    ("sampling_oracle", "sampling_learned"),
    ("sampling_clairvoyant", "sampling_learned"),
)


MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def calibrate_margin(seed, learned, validation):
    """Smallest latest-departure safety margin that is feasible on every validation day."""
    for margin in MARGINS:
        policy = latest_departure_policy(learned, random.Random(seed + SCENARIO_RNG_OFFSET), {}, margin=margin)
        if all(simulate(inst, policy)["feasible"] for inst in validation):
            return margin
    return MARGINS[-1]


def build_policies(seed, learned, uniform, margin=0.0):
    worlds = {
        "known": BlindWorld(learned),
        "uniform": uniform,
        "learned": learned,
        "oracle": OracleWorld(learned),
        "clairvoyant": ClairvoyantWorld(learned),
    }
    stats = {}

    def rng():
        return random.Random(seed + SCENARIO_RNG_OFFSET)

    def tracked(name):
        stats[name] = {}
        return stats[name]

    policies = {
        "nearest": None,
        "wait_all": wait_all_policy(learned),
        "latest_departure_uncalibrated": latest_departure_policy(
            learned, rng(), tracked("latest_departure_uncalibrated")
        ),
        "latest_departure": latest_departure_policy(learned, rng(), tracked("latest_departure"), margin=margin),
        "latest_departure_clairvoyant": latest_departure_policy(
            worlds["clairvoyant"], rng(), tracked("latest_departure_clairvoyant"), futures=1
        ),
        "rollout_known": sampling_policy(worlds["known"], rng(), tracked("rollout_known"), futures=1, go_only=True),
        "wait_rule": wait_rule_policy(learned, rng(), tracked("wait_rule")),
        "faros_go": faros_policy(learned, rng(), tracked("faros_go"), go_only=True),
        "faros_all": faros_policy(learned, rng(), tracked("faros_all")),
    }
    for key in ("uniform", "learned", "oracle", "clairvoyant"):
        futures = 1 if key == "clairvoyant" else FUTURES
        policies[f"sampling_{key}"] = sampling_policy(worlds[key], rng(), tracked(f"sampling_{key}"), futures)
        if key != "learned":
            policies[f"faros_all_{key}"] = faros_policy(worlds[key], rng(), tracked(f"faros_all_{key}"), futures)
    return policies, stats


def run_seed(seed):
    train_eps, val_eps, test_eps = split_episodes(DEFAULT_EPISODES)
    instances = load_or_create_bank(seed, DEFAULT_EPISODES)
    certificate = [simulate(inst)["finish"] for inst in instances]
    if max(certificate) > T_RETURN:
        raise RuntimeError(f"seed {seed}: nearest misses the deadline; instance bank is not feasible by construction")
    train = [instances[ep] for ep in train_eps]
    learned = RequestWorldModel.fit(train)
    uniform = RequestWorldModel.fit(train, spatial="uniform")
    margin = calibrate_margin(seed, learned, [instances[ep] for ep in val_eps])
    print(f"seed {seed}: latest-departure margin {margin}", flush=True)
    policies, stats = build_policies(seed, learned, uniform, margin)
    row = {
        "seed": seed,
        "nearest_finish_max": max(certificate),
        "latest_departure_margin": margin,
        "results": {},
        "seconds": {},
        "scoring_stats": stats,
    }
    for name, policy in policies.items():
        started = wall.perf_counter()
        row["results"][name] = [simulate(instances[ep], policy) for ep in test_eps]
        row["seconds"][name] = wall.perf_counter() - started
        mean = np.mean([r["cost"] for r in row["results"][name]])
        print(f"seed {seed} {name}: {mean:.3f} ({row['seconds'][name]:.0f}s)", flush=True)
    (LOG / f"deadline_seed_{seed}.json").write_text(json.dumps(row, indent=2, default=_jsonable))
    return row


def _jsonable(value):
    return value.item() if hasattr(value, "item") else str(value)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def main():
    probe = "--probe" in sys.argv
    seeds = DEFAULT_SEEDS[:1] if probe else DEFAULT_SEEDS
    with Pool(min(WORKERS, len(seeds))) as pool:
        per_seed = pool.map(run_seed, seeds)
    names = list(per_seed[0]["results"])
    days = {name: [r for row in per_seed for r in row["results"][name]] for name in names}
    table = {}
    for name, rows in days.items():
        cost = np.array([r["cost"] for r in rows])
        table[name] = {
            "distance": float(cost.mean()),
            "distance_sd": float(cost.std(ddof=1)),
            "return_time": float(np.mean([r["finish"] for r in rows])),
            "response_time": float(np.mean([r["mean_response"] for r in rows])),
            "feasible_share": float(np.mean([r["feasible"] for r in rows])),
            "distance_feasible_days": float(np.mean([r["cost"] for r in rows if r["feasible"]] or [np.nan])),
        }
    paired = {}
    for method, base in PRIMARY:
        paired[f"{method} vs {base}"] = sign_test(
            [r["cost"] for r in days[base]], [r["cost"] for r in days[method]]
        )
    adjusted = holm(np.array([v["p_two_sided"] for v in paired.values()]))
    for value, p in zip(paired.values(), adjusted):
        value["p_holm"] = float(p)
    out = {
        "protocol": {
            "t_arrival": T_ARRIVAL,
            "t_return": T_RETURN,
            "futures": FUTURES,
            "admissible": "every sampled future returns by t_return",
            "common_random_numbers": True,
            "information": "revealed requests only",
        },
        "seeds": list(seeds),
        "test_episodes": split_episodes(DEFAULT_EPISODES)[2],
        "table": table,
        "paired": paired,
        "per_seed": per_seed,
    }
    suffix = "_probe" if probe else ""
    (LOG / f"deadline_main{suffix}.json").write_text(json.dumps(out, indent=2, default=_jsonable))
    for name, entry in table.items():
        print(
            f"{name:24s} dist {entry['distance']:6.2f}  return {entry['return_time']:5.1f}  "
            f"response {entry['response_time']:5.2f}  feasible {100 * entry['feasible_share']:5.1f}%"
        )
    for key, value in paired.items():
        print(f"{key:44s} {value['wins']}/{value['losses']}/{value['ties']}  d={value['mean_delta']:.3f}  p_holm={value['p_holm']:.3g}")


if __name__ == "__main__":
    main()
