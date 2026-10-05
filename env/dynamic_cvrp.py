"""Dynamic CVRP with exact transitions for customers already revealed.

Travel time equals Euclidean distance. A customer is visible only once
clock time reaches its reveal time. The vehicle replans only when it is
stationary: after a stop, a depot return, or a wait for the next request.
Requests that appear during travel are ignored until the vehicle stops.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


def generate_instance(
    n_customers: int,
    capacity: float,
    horizon: float,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    coords = np.array(
        [[rng.random(), rng.random()] for _ in range(n_customers + 1)],
        dtype=np.float64,
    )
    demands = np.array(
        [0.0] + [float(rng.randrange(1, 10)) for _ in range(n_customers)],
        dtype=np.float64,
    )
    reveal = np.zeros(n_customers + 1, dtype=np.float64)
    reveal[1:] = [rng.random() * horizon for _ in range(n_customers)]
    return _pack_instance(coords, demands, reveal, capacity, horizon)


def generate_clustered_instance(
    n_customers: int,
    capacity: float,
    horizon: float,
    rng: random.Random,
    n_clusters: int = 4,
    sigma: float = 0.04,
) -> Dict[str, np.ndarray]:
    """Customers arrive in tight spatial waves.

    Each cluster has its own reveal window. Once a few members are visible,
    the rest of that cluster appears nearby. Hidden coordinates stay hidden
    from the model; only the simulator uses them.
    """
    coords = np.zeros((n_customers + 1, 2), dtype=np.float64)
    demands = np.zeros(n_customers + 1, dtype=np.float64)
    reveal = np.zeros(n_customers + 1, dtype=np.float64)
    window_start = np.zeros(n_customers + 1, dtype=np.float64)
    window_width = np.zeros(n_customers + 1, dtype=np.float64)
    coords[0] = [rng.random(), rng.random()]
    centers = [(rng.random(), rng.random()) for _ in range(n_clusters)]
    width = horizon / float(n_clusters)
    for index in range(1, n_customers + 1):
        cluster = (index - 1) % n_clusters
        center = centers[cluster]
        coords[index, 0] = center[0] + rng.gauss(0.0, sigma)
        coords[index, 1] = center[1] + rng.gauss(0.0, sigma)
        demands[index] = float(rng.randrange(1, 10))
        start = cluster * width * 0.85
        window_start[index] = start
        window_width[index] = width
        reveal[index] = start + rng.random() * width
    packed = _pack_instance(coords, demands, reveal, capacity, horizon)
    packed["window_start"] = window_start
    packed["window_width"] = window_width
    return packed


def generate_burst_instance(
    n_customers: int,
    capacity: float,
    horizon: float,
    rng: random.Random,
    n_clusters: int = 4,
    sigma: float = 0.015,
    min_center_gap: float = 0.45,
    burst_width: float = 8.0,
    spacing: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Separated bursts whose reveal windows overlap.

    A new wave starts before the previous one is finished, so the vehicle
    can face a nearby leftover and a far customer from the new wave at once.
    """
    centers = []
    for _attempt in range(5000):
        point = (rng.random(), rng.random())
        if all((point[0] - x) ** 2 + (point[1] - y) ** 2 >= min_center_gap ** 2 for x, y in centers):
            centers.append(point)
        if len(centers) == n_clusters:
            break
    if len(centers) < n_clusters:
        raise RuntimeError("could not place separated cluster centers")
    coords = np.zeros((n_customers + 1, 2), dtype=np.float64)
    demands = np.zeros(n_customers + 1, dtype=np.float64)
    reveal = np.zeros(n_customers + 1, dtype=np.float64)
    window_start = np.zeros(n_customers + 1, dtype=np.float64)
    window_width = np.full(n_customers + 1, burst_width, dtype=np.float64)
    coords[0] = [rng.random(), rng.random()]
    for index in range(1, n_customers + 1):
        cluster = (index - 1) % n_clusters
        center = centers[cluster]
        coords[index, 0] = center[0] + rng.gauss(0.0, sigma)
        coords[index, 1] = center[1] + rng.gauss(0.0, sigma)
        demands[index] = float(rng.randrange(1, 10))
        start = cluster * spacing
        window_start[index] = start
        reveal[index] = min(horizon * 0.98, start + rng.random() * burst_width)
    packed = _pack_instance(coords, demands, reveal, capacity, horizon)
    packed["window_start"] = window_start
    packed["window_width"] = window_width
    return packed


def generate_larsen_instance(
    rng: random.Random,
    expected_total: int = 40,
    horizon: float = 32.0,
) -> Dict[str, np.ndarray]:
    """Partially dynamic TRP of Larsen, Madsen and Solomon (2002).

    Unit square = 10 km x 10 km, depot at the centre, unit speed. With the
    vehicle at 40 km/h one time unit is 15 minutes, so an 8-hour day is
    horizon=32 (their region); horizon=8 is the 40 km x 40 km region.
    n_adv ~ U{0..E}; immediate requests follow a Poisson process with rate
    (E - n_adv) / horizon. No capacity; service times are not modelled.
    """
    n_adv = rng.randint(0, expected_total)
    rate = (expected_total - n_adv) / horizon
    reveal = [0.0] * n_adv
    clock = rng.expovariate(rate) if rate > 0 else horizon + 1.0
    while clock <= horizon:
        reveal.append(clock)
        clock += rng.expovariate(rate)
    n = len(reveal)
    coords = np.zeros((n + 1, 2), dtype=np.float64)
    coords[0] = [0.5, 0.5]
    for index in range(1, n + 1):
        coords[index] = [rng.random(), rng.random()]
    demands = np.ones(n + 1, dtype=np.float64)
    demands[0] = 0.0
    packed = _pack_instance(coords, demands, np.array([0.0] + reveal), float(n + 1), horizon)
    packed["n_adv"] = n_adv
    packed["dod"] = (n - n_adv) / n if n else 0.0
    return packed


def _pack_instance(coords, demands, reveal, capacity, horizon) -> Dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    reveal = np.asarray(reveal, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    return {
        "coords": coords,
        "demands": demands,
        "reveal": reveal,
        "dist": dist,
        "capacity": float(capacity),
        "horizon": float(horizon),
    }


def initial_state(n_nodes: int, capacity: float) -> Dict[str, np.ndarray]:
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    return {
        "visited": visited,
        "current": np.int64(0),
        "time": np.float64(0.0),
        "remaining_capacity": np.float64(capacity),
        "cost": np.float64(0.0),
    }


def _known(reveal: np.ndarray, time: float) -> np.ndarray:
    return reveal <= time + 1e-9


def feasible_customers(inst: dict, state: dict) -> List[int]:
    known = _known(inst["reveal"], float(state["time"]))
    out = []
    rem = float(state["remaining_capacity"])
    for node in range(1, inst["demands"].shape[0]):
        if state["visited"][node] or not known[node]:
            continue
        if float(inst["demands"][node]) <= rem + 1e-9:
            out.append(node)
    return out


def rank_nearest(inst: dict, state: dict, nodes: List[int]) -> List[int]:
    current = int(state["current"])
    dist = inst["dist"]
    return sorted(nodes, key=lambda node: (float(dist[current, node]), node))


def policy_action(inst: dict, state: dict) -> Tuple[str, int]:
    """Frozen nearest-feasible dispatcher.

    Returns ('go', node), ('depot', 0), ('wait', 0), or ('done', 0).
    """
    n = inst["demands"].shape[0]
    if bool(state["visited"][1:].all()):
        if int(state["current"]) != 0:
            return ("depot", 0)
        return ("done", 0)

    feas = feasible_customers(inst, state)
    if feas:
        return ("go", rank_nearest(inst, state, feas)[0])

    known = _known(inst["reveal"], float(state["time"]))
    known_left = [
        node
        for node in range(1, n)
        if known[node] and not state["visited"][node]
    ]
    if known_left and int(state["current"]) != 0:
        return ("depot", 0)

    future = [
        float(inst["reveal"][node])
        for node in range(1, n)
        if not state["visited"][node] and float(inst["reveal"][node]) > float(state["time"]) + 1e-9
    ]
    if future:
        return ("wait", 0)
    if int(state["current"]) != 0:
        return ("depot", 0)
    raise RuntimeError("no feasible action and no future reveal")


def apply_action(inst: dict, state: dict, kind: str, node: int) -> dict:
    dist = inst["dist"]
    new = {
        "visited": state["visited"].copy(),
        "current": np.int64(state["current"]),
        "time": np.float64(state["time"]),
        "remaining_capacity": np.float64(state["remaining_capacity"]),
        "cost": np.float64(state["cost"]),
    }
    if kind == "done":
        return new
    if kind == "wait":
        future = inst["reveal"][1:][ ~new["visited"][1:] & (inst["reveal"][1:] > float(new["time"]) + 1e-9) ]
        if future.size == 0:
            raise RuntimeError("wait with no future reveal")
        new["time"] = np.float64(future.min())
        return new
    if kind == "depot":
        current = int(new["current"])
        if current == 0:
            raise RuntimeError("already at depot")
        step = float(dist[current, 0])
        new["cost"] = np.float64(float(new["cost"]) + step)
        new["time"] = np.float64(float(new["time"]) + step)
        new["current"] = np.int64(0)
        new["remaining_capacity"] = np.float64(inst["capacity"])
        if bool(new["visited"][1:].all()):
            pass
        return new
    if kind != "go":
        raise ValueError(kind)
    node = int(node)
    if new["visited"][node]:
        raise RuntimeError(f"customer {node} already visited")
    demand = float(inst["demands"][node])
    if demand > float(new["remaining_capacity"]) + 1e-9:
        raise RuntimeError(f"capacity violation for customer {node}")
    current = int(new["current"])
    step = float(dist[current, node])
    new["cost"] = np.float64(float(new["cost"]) + step)
    new["time"] = np.float64(float(new["time"]) + step)
    new["current"] = np.int64(node)
    new["visited"][node] = True
    new["remaining_capacity"] = np.float64(float(new["remaining_capacity"]) - demand)
    if bool(new["visited"][1:].all()) and node != 0:
        back = float(dist[node, 0])
        new["cost"] = np.float64(float(new["cost"]) + back)
        new["time"] = np.float64(float(new["time"]) + back)
        new["current"] = np.int64(0)
        new["remaining_capacity"] = np.float64(inst["capacity"])
    return new


def rollout_cost(inst: dict, state: dict, max_events: int = 5000) -> float:
    current = state
    for _ in range(max_events):
        kind, node = policy_action(inst, current)
        if kind == "done":
            return float(current["cost"])
        current = apply_action(inst, current, kind, node)
        if kind == "go" and bool(current["visited"][1:].all()):
            return float(current["cost"])
    raise RuntimeError("rollout exceeded max events")


def batched_rollout_cost(
    dist: np.ndarray,
    demands: np.ndarray,
    reveal: np.ndarray,
    visited: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    remaining: np.ndarray,
    cost: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Finish many states with the same nearest-feasible policy.

    reveal/visited: [B, N]. current/time/remaining/cost: [B].
    """
    reveal = np.array(reveal, dtype=np.float64, copy=True)
    visited = np.array(visited, dtype=bool, copy=True)
    current = np.array(current, dtype=np.int64, copy=True)
    time = np.array(time, dtype=np.float64, copy=True)
    remaining = np.array(remaining, dtype=np.float64, copy=True)
    cost = np.array(cost, dtype=np.float64, copy=True)
    demands = np.asarray(demands, dtype=np.float64)
    batch, n_nodes = visited.shape
    active = np.ones(batch, dtype=bool)
    known = np.empty((batch, n_nodes), dtype=bool)
    feasible = np.empty((batch, n_nodes), dtype=bool)
    known_left = np.empty((batch, n_nodes), dtype=bool)
    future = np.empty((batch, n_nodes), dtype=bool)
    masked = np.empty((batch, n_nodes), dtype=np.float64)
    time_limit = np.empty((batch, 1), dtype=np.float64)
    cap_limit = np.empty((batch, 1), dtype=np.float64)

    for _ in range(n_nodes * 4 + 8):
        if not active.any():
            break
        all_served = visited[:, 1:].all(axis=1)

        finishing = active & all_served & (current != 0)
        if finishing.any():
            idx = np.flatnonzero(finishing)
            step = dist[current[idx], 0]
            cost[idx] += step
            time[idx] += step
            current[idx] = 0
            remaining[idx] = capacity
        active &= ~all_served
        if not active.any():
            break

        np.add(time, 1e-9, out=time_limit[:, 0])
        np.less_equal(reveal, time_limit, out=known)
        np.subtract(capacity, capacity, out=cap_limit[:, 0])
        np.add(remaining, 1e-9, out=cap_limit[:, 0])
        np.less_equal(demands, cap_limit, out=feasible)
        np.logical_and(feasible, known, out=feasible)
        np.logical_and(feasible, ~visited, out=feasible)
        feasible[:, 0] = False
        feasible[~active] = False
        row = dist[current]
        np.copyto(masked, row)
        masked[~feasible] = np.inf
        has = np.isfinite(masked).any(axis=1)
        choice = np.argmin(masked, axis=1)

        if has.any():
            idx = np.flatnonzero(has)
            nxt = choice[idx]
            step = dist[current[idx], nxt]
            cost[idx] += step
            time[idx] += step
            remaining[idx] -= demands[nxt]
            current[idx] = nxt
            visited[idx, nxt] = True

        np.logical_and(known, ~visited, out=known_left)
        known_left[:, 0] = False
        need_depot = active & (~has) & known_left.any(axis=1) & (current != 0)
        if need_depot.any():
            idx = np.flatnonzero(need_depot)
            step = dist[current[idx], 0]
            cost[idx] += step
            time[idx] += step
            current[idx] = 0
            remaining[idx] = capacity

        np.add(time, 1e-9, out=time_limit[:, 0])
        np.greater(reveal, time_limit, out=future)
        np.logical_and(future, ~visited, out=future)
        future[:, 0] = False
        can_wait = active & (~has) & (~need_depot) & future.any(axis=1)
        if can_wait.any():
            next_time = np.where(future, reveal, np.inf).min(axis=1)
            time[can_wait] = next_time[can_wait]

        stuck = active & (~has) & (~need_depot) & (~can_wait)
        if stuck.any():
            raise RuntimeError(f"{int(stuck.sum())} rollouts stuck")
    if active.any():
        raise RuntimeError(f"{int(active.sum())} rollouts did not finish")
    return cost


def batched_rollout_cost_coords(
    coords: np.ndarray,
    demands: np.ndarray,
    reveal: np.ndarray,
    visited: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    remaining: np.ndarray,
    cost: np.ndarray,
    capacity: float,
    return_time: bool = False,
):
    """``batched_rollout_cost`` where every rollout has its own node set.

    coords: [B, N, 2]; demands/reveal/visited: [B, N]; node 0 is the depot.
    Padding nodes are passed as visited. With ``return_time`` also returns
    the depot return time of each rollout.
    """
    coords = np.asarray(coords, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    reveal = np.asarray(reveal, dtype=np.float64)
    visited = np.array(visited, dtype=bool, copy=True)
    current = np.array(current, dtype=np.int64, copy=True)
    time = np.array(time, dtype=np.float64, copy=True)
    remaining = np.array(remaining, dtype=np.float64, copy=True)
    cost = np.array(cost, dtype=np.float64, copy=True)
    batch, n_nodes = visited.shape
    rows = np.arange(batch)
    active = np.ones(batch, dtype=bool)

    def step_to(idx, nxt):
        delta = coords[idx, nxt] - coords[idx, current[idx]]
        step = np.sqrt((delta * delta).sum(axis=-1))
        cost[idx] += step
        time[idx] += step

    for _ in range(n_nodes * 4 + 8):
        all_served = visited[:, 1:].all(axis=1)
        finishing = active & all_served & (current != 0)
        if finishing.any():
            idx = np.flatnonzero(finishing)
            step_to(idx, np.zeros(idx.size, dtype=np.int64))
            current[idx] = 0
        active &= ~all_served
        if not active.any():
            break
        known = reveal <= time[:, None] + 1e-9
        open_nodes = known & ~visited
        open_nodes[:, 0] = False
        feasible = open_nodes & (demands <= remaining[:, None] + 1e-9)
        feasible[~active] = False
        delta = coords - coords[rows, current][:, None, :]
        masked = np.where(feasible, (delta * delta).sum(axis=-1), np.inf)
        has = feasible.any(axis=1)
        if has.any():
            idx = np.flatnonzero(has)
            nxt = np.argmin(masked[idx], axis=1)
            step_to(idx, nxt)
            remaining[idx] -= demands[idx, nxt]
            current[idx] = nxt
            visited[idx, nxt] = True
        need_depot = active & ~has & open_nodes.any(axis=1) & (current != 0)
        if need_depot.any():
            idx = np.flatnonzero(need_depot)
            step_to(idx, np.zeros(idx.size, dtype=np.int64))
            current[idx] = 0
            remaining[idx] = capacity
        future = (reveal > time[:, None] + 1e-9) & ~visited
        future[:, 0] = False
        can_wait = active & ~has & ~need_depot & future.any(axis=1)
        if can_wait.any():
            time[can_wait] = np.where(future, reveal, np.inf).min(axis=1)[can_wait]
        stuck = active & ~has & ~need_depot & ~can_wait
        if stuck.any():
            raise RuntimeError(f"{int(stuck.sum())} rollouts stuck")
    if active.any():
        raise RuntimeError(f"{int(active.sum())} rollouts did not finish")
    return (cost, time) if return_time else cost


def self_test() -> None:
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    demands = np.array([0.0, 2.0, 2.0, 3.0], dtype=np.float64)
    reveal = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    inst = {
        "coords": coords,
        "demands": demands,
        "reveal": reveal,
        "dist": dist,
        "capacity": 4.0,
        "horizon": 10.0,
    }
    state = initial_state(4, 4.0)
    cost = rollout_cost(inst, state)
    # 0->1->2, wait, 2->0, 0->3->0
    expected = 1.0 + 1.0 + 2.0 + 2.0 + 2.0
    if abs(cost - expected) > 1e-8:
        raise AssertionError(f"scalar rollout {cost} != {expected}")

    batch_cost = batched_rollout_cost(
        dist,
        demands,
        reveal[None, :],
        state["visited"][None, :],
        np.array([0]),
        np.array([0.0]),
        np.array([4.0]),
        np.array([0.0]),
        4.0,
    )
    if abs(float(batch_cost[0]) - expected) > 1e-8:
        raise AssertionError(f"batched rollout {batch_cost[0]} != {expected}")
    print("dynamic self-test: OK")
