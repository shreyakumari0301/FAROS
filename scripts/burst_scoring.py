"""Scenario rollouts for action scoring (Bent / SS-VRPTW-CR style recourse)."""

from __future__ import annotations

import random

import numpy as np

from env.dynamic_cvrp import apply_action, batched_rollout_cost


def score_actions(
    inst: dict,
    state: dict,
    actions,
    n_futures: int,
    rng: random.Random,
    stats: dict | None = None,
    sampler=None,
    common: bool = False,
) -> np.ndarray:
    """Mean nearest-completion cost of each action over sampled futures.

    ``sampler(inst, time, batch, rng)`` replaces the oracle window sampler.
    ``common=True`` scores every action on the same futures (common random
    numbers, as in MSA) and sets the post-wait time from each sampled future
    instead of the realized next reveal.
    """
    if stats is not None:
        stats["msa_calls"] = int(stats.get("msa_calls", 0)) + 1
        stats["futures_scored"] = int(stats.get("futures_scored", 0)) + len(actions) * n_futures
    decision_time = float(state["time"])
    sampled_wait = sampler is not None or common
    posts = []
    for kind, node in actions:
        if kind == "wait":
            posts.append(dict(state) if sampled_wait else apply_action(inst, state, "wait", 0))
        else:
            posts.append(apply_action(inst, state, "go", int(node)))
    k = len(posts)
    batch = k * n_futures
    draws_needed = n_futures if common else batch
    if sampler is not None:
        reveal = sampler(inst, decision_time, draws_needed, rng)
    else:
        reveal = np.repeat(inst["reveal"][None, :], draws_needed, axis=0)
    unknown = np.flatnonzero(inst["reveal"] > decision_time + 1e-9)
    unknown = unknown[unknown != 0]
    if unknown.size and sampler is None:
        low = np.maximum(inst["window_start"][unknown], decision_time)
        high = inst["window_start"][unknown] + inst["window_width"][unknown]
        span = np.maximum(high - low, 0.0)
        draws = np.array(
            [rng.random() for _ in range(draws_needed * int(unknown.size))],
            dtype=np.float64,
        ).reshape(draws_needed, unknown.size)
        sampled = low + draws * span
        reveal[:, unknown] = np.where(span > 0.0, sampled, decision_time)
    if common:
        reveal = np.tile(reveal, (k, 1))
    visited = np.repeat(np.stack([post["visited"] for post in posts]), n_futures, axis=0)
    current = np.repeat([int(post["current"]) for post in posts], n_futures)
    time = np.repeat([float(post["time"]) for post in posts], n_futures)
    remaining = np.repeat([float(post["remaining_capacity"]) for post in posts], n_futures)
    cost = np.repeat([float(post["cost"]) for post in posts], n_futures)
    if sampled_wait:
        for index, (kind, _node) in enumerate(actions):
            if kind != "wait":
                continue
            rows = slice(index * n_futures, (index + 1) * n_futures)
            pending = np.where(reveal[rows] > decision_time + 1e-9, reveal[rows], np.inf)
            nxt = pending.min(axis=1)
            time[rows] = np.where(np.isfinite(nxt), nxt, decision_time)
    finished = batched_rollout_cost(
        inst["dist"],
        inst["demands"],
        reveal,
        visited,
        current,
        time,
        remaining,
        cost,
        float(inst["capacity"]),
    )
    return finished.reshape(k, n_futures).mean(axis=1)
