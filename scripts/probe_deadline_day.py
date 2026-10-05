"""Run every deadline-protocol policy on one test day (smoke test and timing)."""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.request_world_model import RequestWorldModel
from scripts.burst_instances import load_or_create_bank
from scripts.deadline_protocol import build_policies, simulate

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    episode = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    instances = load_or_create_bank(seed, 20)
    train = instances[:12]
    policies, _stats = build_policies(
        seed, RequestWorldModel.fit(train), RequestWorldModel.fit(train, spatial="uniform")
    )
    for name, policy in policies.items():
        started = time.perf_counter()
        r = simulate(instances[episode], policy)
        print(
            f"{name:24s} cost {r['cost']:.3f}  finish {r['finish']:.2f}  response {r['mean_response']:.2f}"
            f"  feasible {r['feasible']}  waits {r['waits']}  {time.perf_counter() - started:.0f}s",
            flush=True,
        )
