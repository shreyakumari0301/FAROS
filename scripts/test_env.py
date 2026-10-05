"""Manual CVRP env smoke test (3 customers)."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.cvrp_env import CVRPEnv


def main() -> None:
    coords = np.array(
        [
            [0.0, 0.0],  # depot
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    demands = np.array([0.0, 2.0, 2.0, 3.0], dtype=np.float64)
    capacity = 4.0

    env = CVRPEnv(coords, demands, capacity)
    state = env.reset()

    print("Manual tour: 1 -> 2 -> 0 -> 3 (-> auto depot)")
    print("action  current  rem_cap   cost")

    for action in [1, 2, 0, 3]:
        state = env.step(state, action)
        print(
            f"{action:6d}  {state.current_node:7d}  "
            f"{state.remaining_capacity:7.1f}  {state.total_cost:8.4f}"
        )

    print("Done:", state.done)
    print("Routes used:", state.routes_used)
    print("All visited:", bool(np.all(state.visited[1:])))

    # Expected path cost:
    # 0->1 = 1, 1->2 = 1, 2->0 = 2, 0->3 = 2, 3->0 = 2  => 8.0
    expected = 8.0
    assert state.done, "tour should be finished"
    assert abs(state.total_cost - expected) < 1e-9, (
        f"cost {state.total_cost} != expected {expected}"
    )
    assert state.routes_used == 2

    # Capacity violation must raise
    state2 = env.reset()
    state2 = env.step(state2, 1)
    try:
        env.step(state2, 3)  # demand 3 > rem_cap 2
        raise AssertionError("expected capacity violation")
    except ValueError as exc:
        assert "Capacity violation" in str(exc)

    # Clone independence
    state3 = env.reset()
    clone = state3.clone()
    clone.visited[1] = True
    assert not bool(state3.visited[1]), "clone must not alias visited"

    print("test_env: OK")


if __name__ == "__main__":
    main()
