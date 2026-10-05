from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CVRPState:
    """Exact CVRP decision-state for FAROS Phase 0.

    Node 0 is the depot; nodes 1..N are customers.
    For CVRP200: coords [201, 2], demands [201].

    Phase 0 uses NumPy so the simulator runs without Torch.
    MnLP/POMO integration can convert at the policy boundary.
    """

    coords: np.ndarray
    demands: np.ndarray
    visited: np.ndarray
    current_node: int
    remaining_capacity: float
    total_cost: float
    done: bool = False
    routes_used: int = 0

    def clone(self) -> "CVRPState":
        return CVRPState(
            coords=self.coords.copy(),
            demands=self.demands.copy(),
            visited=self.visited.copy(),
            current_node=self.current_node,
            remaining_capacity=self.remaining_capacity,
            total_cost=self.total_cost,
            done=self.done,
            routes_used=self.routes_used,
        )
