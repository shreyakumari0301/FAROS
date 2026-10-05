from __future__ import annotations

from typing import List

import numpy as np

from .state import CVRPState


class CVRPEnv:
    """Exact CVRP simulator.

    Depot action is explicit: action == 0 returns to depot and resets capacity.
    Distance is Euclidean L2, matching standard CVRP benchmarks.
    """

    def __init__(
        self,
        coords: np.ndarray,
        demands: np.ndarray,
        capacity: float,
    ):
        coords = np.asarray(coords, dtype=np.float64)
        demands = np.asarray(demands, dtype=np.float64)

        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be [N, 2], got {coords.shape}")
        if demands.ndim != 1 or demands.shape[0] != coords.shape[0]:
            raise ValueError("demands must be [N] matching coords")
        if float(demands[0]) != 0.0:
            raise ValueError("demands[0] (depot) must be 0")

        self.coords = coords
        self.demands = demands
        self.capacity = float(capacity)
        self.n_nodes = int(coords.shape[0])

    def reset(self) -> CVRPState:
        visited = np.zeros(self.n_nodes, dtype=bool)
        visited[0] = True
        return CVRPState(
            coords=self.coords.copy(),
            demands=self.demands.copy(),
            visited=visited,
            current_node=0,
            remaining_capacity=self.capacity,
            total_cost=0.0,
            done=False,
            routes_used=0,
        )

    def distance(self, i: int, j: int) -> float:
        p1 = self.coords[i]
        p2 = self.coords[j]
        return float(np.linalg.norm(p1 - p2, ord=2))

    def feasible_customers(self, state: CVRPState) -> List[int]:
        feasible: List[int] = []
        for node in range(1, self.n_nodes):
            if state.visited[node]:
                continue
            if float(state.demands[node]) <= state.remaining_capacity:
                feasible.append(node)
        return feasible

    def feasible_actions(self, state: CVRPState) -> List[int]:
        """Return valid next nodes, including depot when allowed.

        Rule (fixed for Phase 0, do not change silently):
        - Unvisited customers with demand <= remaining capacity are feasible.
        - Depot (0) is feasible iff current node is not the depot and either
          the open route has served a customer, or no customers are feasible
          while work remains.
        - If at depot, depot is never selected again.
        """
        if state.done:
            return []

        customers = self.feasible_customers(state)
        actions = list(customers)

        at_depot = state.current_node == 0
        all_customers_visited = bool(np.all(state.visited[1:]))

        if at_depot:
            return actions

        open_route = state.remaining_capacity < self.capacity
        must_return = len(customers) == 0 and not all_customers_visited
        if open_route or must_return:
            actions.append(0)

        return actions

    def feasibility_mask(self, state: CVRPState) -> np.ndarray:
        mask = np.zeros(self.n_nodes, dtype=bool)
        for a in self.feasible_actions(state):
            mask[a] = True
        return mask

    def step(self, state: CVRPState, action: int) -> CVRPState:
        if state.done:
            raise ValueError("Cannot step a done state")

        action = int(action)
        if action < 0 or action >= self.n_nodes:
            raise ValueError(f"Invalid action {action}")

        new_state = CVRPState(
            coords=state.coords,
            demands=state.demands,
            visited=state.visited.copy(),
            current_node=state.current_node,
            remaining_capacity=state.remaining_capacity,
            total_cost=state.total_cost,
            done=state.done,
            routes_used=state.routes_used,
        )

        travel_cost = self.distance(state.current_node, action)
        new_state.total_cost += travel_cost

        if action != 0:
            demand = float(state.demands[action])

            if bool(new_state.visited[action]):
                raise ValueError(f"Customer {action} already visited")

            if demand > new_state.remaining_capacity:
                raise ValueError(f"Capacity violation for customer {action}")

            new_state.visited[action] = True
            new_state.remaining_capacity -= demand
            new_state.current_node = action
        else:
            if state.current_node == 0:
                raise ValueError("Already at depot; cannot select depot again")
            new_state.current_node = 0
            new_state.remaining_capacity = self.capacity
            new_state.routes_used += 1

        all_visited = bool(np.all(new_state.visited[1:]))
        if all_visited:
            if new_state.current_node != 0:
                new_state.total_cost += self.distance(new_state.current_node, 0)
                new_state.current_node = 0
                new_state.remaining_capacity = self.capacity
                new_state.routes_used += 1
            new_state.done = True

        return new_state
