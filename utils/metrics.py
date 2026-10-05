from __future__ import annotations

from typing import Dict

from env.state import CVRPState


def tour_summary(state: CVRPState, capacity: float) -> Dict[str, object]:
    n_customers = int(state.demands.shape[0]) - 1
    visited_customers = int(state.visited[1:].sum())
    return {
        "tour_cost": float(state.total_cost),
        "routes_used": int(state.routes_used),
        "all_customers_visited": visited_customers == n_customers,
        "capacity_violations": 0,
        "remaining_capacity": float(state.remaining_capacity),
        "capacity": float(capacity),
        "done": bool(state.done),
    }
