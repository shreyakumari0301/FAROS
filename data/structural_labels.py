"""Structural consequence labels for future CVRP states (Phase 1)."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Sequence

import numpy as np

# Ordered feature vector used for branch-divergence analysis.
PHI_KEYS: Sequence[str] = (
    "remaining_capacity_ratio",  # capacity slack S_Q
    "remaining_demand_ratio",
    "fragmentation",
    "depot_distance",
    "routes_lower_bound",
)


def structural_labels(
    coords: np.ndarray,
    demands: np.ndarray,
    visited: np.ndarray,
    remaining_capacity: float,
    capacity: float,
    prefix_cost: float,
) -> Dict[str, float]:
    """Structural features of the remaining optimization problem.

    coords/demands/visited include depot at index 0.
    """
    coords = np.asarray(coords, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    visited = np.asarray(visited, dtype=bool)

    n_customers = int(coords.shape[0] - 1)
    remaining_mask = ~visited.copy()
    remaining_mask[0] = False
    remaining_idx = np.where(remaining_mask)[0]
    n_rem = int(remaining_idx.size)

    total_demand = float(demands[1:].sum())
    rem_demand = float(demands[remaining_idx].sum()) if n_rem > 0 else 0.0

    r_n = n_rem / float(n_customers) if n_customers > 0 else 0.0
    r_d = rem_demand / total_demand if total_demand > 0 else 0.0
    # Capacity slack S_Q = rem_cap / Q
    s_q = float(remaining_capacity) / float(capacity) if capacity > 0 else 0.0
    # Crude remaining-route lower bound
    r_min = float(ceil(rem_demand / float(capacity))) if capacity > 0 and rem_demand > 0 else 0.0

    if n_rem == 0:
        fragmentation = 0.0
        depot_distance = 0.0
    elif n_rem == 1:
        fragmentation = 0.0
        depot_distance = float(np.linalg.norm(coords[remaining_idx[0]] - coords[0], ord=2))
    else:
        pts = coords[remaining_idx]
        dsum = 0.0
        for i in range(n_rem):
            diffs = pts - pts[i]
            dist = np.sqrt((diffs * diffs).sum(axis=1) + 1e-12)
            dist[i] = np.inf
            dsum += float(dist.min())
        fragmentation = dsum / float(n_rem)
        depot_distance = float(
            np.mean(np.linalg.norm(pts - coords[0], ord=2, axis=1))
        )

    return {
        "remaining_customer_ratio": float(r_n),
        "remaining_demand_ratio": float(r_d),
        "remaining_capacity_ratio": float(s_q),
        "capacity_slack": float(s_q),
        "capacity_pressure": float(1.0 - s_q),
        "fragmentation": float(fragmentation),
        "depot_distance": float(depot_distance),
        "routes_lower_bound": float(r_min),
        "remaining_customers": float(n_rem),
        "remaining_demand": float(rem_demand),
        "prefix_cost": float(prefix_cost),
    }


def phi_vector(labels: Dict[str, float]) -> List[float]:
    """Compact structural embedding for divergence comparisons."""
    return [float(labels.get(k, 0.0)) for k in PHI_KEYS]
