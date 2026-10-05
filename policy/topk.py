from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from env.cvrp_env import CVRPEnv
from env.state import CVRPState


def topk_feasible(
    logits: np.ndarray,
    state: CVRPState,
    k: int = 5,
    env: Optional[CVRPEnv] = None,
) -> List[Dict[str, float]]:
    """Return top-K feasible actions under a feasibility mask.

    Prefer passing `env` so depot rules match CVRPEnv.feasible_actions.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"logits must be [n_nodes], got {logits.shape}")

    masked = logits.copy()

    if env is not None:
        feasible = env.feasibility_mask(state)
        masked[~feasible] = -np.inf
    else:
        masked[state.visited] = -np.inf
        for node in range(1, masked.size):
            if float(state.demands[node]) > state.remaining_capacity:
                masked[node] = -np.inf
        if state.current_node == 0:
            masked[0] = -np.inf

    finite = np.isfinite(masked)
    n_avail = int(finite.sum())
    if n_avail == 0:
        return []

    kk = min(int(k), n_avail)
    # argpartition then sort the top-k slice for stable ordering
    idx = np.argpartition(-masked, kk - 1)[:kk]
    idx = idx[np.argsort(-masked[idx])]

    return [
        {
            "action": int(idx[i]),
            "score": float(masked[idx[i]]),
        }
        for i in range(kk)
    ]
