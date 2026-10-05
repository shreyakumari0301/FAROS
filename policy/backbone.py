from __future__ import annotations

from typing import Optional

import numpy as np

from env.state import CVRPState


class RoutingPolicy:
    """Frozen routing-policy interface.

    FAROS must not train this model. Plug MnLP/POMO behind this API.
    logits(state) -> [n_actions] scores for next-action selection.
    """

    def __init__(self, model=None):
        self.model = model

    def logits(self, state: CVRPState) -> np.ndarray:
        raise NotImplementedError


class NearestNeighborPolicy(RoutingPolicy):
    """Temporary Phase-0 engineering stand-in (not for parity / paper numbers)."""

    def logits(self, state: CVRPState) -> np.ndarray:
        current = state.coords[state.current_node]
        dists = np.linalg.norm(state.coords - current, ord=2, axis=-1)
        return -dists.astype(np.float64)


def load_policy(
    name: str = "nearest_neighbor",
    checkpoint: Optional[str] = None,
    **kwargs,
) -> RoutingPolicy:
    """Factory for Phase 0.

    name:
      - nearest_neighbor / nn / heuristic
      - mnlp / lehd
    """
    if name in ("nearest_neighbor", "nn", "heuristic"):
        return NearestNeighborPolicy()

    if name in ("mnlp", "lehd"):
        from policy.mnlp_adapter import MnLPPolicy

        return MnLPPolicy(checkpoint=checkpoint, **kwargs)

    raise NotImplementedError(
        f"Policy '{name}' is not wired yet. "
        "Use 'mnlp' (LaRP/MnLP CVRP200) or 'nearest_neighbor'."
    )
