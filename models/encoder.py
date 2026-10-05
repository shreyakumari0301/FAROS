"""Shared state encoder for later baselines (Q / JEPA / FAROS).

Phase 0: stub only. Do not implement FAROS here.
Torch will be required when this module is implemented in Phase 2.
"""

from __future__ import annotations


class CVRPEncoder:
    def __init__(self, d_model: int = 128, node_dim: int = 6):
        self.d_model = d_model
        self.node_dim = node_dim

    def forward(self, node_features):
        """Placeholder forward.

        node_features: [batch, n_nodes, node_dim]
        returns (z_state, node_embeddings)
        """
        raise NotImplementedError("Encoder is implemented in Phase 2")
