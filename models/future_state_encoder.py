"""Small actual-state ranking probe; this is not a FAROS predictor."""
import torch
from torch import nn


class FutureStateEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.embed = nn.Linear(8, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.0,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(width * 2 + 4, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, nodes, context):
        # No positional embeddings: node order must not determine the score.
        z = self.encoder(self.embed(nodes))
        remaining = (1 - nodes[..., 3:4])
        pooled = (z * remaining).sum(1) / remaining.sum(1).clamp_min(1)
        return self.head(torch.cat((z[:, 0], pooled, context), -1)).squeeze(-1)
