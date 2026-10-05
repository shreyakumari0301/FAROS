"""H=8 state probe with context-conditioned attention and explicit current node."""
import torch
from torch import nn


class RegretStateEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.embed = nn.Linear(8, width)
        self.context = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3)
        self.pool = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.head = nn.Sequential(nn.Linear(width * 5 + 4, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, nodes, context):
        ctx = self.context(context)
        z = self.encoder(self.embed(nodes) + ctx[:, None])
        depot = (z * nodes[..., 4:5]).sum(1)
        current = (z * nodes[..., 5:6]).sum(1)
        remaining = 1 - nodes[..., 3:4]
        mean = (z * remaining).sum(1) / remaining.sum(1).clamp_min(1)
        # Keep the depot available, including for terminal states.
        mask = (nodes[..., 3] > .5) & (nodes[..., 4] < .5)
        attended, _ = self.pool((current + ctx)[:, None], z, z,
                                key_padding_mask=mask, need_weights=False)
        return self.head(torch.cat((depot, current, mean, attended[:, 0], ctx, context), -1)).squeeze(-1)
