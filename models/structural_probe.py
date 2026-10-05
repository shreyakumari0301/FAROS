"""Nonlinear baseline using structural summaries and matched cost context."""
from torch import nn


class StructuralProbe(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)
