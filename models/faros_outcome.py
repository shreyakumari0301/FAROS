"""Exact H=1, set-conditioned branch regret and harmful-regret probability."""
import torch
from torch import nn
from torch.nn import functional as F
from models.regret_state_encoder import RegretStateEncoder


class FAROSOutcome(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = RegretStateEncoder(width=64)
        self.state.head = nn.Identity()  # Reuse architecture, expose state features.
        self.project = nn.Linear(64*5+4, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0., batch_first=True)
        self.candidates = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.regret = nn.Linear(64, 1)
        self.risk = nn.Linear(64, 1)

    def forward(self, nodes, context, valid):
        if valid.ndim != 2 or not valid.any(1).all():
            raise ValueError('Each decision requires at least one valid candidate')
        # Do not encode padding: even NaN padding cannot contaminate valid scores.
        latent = nodes.new_zeros((*valid.shape, 64))
        latent[valid] = self.project(self.state(nodes[valid], context[valid]))
        latent = self.candidates(latent, src_key_padding_mask=~valid)
        r = F.softplus(self.regret(latent).squeeze(-1))
        logits = self.risk(latent).squeeze(-1)
        return r.masked_fill(~valid, 0.), logits.masked_fill(~valid, 0.)


def selection_score(regret, risk_logits, valid, beta):
    return (regret + beta*risk_logits.sigmoid()).masked_fill(~valid, float('inf'))


def outcome_loss(regret, risk_logits, costs, valid, cfg):
    safe = costs.masked_fill(~valid, float('inf'))
    target = (safe-safe.min(1, keepdim=True).values).masked_fill(~valid, 0.)
    target_logits = (-target/cfg['tau']).masked_fill(~valid, -torch.inf)
    probabilities = target_logits.softmax(1)
    log_prob = (-regret/cfg['tau']).masked_fill(~valid, -torch.inf).log_softmax(1)
    kl = (probabilities*(probabilities.clamp_min(1e-30).log()-log_prob.masked_fill(~valid, 0.))).sum(1)
    huber = F.smooth_l1_loss(regret, target, reduction='none')*(1+cfg['alpha']*target)
    labels = (target > cfg['delta']).to(regret.dtype)
    risk = F.binary_cross_entropy_with_logits(risk_logits, labels, reduction='none')
    count = valid.sum(1)
    regression = (huber.masked_fill(~valid, 0.).sum(1)/count).mean()
    bce = (risk.masked_fill(~valid, 0.).sum(1)/count).mean()
    return kl.mean()+cfg['lambda']*regression+cfg['gamma']*bce
