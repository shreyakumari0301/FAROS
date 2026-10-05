"""Learned world model of request arrivals for the burst dynamic CVRP.

Routing transitions are exact and need no model; what the planner lacks is
when hidden customers will request service. This model learns that from
past days only (locations and realized reveal times), then tracks a
posterior over each wave's start during the day:

  * waves     -- customers grouped by location (single-linkage components);
  * width W   -- wave duration, pooled unbiased estimate over training waves;
  * prior     -- kernel density over estimated wave starts in training days;
  * posterior -- prior x P(observed reveals) x P(hidden still hidden at t);
  * imagine   -- sample a start per wave, then hidden reveals uniform on
                 [max(start, t), start + W].

It never reads ``window_start``, ``window_width`` or hidden ``reveal``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

LINK = 0.1
GRID_STEP = 0.05
BANDWIDTH = 0.5
PRIOR_FLOOR = 1e-4


def spatial_groups(inst: dict, link: float = LINK) -> np.ndarray:
    """Label customers by connected component of the ``dist < link`` graph."""
    n_nodes = inst["demands"].shape[0]
    close = inst["dist"] < link
    labels = np.full(n_nodes, -1, dtype=np.int64)
    current = 0
    for seed_node in range(1, n_nodes):
        if labels[seed_node] >= 0:
            continue
        stack = [seed_node]
        labels[seed_node] = current
        while stack:
            node = stack.pop()
            for nb in np.flatnonzero(close[node]):
                if nb != 0 and labels[nb] < 0:
                    labels[nb] = current
                    stack.append(int(nb))
        current += 1
    return labels


@dataclass
class ArrivalWorldModel:
    width: float
    grid: np.ndarray
    log_prior: np.ndarray
    horizon: float
    link: float = LINK
    fitted_waves: int = 0
    _groups: Dict[int, tuple] = field(default_factory=dict, repr=False)

    @classmethod
    def fit(
        cls,
        train_instances: Sequence[dict],
        link: float = LINK,
        grid_step: float = GRID_STEP,
        bandwidth: float = BANDWIDTH,
        flat_prior: bool = False,
    ) -> "ArrivalWorldModel":
        widths, waves = [], []
        for inst in train_instances:
            labels = spatial_groups(inst, link)
            for g in range(int(labels.max()) + 1):
                reveal = inst["reveal"][labels == g]
                if reveal.size < 3:
                    continue
                span = float(reveal.max() - reveal.min())
                widths.append(span * (reveal.size + 1) / (reveal.size - 1))
                waves.append((float(reveal.min()), span, reveal.size))
        if not widths:
            raise ValueError("no training waves with >= 3 customers")
        width = float(np.median(widths))
        starts = np.array([lo - span / (n - 1) for lo, span, n in waves])
        horizon = float(train_instances[0]["horizon"])
        grid = np.arange(-width, horizon + grid_step, grid_step)
        if flat_prior:
            density = np.where((grid >= 0.0) & (grid <= horizon - width), 1.0, 0.0)
        else:
            z = (grid[:, None] - starts[None, :]) / bandwidth
            density = np.exp(-0.5 * z * z).sum(axis=1)
        density = density / density.sum() + PRIOR_FLOOR / grid.size
        return cls(
            width=width,
            grid=grid,
            log_prior=np.log(density / density.sum()),
            horizon=horizon,
            link=link,
            fitted_waves=len(waves),
        )

    def _labels(self, inst: dict) -> np.ndarray:
        key = id(inst)
        cached = self._groups.get(key)
        if cached is None or cached[0] is not inst:
            cached = (inst, spatial_groups(inst, self.link))
            self._groups[key] = cached
        return cached[1]

    def _posterior(self, observed: np.ndarray, n_hidden: int, time: float):
        """Return (weights over grid, effective width) for one wave."""
        log_post = self.log_prior.copy()
        width = self.width
        if observed.size:
            lo, hi = float(observed.min()), float(observed.max())
            width = max(width, hi - lo + 1e-6)
            log_post[(self.grid > lo + 1e-9) | (self.grid + width < hi - 1e-9)] = -np.inf
        if n_hidden:
            survive = np.clip((self.grid + width - time) / width, 0.0, 1.0)
            with np.errstate(divide="ignore"):
                log_post += n_hidden * np.log(survive)
        if not np.isfinite(log_post).any():
            anchor = float(observed.min()) if observed.size else time
            weights = np.zeros_like(self.grid)
            weights[int(np.argmin(np.abs(self.grid - anchor)))] = 1.0
            return weights, width
        weights = np.exp(log_post - log_post.max())
        return weights / weights.sum(), width

    def _hidden_waves(self, inst: dict, time: float):
        labels = self._labels(inst)
        known = inst["reveal"] <= time + 1e-9
        known[0] = True
        for g in np.unique(labels[~known]):
            members = labels == g
            hidden = np.flatnonzero(members & ~known)
            observed = inst["reveal"][members & known]
            yield int(g), hidden, observed

    def wave_of(self, inst: dict, node: int) -> int:
        return int(self._labels(inst)[node])

    def expected_next_reveal(self, inst: dict, node: int, time: float) -> Optional[float]:
        """Expected first reveal among hidden customers in ``node``'s wave."""
        labels = self._labels(inst)
        known = inst["reveal"] <= time + 1e-9
        members = labels == labels[node]
        members[0] = False
        hidden = np.flatnonzero(members & ~known)
        if hidden.size == 0:
            return None
        weights, width = self._posterior(inst["reveal"][members & known], hidden.size, time)
        low = np.maximum(self.grid, time)
        span = np.maximum(self.grid + width - low, 0.0)
        return float((weights * (low + span / (hidden.size + 1))).sum())

    def imagine(self, inst: dict, time: float, batch: int, rng: random.Random) -> np.ndarray:
        """Sample ``batch`` reveal-time vectors consistent with what is known at ``time``."""
        gen = np.random.default_rng(rng.getrandbits(63))
        reveal = np.repeat(inst["reveal"][None, :], batch, axis=0)
        for _g, hidden, observed in self._hidden_waves(inst, time):
            weights, width = self._posterior(observed, hidden.size, time)
            starts = self.grid[gen.choice(self.grid.size, size=batch, p=weights)]
            low = np.maximum(starts, time)
            span = np.maximum(starts + width - low, 0.0)
            draws = gen.random((batch, hidden.size))
            reveal[:, hidden] = low[:, None] + draws * span[:, None]
        return reveal

    def start_error(self, inst: dict, time: float) -> List[float]:
        """|posterior mean start - true start| per wave; diagnostics only."""
        out = []
        for _g, hidden, observed in self._hidden_waves(inst, time):
            weights, _ = self._posterior(observed, hidden.size, time)
            true_start = float(inst["window_start"][int(hidden[0])])
            out.append(abs(float((weights * self.grid).sum()) - true_start))
        return out
