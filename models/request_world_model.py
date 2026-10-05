"""Learned generative world model of future requests (locations unknown).

The planner sees only requests revealed so far. From past days the model
learns the structure of a day -- how many waves, how large, how long, when
they start, how tight they are, how far apart their centres lie, and the
demand distribution. During the day it groups revealed requests into
waves and imagines the missing ones:

  * started wave  -- remaining size ~ learned sizes given what is seen;
                     start ~ posterior from its observed reveal times;
                     locations ~ N(estimated centre, learned spread);
  * unseen wave   -- centre ~ learned region, repelled from other centres
                     by the learned minimum gap; start ~ prior given that
                     no member has appeared yet.

Every imagined future is a full set of (location, demand, reveal time)
for customers the planner has not seen. ``spatial="uniform"`` drops the
learned spatial structure and places imagined customers uniformly in the
learned region (ablation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from models.arrival_world_model import BANDWIDTH, GRID_STEP, LINK, PRIOR_FLOOR

MAX_CENTRE_TRIES = 200


def _components(coords: np.ndarray, link: float) -> np.ndarray:
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    diff = coords[:, None, :] - coords[None, :, :]
    close = (diff * diff).sum(axis=-1) < link * link
    labels = np.full(n, -1, dtype=np.int64)
    current = 0
    for seed_node in range(n):
        if labels[seed_node] >= 0:
            continue
        stack = [seed_node]
        labels[seed_node] = current
        while stack:
            node = stack.pop()
            for nb in np.flatnonzero(close[node] & (labels < 0)):
                labels[nb] = current
                stack.append(int(nb))
        current += 1
    return labels


@dataclass
class RequestWorldModel:
    width: float
    grid: np.ndarray
    log_prior: np.ndarray
    sizes: np.ndarray
    waves_per_day: np.ndarray
    spread: float
    min_gap: float
    region: np.ndarray
    demand_values: np.ndarray
    spatial: str = "learned"
    link: float = LINK

    @classmethod
    def fit(
        cls,
        train_instances: Sequence[dict],
        spatial: str = "learned",
        link: float = LINK,
        grid_step: float = GRID_STEP,
        bandwidth: float = BANDWIDTH,
    ) -> "RequestWorldModel":
        widths, waves, sizes, counts, offsets, gaps, centres, demands = [], [], [], [], [], [], [], []
        for inst in train_instances:
            coords = inst["coords"][1:]
            reveal = inst["reveal"][1:]
            labels = _components(coords, link)
            day_centres = []
            for g in range(int(labels.max()) + 1):
                members = labels == g
                r = reveal[members]
                xy = coords[members]
                sizes.append(int(members.sum()))
                centre = xy.mean(axis=0)
                day_centres.append(centre)
                offsets.append(xy - centre)
                if r.size >= 3:
                    span = float(r.max() - r.min())
                    widths.append(span * (r.size + 1) / (r.size - 1))
                    waves.append((float(r.min()), span, r.size))
            counts.append(len(day_centres))
            day_centres = np.array(day_centres)
            centres.append(day_centres)
            if len(day_centres) > 1:
                d = np.sqrt(((day_centres[:, None] - day_centres[None]) ** 2).sum(-1))
                gaps.append(float(d[np.triu_indices(len(day_centres), 1)].min()))
            demands.append(inst["demands"][1:])
        width = float(np.median(widths))
        starts = np.array([lo - span / (n - 1) for lo, span, n in waves])
        horizon = float(train_instances[0]["horizon"])
        grid = np.arange(-width, horizon + grid_step, grid_step)
        z = (grid[:, None] - starts[None, :]) / bandwidth
        density = np.exp(-0.5 * z * z).sum(axis=1)
        density = density / density.sum() + PRIOR_FLOOR / grid.size
        all_centres = np.concatenate(centres)
        return cls(
            width=width,
            grid=grid,
            log_prior=np.log(density / density.sum()),
            sizes=np.array(sizes),
            waves_per_day=np.array(counts),
            spread=float(np.concatenate(offsets).std()),
            min_gap=float(min(gaps)) if gaps else 0.0,
            region=np.array([all_centres.min(axis=0), all_centres.max(axis=0)]),
            demand_values=np.concatenate(demands),
            spatial=spatial,
            link=link,
        )

    # ---- in-day belief ---------------------------------------------------

    def observe(self, coords: np.ndarray, reveal: np.ndarray, time: float = 0.0):
        """Group revealed customers (non-depot rows) into seen waves."""
        labels = _components(coords, self.link)
        waves = []
        for g in range(int(labels.max()) + 1 if labels.size else 0):
            members = np.flatnonzero(labels == g)
            waves.append(
                {
                    "members": members,
                    "centre": coords[members].mean(axis=0),
                    "reveals": reveal[members],
                }
            )
        return labels, waves

    def _posterior(self, observed: np.ndarray, n_hidden: int, time: float):
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
            weights = np.zeros_like(self.grid)
            anchor = float(observed.min()) if observed.size else time
            weights[int(np.argmin(np.abs(self.grid - anchor)))] = 1.0
            return weights, width
        weights = np.exp(log_post - log_post.max())
        return weights / weights.sum(), width

    def _remaining_sizes(self, n_obs: int) -> np.ndarray:
        options = self.sizes[self.sizes >= n_obs]
        return options - n_obs if options.size else np.zeros(1, dtype=np.int64)

    def expected_hidden(self, waves) -> float:
        seen = sum(float(self._remaining_sizes(w["members"].size).mean()) for w in waves)
        extra = self.waves_per_day[self.waves_per_day >= len(waves)]
        unseen = float((extra - len(waves)).mean()) * float(self.sizes.mean()) if extra.size else 0.0
        return seen + unseen

    def expected_next_reveal(self, wave: dict, time: float) -> Optional[float]:
        n_hidden = int(round(float(self._remaining_sizes(wave["members"].size).mean())))
        if n_hidden <= 0:
            return None
        weights, width = self._posterior(wave["reveals"], n_hidden, time)
        low = np.maximum(self.grid, time)
        span = np.maximum(self.grid + width - low, 0.0)
        return float((weights * (low + span / (n_hidden + 1))).sum())

    # ---- imagination -----------------------------------------------------

    def _centre(self, gen, taken):
        lo, hi = self.region
        point = lo + gen.random(2) * (hi - lo)
        for _ in range(MAX_CENTRE_TRIES):
            if all(float(np.hypot(*(point - c))) >= self.min_gap for c in taken):
                break
            point = lo + gen.random(2) * (hi - lo)
        return point

    def imagine(self, waves, time: float, batch: int, rng: random.Random):
        """Return per-future lists of (coords [m,2], demands [m], reveal [m])."""
        gen = np.random.default_rng(rng.getrandbits(63))
        cache = {}

        def start_sampler(key, observed, n_hidden):
            if (key, n_hidden) not in cache:
                cache[(key, n_hidden)] = self._posterior(observed, n_hidden, time)
            weights, width = cache[(key, n_hidden)]
            return float(self.grid[gen.choice(self.grid.size, p=weights)]), width

        lo, hi = self.region
        futures = []
        for _ in range(batch):
            xy, dem, rev = [], [], []

            def add(centre, n_new, start, width):
                if n_new <= 0:
                    return
                if self.spatial == "uniform":
                    pts = lo + gen.random((n_new, 2)) * (hi - lo)
                else:
                    pts = centre + gen.normal(0.0, self.spread, (n_new, 2))
                low = max(start, time)
                xy.append(pts)
                dem.append(gen.choice(self.demand_values, n_new))
                rev.append(low + gen.random(n_new) * max(start + width - low, 0.0))

            taken = [w["centre"] for w in waves]
            for key, wave in enumerate(waves):
                n_obs = wave["members"].size
                n_hidden = int(gen.choice(self._remaining_sizes(n_obs)))
                if n_hidden:
                    centre = wave["centre"] + gen.normal(0.0, self.spread / np.sqrt(n_obs), 2)
                    start, width = start_sampler(key, wave["reveals"], n_hidden)
                    add(centre, n_hidden, start, width)
            extra = self.waves_per_day[self.waves_per_day >= len(waves)]
            n_unseen = int(gen.choice(extra)) - len(waves) if extra.size else 0
            for _u in range(n_unseen):
                size = int(gen.choice(self.sizes))
                centre = self._centre(gen, taken)
                taken.append(centre)
                start, width = start_sampler(-1, np.zeros(0), size)
                add(centre, size, start, width)
            if xy:
                futures.append((np.concatenate(xy), np.concatenate(dem), np.concatenate(rev)))
            else:
                futures.append((np.zeros((0, 2)), np.zeros(0), np.zeros(0)))
        return futures
