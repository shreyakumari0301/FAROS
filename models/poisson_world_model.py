"""Learned world model for the Larsen et al. (2002) partially dynamic setting.

From past days it learns the expected daily total E{n} and the service
region. At time t it reads today's advance requests n_adv (those revealed
at time 0), infers the immediate-request rate (E{n} - n_adv) / T, and
imagines the remaining requests: a Poisson count on (t, T], uniform times,
uniform locations in the learned region, unit demand.

Exposes the same interface as ``RequestWorldModel`` so the Level B
policies run unchanged. There are no waves, so the waiting rule never
fires and FAROS reduces to sampling at forks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PoissonWorldModel:
    expected_total: float
    horizon: float
    region: np.ndarray

    @classmethod
    def fit(cls, train_instances: Sequence[dict]) -> "PoissonWorldModel":
        totals = [inst["demands"].shape[0] - 1 for inst in train_instances]
        coords = np.concatenate([inst["coords"][1:] for inst in train_instances])
        return cls(
            expected_total=float(np.mean(totals)),
            horizon=float(train_instances[0]["horizon"]),
            region=np.array([coords.min(axis=0), coords.max(axis=0)]),
        )

    def observe(self, coords: np.ndarray, reveal: np.ndarray, time: float = 0.0):
        n_adv = int((reveal <= 1e-9).sum())
        wave = {
            "members": np.arange(coords.shape[0]),
            "centre": coords.mean(axis=0) if coords.size else self.region.mean(axis=0),
            "reveals": reveal,
            "time": float(time),
            "rate": max(self.expected_total - n_adv, 0.0) / self.horizon,
        }
        return np.zeros(coords.shape[0], dtype=np.int64), [wave]

    def expected_hidden(self, waves) -> float:
        wave = waves[0]
        return wave["rate"] * max(self.horizon - wave["time"], 0.0)

    def expected_next_reveal(self, wave: dict, time: float):
        return None

    def imagine(self, waves, time: float, batch: int, rng: random.Random):
        gen = np.random.default_rng(rng.getrandbits(63))
        mean = self.expected_hidden(waves)
        lo, hi = self.region
        futures = []
        for count in gen.poisson(mean, batch):
            futures.append(
                (
                    lo + gen.random((count, 2)) * (hi - lo),
                    np.ones(count),
                    time + gen.random(count) * max(self.horizon - time, 0.0),
                )
            )
        return futures
