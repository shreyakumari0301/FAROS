from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np


def generate_cvrp(
    n_customers: int = 200,
    capacity: float = 50.0,
    seed: Optional[int] = None,
    demand_low: int = 1,
    demand_high: int = 10,
) -> Dict[str, object]:
    """Temporary uniform CVRP generator for engineering tests.

    Uses stdlib random (not numpy.random) so it works under Windows App Control
    policies that block some NumPy RNG DLLs.

    Replace later with the exact instance convention used by the MnLP/POMO
    baseline so costs match published/eval numbers.
    """
    rng = random.Random(seed)

    coords = np.array(
        [[rng.random(), rng.random()] for _ in range(n_customers + 1)],
        dtype=np.float64,
    )
    demands = np.array(
        [0.0]
        + [float(rng.randrange(demand_low, demand_high)) for _ in range(n_customers)],
        dtype=np.float64,
    )

    return {
        "coords": coords,
        "demands": demands,
        "capacity": float(capacity),
        "n_customers": int(n_customers),
    }
