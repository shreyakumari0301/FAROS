from __future__ import annotations

import random
from typing import Optional


def set_seed(seed: int) -> None:
    random.seed(seed)
    # NumPy / Torch RNG modules may be blocked by Windows App Control.
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
