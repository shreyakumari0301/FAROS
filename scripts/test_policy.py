"""Policy interface + top-K feasibility smoke test."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.generate import generate_cvrp
from env.cvrp_env import CVRPEnv
from policy.backbone import load_policy
from policy.topk import topk_feasible
from utils.seed import set_seed


def main() -> None:
    set_seed(0)
    inst = generate_cvrp(n_customers=20, capacity=50, seed=0)
    env = CVRPEnv(inst["coords"], inst["demands"], inst["capacity"])
    policy = load_policy("nearest_neighbor")
    state = env.reset()

    logits = policy.logits(state)
    assert logits.shape == (21,), logits.shape

    candidates = topk_feasible(logits, state, k=5, env=env)
    assert 1 <= len(candidates) <= 5
    assert all(c["action"] != 0 for c in candidates), "at depot, first picks are customers"

    # Step until a return-to-depot becomes necessary / available
    for _ in range(50):
        if state.done:
            break
        logits = policy.logits(state)
        cands = topk_feasible(logits, state, k=5, env=env)
        assert cands, "must have at least one feasible action"
        state = env.step(state, cands[0]["action"])

    assert state.done
    print("logits shape:", tuple(logits.shape))
    print("final cost:", round(state.total_cost, 4))
    print("test_policy: OK")


if __name__ == "__main__":
    main()
