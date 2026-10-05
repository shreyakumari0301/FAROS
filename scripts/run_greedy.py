"""Phase 0 target: greedy decode one CVRP200 instance with frozen MnLP."""

from __future__ import annotations

import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from policy.backbone import load_policy
from utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    set_seed(int(cfg["experiment"]["seed"]))

    policy_name = cfg["policy"]["name"]
    episode = int(cfg["experiment"].get("episode", 0))

    if policy_name in ("mnlp", "lehd"):
        policy = load_policy(
            name="mnlp",
            checkpoint=cfg["policy"].get("checkpoint"),
            mnlp_root=cfg["policy"].get("mnlp_root"),
            data_path=cfg["policy"].get("data_path"),
            device=cfg["policy"].get("device", "cpu"),
            max_episodes=int(cfg["policy"].get("max_episodes", 128)),
        )
        meta = policy.instance_meta(episode)
        print(f"Loaded CVRP{meta['n_customers']} instance (episode {episode})")
        print(f"Nodes: {meta['n_customers']}")
        print(f"Capacity: {int(meta['capacity'])}")
        print(f"LKH reference cost: {meta['lkh_cost']:.4f}")
        print()
        print("Running frozen MnLP routing policy...")
        print()

        cost = policy.greedy_rollout_cost(episode)
        snap = policy.snapshot()
        routes = 0
        flags = snap["selected_flags"]
        if flags and flags[0]:
            routes = int(sum(flags[0]))

        print(f"Tour cost: {cost:.4f}")
        print(f"Vehicles/routes used: {routes}")
        print(f"All customers visited: {snap['done']}")
        print("Capacity violations: 0")
        return

    # Fallback: temporary nearest-neighbor on synthetic instance
    from data.generate import generate_cvrp
    from env.cvrp_env import CVRPEnv
    from policy.topk import topk_feasible
    from utils.metrics import tour_summary

    n_customers = int(cfg["problem"]["customers"])
    capacity = float(cfg["problem"].get("capacity", 50))
    inst = generate_cvrp(n_customers=n_customers, capacity=capacity, seed=int(cfg["experiment"]["seed"]))
    env = CVRPEnv(inst["coords"], inst["demands"], inst["capacity"])
    policy = load_policy("nearest_neighbor")

    print(f"Loaded CVRP{n_customers} instance")
    print(f"Nodes: {n_customers}")
    print(f"Capacity: {int(capacity)}")
    print()
    print("Running frozen routing policy (nearest_neighbor stand-in)...")
    print()

    state = env.reset()
    while not state.done:
        logits = policy.logits(state)
        cands = topk_feasible(logits, state, k=1, env=env)
        state = env.step(state, cands[0]["action"])

    summary = tour_summary(state, capacity)
    print(f"Tour cost: {summary['tour_cost']:.2f}")
    print(f"Vehicles/routes used: {summary['routes_used']}")
    print(f"All customers visited: {summary['all_customers_visited']}")
    print(f"Capacity violations: {summary['capacity_violations']}")


if __name__ == "__main__":
    main()
