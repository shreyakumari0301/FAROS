"""Phase 0 criterion 6: FAROS MnLP cost must match LaRP MnLP greedy cost.

Compares:
  A) LaRP policy_rollout.greedy_rollout + _get_travel_distance_2
  B) FAROS policy.mnlp_adapter.MnLPPolicy.greedy_rollout_cost

On the same episode from vrp200_test_lkh.txt.
"""

from __future__ import annotations

import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config():
    from utils.paths import to_local_path

    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "mnlp_root", "data_path"):
        if cfg.get("policy", {}).get(key):
            cfg["policy"][key] = str(to_local_path(cfg["policy"][key]))
    return cfg


def lar_cost(cfg, episode: int) -> float:
    """Cost from the original LaRP/MnLP greedy path."""
    import sys as _sys
    from pathlib import Path

    mnlp_root = Path(cfg["policy"]["mnlp_root"])
    _sys.path.insert(0, str(mnlp_root.resolve()))
    _sys.path.insert(0, str((mnlp_root / "CVRP").resolve()))

    import torch
    from VRPEnv import VRPEnv
    from VRPModel import VRPModel
    from policy_rollout import greedy_rollout

    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    env = VRPEnv(
        mode="test",
        data_path=str(cfg["policy"]["data_path"]),
        sub_path=False,
        RRC_budget=0,
    )
    model = VRPModel(
        mode="test",
        embedding_dim=128,
        sqrt_embedding_dim=128 ** 0.5,
        decoder_layer_num=6,
        qkv_dim=16,
        head_num=8,
        ff_hidden_dim=512,
        mtp_depth=4,
        mtp_weight=0.3,
    )
    ckpt = torch.load(str(cfg["policy"]["checkpoint"]), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env.load_raw_data(int(cfg["policy"].get("max_episodes", 128)))
    problems, policy_solution, _, _, _ = greedy_rollout(env, model, episode, device)
    return float(env._get_travel_distance_2(problems, policy_solution).item())


def faros_cost(cfg, episode: int) -> float:
    from policy.backbone import load_policy

    policy = load_policy(
        name="mnlp",
        checkpoint=cfg["policy"].get("checkpoint"),
        mnlp_root=cfg["policy"].get("mnlp_root"),
        data_path=cfg["policy"].get("data_path"),
        device=cfg["policy"].get("device", "cpu"),
        max_episodes=int(cfg["policy"].get("max_episodes", 128)),
    )
    return float(policy.greedy_rollout_cost(episode))


def main() -> None:
    cfg = load_config()
    episode = int(cfg["experiment"].get("episode", 0))

    print(f"Parity check on CVRP200 episode={episode}")
    print("Loading LaRP greedy...")
    old = lar_cost(cfg, episode)
    print(f"  old (LaRP):   {old:.10f}")

    print("Loading FAROS MnLP adapter greedy...")
    new = faros_cost(cfg, episode)
    print(f"  new (FAROS):  {new:.10f}")

    delta = abs(old - new)
    print(f"  |delta|:        {delta:.10e}")

    # Exact match expected up to float noise from identical ops.
    if delta > 1e-6:
        raise SystemExit(
            f"PARITY FAIL: costs differ by {delta}. "
            "Investigate depot/flag semantics, masking, or distance before Phase 1."
        )
    print("PARITY OK")


if __name__ == "__main__":
    main()
