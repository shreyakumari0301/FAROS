"""Generate counterfactual BranchGroups along frozen MnLP greedy trajectories.

Phase 1 only — no FAROS model training.

Supports:
  --resume          continue from last completed instance
  per-instance JSONL shards + generation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.branch_record import (
    BranchGroup,
    BranchRecord,
    StructuralSnapshot,
    assign_relative_labels,
)
from planner.rollout import (
    greedy_complete,
    labels_from_env,
    prefix_cost_now,
    restore_env,
    rollout_to_horizons,
    snapshot_env,
)
from policy.backbone import load_policy
from policy.mnlp_adapter import topk_feasible_mnlp
from utils.paths import to_local_path
from utils.seed import set_seed


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "mnlp_root", "data_path"):
        if cfg.get("policy", {}).get(key):
            cfg["policy"][key] = str(to_local_path(cfg["policy"][key]))
    return cfg


def _jsonable_state(snap: dict) -> dict:
    coords = snap["coords"]
    return {
        "episode": snap["episode"],
        "step": snap["step"],
        "done": snap["done"],
        "current_node": snap["current_node"],
        "remaining_capacity": snap["remaining_capacity"],
        "capacity": snap["capacity"],
        "n_customers": snap["n_customers"],
        "visited": snap["visited"].astype(int).tolist(),
        "selected_nodes": snap["selected_nodes"],
        "selected_flags": snap["selected_flags"],
        "coords": coords.tolist(),
        "demands": snap["demands"].tolist(),
    }


def generate_episode(policy, episode: int, top_k: int, horizons, step_stride: int = 1, full_states: bool = False) -> List[BranchGroup]:
    from dataclasses import asdict

    groups: List[BranchGroup] = []
    policy.reset_episode(episode)

    while not policy._done:
        step = policy._current_step
        collect = step % max(1, int(step_stride)) == 0

        base_snap = snapshot_env(policy.env)
        policy._step_state = policy.env.step_state
        state_view = policy.snapshot()

        logits = policy.logits()
        candidates = topk_feasible_mnlp(logits, policy, k=top_k)
        if not candidates:
            break

        greedy = candidates[0]

        if collect:
            records = []
            for rank, cand in enumerate(candidates):
                restore_env(policy.env, base_snap)
                policy._step_state = policy.env.step_state
                policy._current_step = step
                policy._done = False

                before = prefix_cost_now(policy)
                policy.step_joint(int(cand["action"]))
                after = prefix_cost_now(policy)
                immediate = after - before

                # Exact immediate future s_{t+1}
                post_snap_view = policy.snapshot()
                post_labels = labels_from_env(policy, after)
                post_action_state = {
                    "current_node": post_snap_view["current_node"],
                    "remaining_capacity": post_snap_view["remaining_capacity"],
                    "done": policy._done,
                    "visited": post_snap_view["visited"].astype(int).tolist(),
                    "selected_nodes": post_snap_view["selected_nodes"],
                    "selected_flags": post_snap_view["selected_flags"],
                    "structural": asdict(StructuralSnapshot.from_labels(post_labels)),
                }

                post_snap = snapshot_env(policy.env)
                post_step = policy._current_step
                post_done = policy._done

                futures = {} if full_states else None
                horizon_labs = rollout_to_horizons(policy, horizons, full_states=futures)

                restore_env(policy.env, post_snap)
                policy._step_state = policy.env.step_state
                policy._current_step = post_step
                policy._done = post_done
                completion = greedy_complete(policy)

                rollout_states = {
                    str(h): StructuralSnapshot.from_labels(v)
                    for h, v in horizon_labs.items()
                }

                records.append(
                    BranchRecord(
                        instance_id=episode,
                        step=step,
                        action=int(cand["action"]),
                        node=int(cand["node"]),
                        flag=int(cand["flag"]),
                        policy_score=float(cand["score"]),
                        candidate_rank=rank,
                        immediate_cost=float(immediate),
                        completion_cost=float(completion),
                        post_action_state=post_action_state,
                        rollout_states=rollout_states,
                        future_states=futures or {},
                    )
                )

            group = BranchGroup(
                instance_id=episode,
                step=step,
                state=_jsonable_state(state_view),
                greedy_action=int(greedy["action"]),
                greedy_node=int(greedy["node"]),
                greedy_flag=int(greedy["flag"]),
                candidates=records,
            )
            assign_relative_labels(group)
            groups.append(group)

        restore_env(policy.env, base_snap)
        policy._step_state = policy.env.step_state
        policy._current_step = step
        policy._done = False
        policy.step_joint(int(greedy["action"]))

    return groups


def _bank_dir(cfg: dict) -> Path:
    out_dir = Path(to_local_path(cfg["paths"].get("processed_data", "data/processed")))
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    bank = out_dir / "branch_bank_v1"
    bank.mkdir(parents=True, exist_ok=True)
    return bank


def _write_manifest(bank: Path, cfg: dict, branch_cfg: dict) -> None:
    manifest = {
        "policy_checkpoint": cfg["policy"].get("checkpoint"),
        "mnlp_root": cfg["policy"].get("mnlp_root"),
        "data_path": cfg["policy"].get("data_path"),
        "n_customers": cfg["problem"].get("customers", 200),
        "top_k": int(branch_cfg.get("top_k", 5)),
        "horizons": list(branch_cfg.get("horizons", [1, 2, 4, 8])),
        "step_stride": int(branch_cfg.get("step_stride", 1)),
        "seed": int(cfg["experiment"].get("seed", 1234)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": {
            "post_action_state": "exact T(s_t, a)",
            "rollout_states[H]": "s_{t+H} after action + (H-1) greedy steps; H=1 == post_action",
        },
    }
    path = bank / "generation_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_progress(bank: Path) -> dict:
    path = bank / "progress.json"
    if not path.exists():
        return {"last_completed_instance": -1, "completed": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(bank: Path, progress: dict) -> None:
    with open(bank / "progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume from last completed instance")
    parser.add_argument("--n-instances", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(int(cfg["experiment"]["seed"]))
    branch_cfg = dict(cfg.get("branch_generation", {}))

    if args.n_instances is not None:
        branch_cfg["n_instances"] = args.n_instances
    if args.start_episode is not None:
        branch_cfg["start_episode"] = args.start_episode
    if args.step_stride is not None:
        branch_cfg["step_stride"] = args.step_stride
    if args.top_k is not None:
        branch_cfg["top_k"] = args.top_k

    top_k = int(branch_cfg.get("top_k", 5))
    horizons = list(branch_cfg.get("horizons", [1, 2, 4, 8]))
    n_instances = int(branch_cfg.get("n_instances", 5))
    start_episode = int(branch_cfg.get("start_episode", 0))
    step_stride = int(branch_cfg.get("step_stride", 1))

    bank = _bank_dir(cfg)
    _write_manifest(bank, cfg, branch_cfg)
    progress = _load_progress(bank) if args.resume else {"last_completed_instance": -1, "completed": []}

    policy = load_policy(
        name="mnlp",
        checkpoint=cfg["policy"].get("checkpoint"),
        mnlp_root=cfg["policy"].get("mnlp_root"),
        data_path=cfg["policy"].get("data_path"),
        device=cfg["policy"].get("device", "cpu"),
        max_episodes=int(cfg["policy"].get("max_episodes", 128)),
    )

    end_episode = start_episode + n_instances - 1
    print(
        f"Branch bank -> {bank}\n"
        f"episodes {start_episode}..{end_episode}, K={top_k}, H={horizons}, stride={step_stride}, "
        f"resume={args.resume}"
    )

    total_groups = 0
    for offset in range(n_instances):
        ep = start_episode + offset
        shard = bank / f"instance_{ep:04d}.jsonl"

        if args.resume and ep <= int(progress.get("last_completed_instance", -1)):
            print(f"  episode {ep}: skip (already completed)", flush=True)
            # Count existing lines for summary
            if shard.exists():
                with open(shard, "r", encoding="utf-8") as f:
                    total_groups += sum(1 for _ in f)
            continue

        t0 = time.time()
        print(f"  episode {ep} ...", flush=True)
        groups = generate_episode(
            policy, ep, top_k=top_k, horizons=horizons, step_stride=step_stride
        )
        with open(shard, "w", encoding="utf-8") as f:
            for g in groups:
                f.write(json.dumps(g.to_dict()) + "\n")

        total_groups += len(groups)
        elapsed = time.time() - t0
        print(f"    wrote {len(groups)} groups -> {shard.name} ({elapsed:.1f}s)", flush=True)

        completed = list(progress.get("completed", []))
        if ep not in completed:
            completed.append(ep)
        progress = {
            "last_completed_instance": ep,
            "completed": sorted(completed),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total_groups_so_far": total_groups,
        }
        _save_progress(bank, progress)

    # Convenience concat pointer file
    index = {
        "bank_dir": str(bank),
        "instances": sorted(progress.get("completed", [])),
        "total_groups": total_groups,
    }
    with open(bank / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Done. ~{total_groups} BranchGroups in {bank}")


if __name__ == "__main__":
    main()
