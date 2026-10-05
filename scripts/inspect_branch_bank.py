"""Inspect one BranchGroup from a JSONL branch bank or shard directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.branch_io import load_groups


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        default=os.path.join(ROOT, "data", "processed", "branch_bank_v1"),
    )
    p.add_argument("--index", type=int, default=0, help="Group index across loaded shards")
    args = p.parse_args()

    groups = load_groups(args.path)
    if not groups:
        raise SystemExit(f"No groups in {args.path}")
    if args.index >= len(groups):
        raise SystemExit(f"index {args.index} out of range (n={len(groups)})")

    group = groups[args.index]
    print(f"Instance: {group['instance_id']}")
    print(f"Step: {group['step']}")
    print()

    for c in group["candidates"]:
        tag = "Greedy" if c["candidate_rank"] == 0 else f"Candidate {c['candidate_rank']}"
        print(
            f"{tag}:\n"
            f"  action/node/flag={c['action']}/{c['node']}/{c['flag']}\n"
            f"  completion={c['completion_cost']:.4f}\n"
            f"  advantage={c['advantage_vs_greedy']:+.4f}  regret={c['regret']:.4f}"
        )
        post = c.get("post_action_state", {}).get("structural")
        if post:
            print(
                f"  post_action: rem_customers={post.get('remaining_customers')}  "
                f"cap_slack={post.get('capacity_slack', post.get('remaining_capacity_ratio')):.3f}  "
                f"frag={post.get('fragmentation'):.4f}  "
                f"depot_d={post.get('depot_distance', 0):.4f}  "
                f"Rmin={post.get('routes_lower_bound', 0):.0f}"
            )
        rmap = c.get("rollout_states") or c.get("horizons") or {}
        h4 = rmap.get("4")
        if h4:
            print(
                f"  H=4 rollout: rem_customers={h4.get('remaining_customers')}  "
                f"cap_slack={h4.get('capacity_slack', h4.get('remaining_capacity_ratio')):.3f}  "
                f"frag={h4.get('fragmentation'):.4f}  "
                f"depot_d={h4.get('depot_distance', 0):.4f}  "
                f"Rmin={h4.get('routes_lower_bound', 0):.0f}"
            )
        print()


if __name__ == "__main__":
    main()
