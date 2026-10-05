"""Print a compact summary of burst_world_model*.json logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parents[1] / "outputs" / "logs"


def summarize(name: str) -> None:
    path = LOG / f"{name}.json"
    if not path.exists():
        print(f"{name}: missing")
        return
    data = json.loads(path.read_text())
    rows = data["per_seed"]
    print(f"== {name} ({len(data['seeds'])} seeds, flat_prior={data['flat_prior']})")
    for key, mean in data["mean_cost"].items():
        print(f"  {key:20s} {mean:7.3f}  sd {data['sd_cost'][key]:.2f}")
    for key, test in data["paired"].items():
        lo, hi = test["ci95"]
        print(
            f"  {key:34s} {test['wins']}/{test['losses']}/{test['ties']}"
            f"  p={test['p_two_sided']:.2g}  d={test['mean_delta']:.3f}  ci=[{lo:.2f}, {hi:.2f}]"
        )
    print("  width  ", [round(r["world_model"]["width"], 2) for r in rows])
    print("  start error at forks", round(float(np.mean([r["start_abs_error_mean"] for r in rows])), 3))
    for policy in rows[0]["seconds"]:
        print(f"  seconds/seed {policy:16s} {np.mean([r['seconds'][policy] for r in rows]):.1f}")
    for key in rows[0]["scoring_stats"]:
        stats = [r["scoring_stats"][key] for r in rows]
        calls = np.mean([s.get("msa_calls", s.get("full_msa_invocations", 0)) for s in stats])
        futures = np.mean([s.get("futures_scored", 0) for s in stats])
        print(f"  calls/seed {key:16s} {calls:.1f}  futures {futures:.0f}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ("burst_world_model", "burst_world_model_flat"):
        summarize(name)
