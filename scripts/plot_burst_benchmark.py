"""Bar chart from BENCHMARK_SUMMARY or burst_benchmark_full.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs"
FIG = ROOT / "outputs" / "figures"


def load_aggregate():
    full = LOG / "burst_benchmark_full.json"
    if full.exists():
        with open(full, encoding="utf-8") as handle:
            data = json.load(handle)
        return data["aggregate"], data.get("seeds", []), data.get("test_episodes", [])
    summary = LOG / "BENCHMARK_SUMMARY.json"
    with open(summary, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["burst_benchmark_full_aggregate"], None, [16, 17, 18, 19]


def main():
    agg, seeds, test_eps = load_aggregate()
    order = (
        "nearest",
        "msa_scenario",
        "wait_first",
        "hybrid_wait_msa_go",
        "lazy_msa",
    )
    labels = ["Nearest", "MSA full", "Wait-first", "Hybrid", "Lazy MSA"]
    means = [agg[k]["mean_of_seed_means"] for k in order]
    stds = [agg[k]["std_of_seed_means"] for k in order]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; writing CSV only", flush=True)
        csv = FIG / "burst_tour_costs.csv"
        FIG.mkdir(parents=True, exist_ok=True)
        with open(csv, "w", encoding="utf-8") as handle:
            handle.write("method,mean,std\n")
            for lab, m, s in zip(labels, means, stds):
                handle.write(f"{lab},{m},{s}\n")
        print(f"wrote {csv}", flush=True)
        return

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    colors = ["#888888", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean tour cost (distance = time)")
    n_seed = len(seeds) if seeds else 10
    ax.set_title(f"Burst dynamic CVRP — test eps {test_eps}, {n_seed} seeds")
    ax.set_ylim(0, max(means) * 1.15)
    fig.tight_layout()
    out = FIG / "burst_tour_cost_bar.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
