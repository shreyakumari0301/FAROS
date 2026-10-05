"""Merge burst experiment JSON logs into one summary for reports."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs"


def load(name):
    path = LOG / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    arrival = load("arrival_value.json")
    learned = load("burst_learned_wait.json")
    paper = load("paper_baselines_burst.json")
    novel = load("novel_burst.json")
    full = load("burst_benchmark_full.json")
    multiseed = load("burst_tours_multiseed.json")
    wait = load("burst_wait.json")

    summary = {
        "uniform_dynamic": {
            "test_myopic_mean": arrival.get("myopic_mean_cost") if arrival else None,
            "test_model_mean": arrival.get("model_mean_cost") if arrival else None,
            "pass_beat_nearest": (
                arrival["model_mean_cost"] < arrival["myopic_mean_cost"] if arrival else False
            ),
        },
        "burst_tours_learned_wait": learned.get("aggregate") if learned else None,
        "paper_baselines_literature": paper.get("aggregate") if paper else None,
        "novel_hybrid_lazy": novel.get("aggregate") if novel else None,
        "burst_benchmark_full_aggregate": full.get("aggregate") if full else None,
        "burst_benchmark_paired_vs_nearest": (
            {k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in full.get("paired_vs_nearest", {}).items()}
            if full
            else None
        ),
        "burst_benchmark_ablation_M": full.get("ablation_M") if full else None,
        "instance_bank": full.get("instance_bank") if full else None,
        "burst_tours_multiseed_go_only": multiseed.get("aggregate") if multiseed else None,
        "burst_wait_seed_1234": {
            "cross_burst_first_action": wait.get("cross_burst_first_action") if wait else None,
            "full_tours": wait.get("full_tours") if wait else None,
        },
        "log_files": [
            "arrival_value.json",
            "set_encoder.json",
            "burst_value.json",
            "burst_tours.json",
            "burst_wait.json",
            "burst_tours_multiseed.json",
            "burst_learned_wait.json",
            "novel_burst.json",
            "burst_benchmark_full.json",
        ],
        "benchmark_doc": "docs/dynamic_burst_benchmark.md",
        "advisor_draft": "docs/ADVISOR_EMAIL.txt",
    }
    out = LOG / "BENCHMARK_SUMMARY.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
