"""Unified burst benchmark: frozen instances, all policies, rollout stats, M ablation."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.dynamic_cvrp import rank_nearest
from scripts.burst_dispatch import HORIZON, is_cross_burst, nearest_choice, rollout_tour
from scripts.burst_instances import DEFAULT_EPISODES, ensure_banks, load_or_create_bank
from scripts.burst_scoring import score_actions
from scripts.run_novel_burst import hybrid_wait_msa_go_factory, lazy_msa_factory
from scripts.run_paper_baselines import (
    _cross_actions,
    scenario_cross_factory,
    wait_first_cross_factory,
)
from scripts.train_burst_wait import split_episodes, train_for_seed, learned_wait_cross_factory

SCENARIO_RNG_OFFSET = 7000
DEFAULT_SEEDS = tuple(range(1234, 1244))


def scenario_cross_factory_stats(
    rng: random.Random, n_futures: int, stats: dict, sampler=None, common: bool = False
):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        actions, _ = _cross_actions(inst, state)
        costs = score_actions(
            inst, state, actions, n_futures, rng, stats=stats, sampler=sampler, common=common
        )
        kind, node = actions[int(np.argmin(costs))]
        if kind == "wait":
            return ("wait", 0)
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def hybrid_factory_stats(
    wait_cb, scenario_rng, n_futures: int, stats: dict, sampler=None, common: bool = False
):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        wf = wait_cb(inst, state, feas)
        if wf[0] == "wait":
            return ("wait", 0)
        all_actions, _ = _cross_actions(inst, state)
        go_actions = [a for a in all_actions if a[0] == "go"]
        if len(go_actions) <= 1:
            return ("go", rank_nearest(inst, state, feas)[0])
        costs = score_actions(
            inst, state, go_actions, n_futures, scenario_rng, stats=stats, sampler=sampler, common=common
        )
        kind, node = go_actions[int(np.argmin(costs))]
        node = int(node)
        return ("go", node if node in feas else rank_nearest(inst, state, feas)[0])

    return choose


def lazy_factory_stats(wait_cb, msa_cb, lazy_stats: dict):
    def choose(inst, state, feas):
        if not is_cross_burst(inst, state, HORIZON):
            return ("go", rank_nearest(inst, state, feas)[0])
        nearest = rank_nearest(inst, state, feas)[0]
        wf = wait_cb(inst, state, feas)
        lazy_stats["cross_burst_decisions"] = int(lazy_stats.get("cross_burst_decisions", 0)) + 1
        if wf[0] == "wait":
            return wf
        if wf == ("go", nearest):
            lazy_stats["full_msa_invocations"] = int(lazy_stats.get("full_msa_invocations", 0)) + 1
            return msa_cb(inst, state, feas)
        return wf

    return choose


def eval_policy_on_seed(
    seed: int,
    episodes: int,
    test_eps,
    instances,
    name: str,
    choose,
    cross,
    stats: dict,
    futures: int,
):
    stats.setdefault("msa_calls", 0)
    stats.setdefault("futures_scored", 0)
    costs = []
    cross_n_total = 0
    for ep in test_eps:
        cost, _, _, cross_n = rollout_tour(
            instances[ep], choose, choose_at_cross=cross, horizon=HORIZON
        )
        costs.append(float(cost))
        cross_n_total += cross_n
        print(f"seed {seed} {name} ep {ep}: {cost:.4f}", flush=True)
    row = {
        "mean_cost": float(np.mean(costs)),
        "costs": costs,
        "test_episodes": list(test_eps),
        "cross_burst_stops": cross_n_total,
        "scoring_stats": stats,
    }
    return row


def build_policies(seed: int, futures: int):
    wf = wait_first_cross_factory()
    s_rng = random.Random(seed + SCENARIO_RNG_OFFSET)
    msa_stats: dict = {}
    hybrid_stats: dict = {}
    lazy_stats: dict = {}
    msa = scenario_cross_factory_stats(s_rng, futures, msa_stats)
    lazy = lazy_factory_stats(wf, msa, lazy_stats)

    return {
        "nearest": (nearest_choice, None, {}),
        "msa_scenario": (nearest_choice, msa, msa_stats),
        "wait_first": (nearest_choice, wf, {}),
        "hybrid_wait_msa_go": (
            nearest_choice,
            hybrid_factory_stats(wf, s_rng, futures, hybrid_stats),
            hybrid_stats,
        ),
        "lazy_msa": (nearest_choice, lazy, lazy_stats),
    }


def eval_seed_or_policies(seed: int, episodes: int, futures: int, policy_names=None):
    _tr, _va, test_eps = split_episodes(episodes)
    instances = load_or_create_bank(seed, episodes)
    all_names = policy_names or (
        "nearest",
        "msa_scenario",
        "wait_first",
        "hybrid_wait_msa_go",
        "lazy_msa",
    )
    out = {}
    for name in all_names:
        choose, cross, pre_stats = build_policies(seed, futures)[name]
        stats = pre_stats
        row = eval_policy_on_seed(
            seed, episodes, test_eps, instances, name, choose, cross, stats, futures
        )
        out[name] = row
    base = out["nearest"]["mean_cost"]
    msa_key = "msa_scenario" if "msa_scenario" in out else "msa_full"
    msa_base = out[msa_key]["mean_cost"] if msa_key in out else None
    for name in out:
        if name != "nearest":
            out[name]["gap_vs_nearest"] = base - out[name]["mean_cost"]
        if msa_base is not None and name not in ("nearest", msa_key):
            out[name]["gap_vs_msa_full"] = msa_base - out[name]["mean_cost"]
    return {"seed": seed, "policies": out}


def paired_analysis(per_seed_rows, test_eps, method: str, baseline: str = "nearest"):
    pairs = []
    for row in per_seed_rows:
        seed = row["seed"]
        b_costs = row["policies"][baseline]["costs"]
        m_costs = row["policies"][method]["costs"]
        for ep, bc, mc in zip(test_eps, b_costs, m_costs):
            pairs.append({"seed": seed, "episode": ep, "delta": float(bc - mc)})
    deltas = np.array([p["delta"] for p in pairs], dtype=np.float64)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        boots.append(float(sample.mean()))
    boots.sort()
    return {
        "n_pairs": len(pairs),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "median_delta": float(np.median(deltas)),
        "frac_improved": float((deltas > 0).mean()),
        "bootstrap_mean_ci_95": [float(boots[50]), float(boots[int(0.025 * len(boots))]), float(boots[int(0.975 * len(boots))])],
        "pairs": pairs,
    }


def aggregate_policies(per_seed):
    names = list(per_seed[0]["policies"].keys())
    agg = {}
    for name in names:
        means = [r["policies"][name]["mean_cost"] for r in per_seed]
        entry = {
            "mean_of_seed_means": float(np.mean(means)),
            "std_of_seed_means": float(np.std(means)),
            "mean_msa_calls": float(
                np.mean([r["policies"][name]["scoring_stats"].get("msa_calls", 0) for r in per_seed])
            ),
            "mean_futures_scored": float(
                np.mean(
                    [r["policies"][name]["scoring_stats"].get("futures_scored", 0) for r in per_seed]
                )
            ),
        }
        if name != "nearest":
            entry["mean_gap_vs_nearest"] = float(
                np.mean([r["policies"][name]["gap_vs_nearest"] for r in per_seed])
            )
        if "gap_vs_msa_full" in per_seed[0]["policies"][name]:
            entry["mean_gap_vs_msa_full"] = float(
                np.mean([r["policies"][name].get("gap_vs_msa_full", 0) for r in per_seed])
            )
        if name == "lazy_msa":
            entry["mean_full_msa_invocations"] = float(
                np.mean(
                    [
                        r["policies"][name]["scoring_stats"].get("full_msa_invocations", 0)
                        for r in per_seed
                    ]
                )
            )
            entry["mean_cross_burst_decisions"] = float(
                np.mean(
                    [
                        r["policies"][name]["scoring_stats"].get("cross_burst_decisions", 0)
                        for r in per_seed
                    ]
                )
            )
        agg[name] = entry
    return agg


def eval_learned_seed(seed: int, episodes: int, futures: int):
    _inst, _grp, test_eps, wait_model, wait_apply, _go_m, _go_a = train_for_seed(
        seed, futures, episodes, use_instance_bank=True
    )
    instances = load_or_create_bank(seed, episodes)
    learn = learned_wait_cross_factory(wait_model, wait_apply)
    return eval_policy_on_seed(
        seed,
        episodes,
        test_eps,
        instances,
        "learned_wait_cross",
        nearest_choice,
        learn,
        {},
        futures,
    )


def main():
    probe = "--probe" in sys.argv
    with_learned = "--with-learned" in sys.argv
    seeds = (1234,) if probe else DEFAULT_SEEDS
    episodes = 8 if probe else DEFAULT_EPISODES
    futures = 8 if probe else 20
    _tr, _va, test_eps = split_episodes(episodes)

    ensure_banks(seeds, episodes)

    per_seed = [eval_seed_or_policies(s, episodes, futures) for s in seeds]

    ablation = {}
    for m in (5, 10, 20) if not probe else (8,):
        ablation[str(m)] = {}
        for s in seeds:
            partial = eval_seed_or_policies(
                s, episodes, m, policy_names=("nearest", "msa_scenario", "hybrid_wait_msa_go")
            )
            ablation[str(m)][str(s)] = {
                name: {
                    "mean_cost": partial["policies"][name]["mean_cost"],
                    "msa_calls": partial["policies"][name]["scoring_stats"].get("msa_calls", 0),
                    "futures_scored": partial["policies"][name]["scoring_stats"].get(
                        "futures_scored", 0
                    ),
                }
                for name in partial["policies"]
            }
        ablation[str(m)]["aggregate"] = {
            "msa_scenario_mean": float(
                np.mean([ablation[str(m)][str(s)]["msa_scenario"]["mean_cost"] for s in seeds])
            ),
            "hybrid_mean": float(
                np.mean([ablation[str(m)][str(s)]["hybrid_wait_msa_go"]["mean_cost"] for s in seeds])
            ),
            "nearest_mean": float(
                np.mean([ablation[str(m)][str(s)]["nearest"]["mean_cost"] for s in seeds])
            ),
        }

    paired = {}
    for method in ("msa_scenario", "wait_first", "hybrid_wait_msa_go", "lazy_msa"):
        if method in per_seed[0]["policies"]:
            paired[method] = paired_analysis(per_seed, test_eps, method, "nearest")
            if method != "msa_scenario":
                paired[f"{method}_vs_msa"] = paired_analysis(
                    per_seed, test_eps, method, "msa_scenario"
                )

    learned_rows = []
    if with_learned:
        learn_seeds = seeds if probe else (1234, 1235, 1236)
        for s in learn_seeds:
            print(f"training learned wait seed {s}...", flush=True)
            row = eval_learned_seed(s, episodes, futures)
            learned_rows.append({"seed": s, "learned_wait_cross": row})

    report = {
        "instance_bank": str(Path(ROOT) / "data" / "processed" / "burst_instances_v1"),
        "seeds": list(seeds),
        "test_episodes": test_eps,
        "futures_per_decision": futures,
        "scenario_rng_offset": SCENARIO_RNG_OFFSET,
        "per_seed": per_seed,
        "aggregate": aggregate_policies(per_seed),
        "paired_vs_nearest": {k: v for k, v in paired.items() if not k.endswith("_vs_msa")},
        "paired_vs_msa": {k: v for k, v in paired.items() if k.endswith("_vs_msa")},
        "ablation_M": ablation,
        "learned_wait_cross": learned_rows if with_learned else None,
    }
    for key in list(report["paired_vs_nearest"].keys()):
        report["paired_vs_nearest"][key].pop("pairs", None)
    for key in list(report["paired_vs_msa"].keys()):
        report["paired_vs_msa"][key].pop("pairs", None)

    log_dir = Path(ROOT) / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / ("burst_benchmark_probe.json" if probe else "burst_benchmark_full.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if probe:
        print(json.dumps({"aggregate": report["aggregate"], "out": str(out)}, indent=2), flush=True)
        return

    # Mirror legacy logs for summarize / docs
    paper = {
        "seeds": list(seeds),
        "futures_per_decision": futures,
        "instance_bank": report["instance_bank"],
        "per_seed": [
            {
                "seed": r["seed"],
                "test_episodes": test_eps,
                "policies": {
                    k: {
                        "mean_cost": v["mean_cost"],
                        "costs": v["costs"],
                        "gap_vs_nearest": v.get("gap_vs_nearest"),
                    }
                    for k, v in r["policies"].items()
                },
            }
            for r in per_seed
        ],
        "aggregate": {
            k: {
                "mean_of_means": v["mean_of_seed_means"],
                "std_of_means": v["std_of_seed_means"],
                **({ "mean_gap_vs_nearest": v["mean_gap_vs_nearest"] } if "mean_gap_vs_nearest" in v else {}),
            }
            for k, v in report["aggregate"].items()
        },
    }
    with open(log_dir / "paper_baselines_burst.json", "w", encoding="utf-8") as handle:
        json.dump(paper, handle, indent=2)

    novel = {
        "title": report.get("title", "Hybrid wait-first + selective MSA"),
        "instance_bank": report["instance_bank"],
        "per_seed": [
            {
                "seed": r["seed"],
                "policies": {
                    k: {kk: vv for kk, vv in v.items() if kk != "scoring_stats" and kk != "test_episodes"}
                    for k, v in r["policies"].items()
                    if k != "msa_scenario" or True
                },
            }
            for r in per_seed
        ],
        "aggregate": {
            k.replace("msa_scenario", "msa_full"): {
                "mean_cost": v["mean_of_seed_means"],
                **(
                    {"mean_gap_vs_nearest": v["mean_gap_vs_nearest"]}
                    if "mean_gap_vs_nearest" in v
                    else {}
                ),
                **(
                    {"mean_gap_vs_msa_full": v["mean_gap_vs_msa_full"]}
                    if "mean_gap_vs_msa_full" in v
                    else {}
                ),
            }
            for k, v in report["aggregate"].items()
        },
    }
    # rename msa_scenario -> msa_full in per_seed for novel_burst compat
    for row in novel["per_seed"]:
        if "msa_scenario" in row["policies"]:
            row["policies"]["msa_full"] = row["policies"].pop("msa_scenario")
    with open(log_dir / "novel_burst.json", "w", encoding="utf-8") as handle:
        json.dump(novel, handle, indent=2)

    print(json.dumps({"aggregate": report["aggregate"], "out": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
