"""Phase 1B/1C diagnostics for a FAROS branch bank.

Reports:
  1. Oracle-best rank frequencies
  2. Delta_best improvement stats (mean/median/quantiles/thresholds)
  3. Mean regret by policy rank
  4. Per-horizon correlations of structural features with regret
  5. Branch divergence: ||phi_i - phi_j|| vs |C_i - C_j|
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.branch_io import load_groups
from data.structural_labels import PHI_KEYS


FEATURE_KEYS = [
    "remaining_capacity_ratio",
    "capacity_pressure",
    "fragmentation",
    "remaining_demand_ratio",
    "depot_distance",
    "routes_lower_bound",
]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rollout_map(cand: dict) -> dict:
    return cand.get("rollout_states") or cand.get("horizons") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=os.path.join(ROOT, "data", "processed", "branch_bank_v1"),
        help="JSONL file or branch_bank_v1 directory",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    groups = load_groups(args.path)
    n = len(groups)
    if n == 0:
        raise SystemExit(f"empty branch bank: {args.path}")

    best_rank: List[int] = []
    delta_best: List[float] = []
    regret_by_rank: Dict[int, List[float]] = defaultdict(list)

    # Per-horizon feature vs regret
    feat_store: Dict[int, Dict[str, List[float]]] = {
        h: {k: [] for k in FEATURE_KEYS + ["regret"]} for h in args.horizons
    }

    # Divergence pairs per horizon
    div_dist: Dict[int, List[float]] = {h: [] for h in args.horizons}
    div_cost: Dict[int, List[float]] = {h: [] for h in args.horizons}

    for g in groups:
        cands = g["candidates"]
        costs = np.asarray([c["completion_cost"] for c in cands], dtype=np.float64)
        best = int(np.argmin(costs))
        best_rank.append(best)
        greedy_c = float(costs[0])
        best_c = float(costs[best])
        delta_best.append(greedy_c - best_c)

        for c in cands:
            regret_by_rank[int(c["candidate_rank"])].append(float(c["regret"]))
            rmap = _rollout_map(c)
            for h in args.horizons:
                snap = rmap.get(str(h)) or rmap.get(h)
                if snap is None:
                    continue
                feat_store[h]["regret"].append(float(c["regret"]))
                for k in FEATURE_KEYS:
                    feat_store[h][k].append(float(snap.get(k, np.nan)))

        # Pairwise divergence within the BranchGroup
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ci, cj = cands[i], cands[j]
                dcost = abs(float(ci["completion_cost"]) - float(cj["completion_cost"]))
                ri, rj = _rollout_map(ci), _rollout_map(cj)
                for h in args.horizons:
                    si = ri.get(str(h)) or ri.get(h)
                    sj = rj.get(str(h)) or rj.get(h)
                    if si is None or sj is None:
                        continue
                    phi_i = si.get("phi")
                    phi_j = sj.get("phi")
                    if not phi_i or not phi_j:
                        phi_i = [float(si.get(k, 0.0)) for k in PHI_KEYS]
                        phi_j = [float(sj.get(k, 0.0)) for k in PHI_KEYS]
                    dist = float(np.linalg.norm(np.asarray(phi_i) - np.asarray(phi_j)))
                    div_dist[h].append(dist)
                    div_cost[h].append(dcost)

    # ---- Report ----
    print("=" * 60)
    print(f"Branch bank: {args.path}")
    print(f"1) BranchGroups: {n}")
    print("=" * 60)

    print("\n## Oracle-best candidate by policy rank")
    counts = Counter(best_rank)
    print(f"{'Oracle-best candidate':<24} {'Frequency':>10} {'Count':>8}")
    for r in range(max(counts) + 1 if counts else 0):
        cnt = counts.get(r, 0)
        print(f"rank {r:<19} {100.0 * cnt / n:9.1f}% {cnt:8d}")
    p_rank1_best = 100.0 * counts.get(0, 0) / n
    print(f"\nP(rank-1 = oracle best) = {p_rank1_best:.1f}%")
    print(f"P(rank-1 NOT best)     = {100.0 - p_rank1_best:.1f}%")

    d = np.asarray(delta_best, dtype=np.float64)
    print("\n## Delta_best = C_greedy - min_i C_i  (improvement if oracle picks best)")
    print(f"mean:     {d.mean():.6f}")
    print(f"median:   {np.median(d):.6f}")
    print(f"p90:      {np.percentile(d, 90):.6f}")
    print(f"max:      {d.max():.6f}")
    for thr in (0.0, 0.01, 0.05, 0.1):
        frac = float(np.mean(d > thr + 1e-12 if thr > 0 else d > 1e-12))
        print(f"fraction > {thr}: {100.0 * frac:.1f}%")

    print("\n## Mean regret by policy rank  (R_i = C_i - C_best)")
    print(f"{'Policy rank':<12} {'Mean regret':>12} {'N':>8}")
    for r in sorted(regret_by_rank):
        vals = np.asarray(regret_by_rank[r], dtype=np.float64)
        print(f"{r:<12} {vals.mean():12.6f} {vals.size:8d}")

    print("\n## Structural feature ↔ regret correlations by horizon")
    print(
        f"{'Feature':<28} "
        + " ".join(f"{'H=' + str(h):>8}" for h in args.horizons)
    )
    for k in FEATURE_KEYS:
        cells = []
        for h in args.horizons:
            x = np.asarray(feat_store[h][k], dtype=np.float64)
            y = np.asarray(feat_store[h]["regret"], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = _safe_corr(x[mask], y[mask])
            cells.append(f"{corr:+8.4f}" if np.isfinite(corr) else f"{'nan':>8}")
        print(f"{k:<28} " + " ".join(cells))

    print("\n## Branch divergence: corr(||φ_i-φ_j||, |C_i-C_j|) by horizon")
    print(f"{'Horizon':<10} {'corr':>8} {'mean D_H':>10} {'mean |ΔC|':>10} {'pairs':>8}")
    for h in args.horizons:
        dd = np.asarray(div_dist[h], dtype=np.float64)
        dc = np.asarray(div_cost[h], dtype=np.float64)
        corr = _safe_corr(dd, dc)
        corr_s = f"{corr:+.4f}" if np.isfinite(corr) else "nan"
        print(
            f"H={h:<7} {corr_s:>8} {dd.mean():10.4f} {dc.mean():10.4f} {dd.size:8d}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
