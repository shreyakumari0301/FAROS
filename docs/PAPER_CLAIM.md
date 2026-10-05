# Paper claim (burst dynamic CVRP)

## Problem (one sentence)

**Dynamic CVRP with stochastic reveal times:** all customer locations and demands are fixed, but each customer becomes known only at its reveal time; the vehicle may **wait** or **serve** among currently known feasible customers while following a frozen nearest-feasible completion policy.

## Negative result (uniform)

Independent uniform reveals over the unit square: myopic nearest **45.06**, learned aggregate features **45.09** — no beat (`arrival_value.json`). Supports abandoning static/full-set world models for this noise model.

## Positive result (burst benchmark) — **authoritative: 10 seeds**

Spatial **burst** arrivals (separated centers, overlapping reveal windows). **Frozen instance bank** `burst_instances_v1`. Test episodes **16–19**, seeds **1234–1243**, **M=20** futures at cross-burst forks. Metric: **realized full-tour cost**.

| Method | Mean tour cost | Std (seeds) | Δ vs nearest | Literature slot |
|--------|----------------|-------------|--------------|-----------------|
| Nearest-feasible | **25.53** | 1.34 | — | Myopic dynamic insertion |
| **MSA scenario** | **21.30** | 1.93 | **+4.23** | Bent scenario consensus |
| **Wait-first recourse** | **20.89** | 1.91 | **+4.65** | SS-VRPTW-CR / Mitrović–Laporte waiting |
| **Hybrid wait + MSA-go** (novel) | **20.23** | 1.67 | **+5.30** (+1.07 vs MSA) | Decoupled recourse |
| **Lazy MSA** (novel) | **20.00** | 1.92 | **+5.53** (+1.30 vs MSA) | Amortized scenarios |
| Learned wait+go (MLP, bank) | **24.58** | 5.15 | **+1.48** | Secondary; below OR |
| Wait oracle (cheat) | ~22.23 | — | upper bound | `burst_tours_multiseed.json` |

**Paired (40 test days):** all OR rows beat nearest on every episode. vs full MSA, hybrid/lazy win on most days (`paired_vs_msa` in log). **Compute:** MSA ~802 calls/seed; hybrid ~562; lazy ~584 full-MSA invocations/seed.

Source: `outputs/logs/burst_benchmark_full.json`, `BENCHMARK_SUMMARY.json`.

## Recommended paper story

1. **Do not** lead with a neural world model. Lead with **recourse at cross-burst forks** and **tour cost** as the only success metric.
2. **Primary method (novel):** **Hybrid wait-first + MSA-go** — wait-first for timing; Bent-style MSA only on **go** candidates. Beats full MSA by **~1.1** mean tour cost (10 seeds).
3. **Stronger variant:** **Lazy MSA** — full scenarios only when wait-first agrees with nearest on go target but routing still needs MSA; **~1.3** vs full MSA on mean tour cost.
4. **Baselines:** MSA and wait-first both strongly beat nearest; wait-first slightly beats full MSA on this benchmark.
5. **Learning:** optional; judge on tours only (prediction accuracy ≠ shorter routes).
6. Cite: Saint-Guillain et al. (reveal times), Bent (MSA), Mitrović–Laporte (wait), Ulmer (ADP).

## Commands (reproduce)

```powershell
.\.venv\Scripts\python.exe scripts\run_burst_benchmark.py
.\.venv\Scripts\python.exe scripts\train_burst_wait.py
.\.venv\Scripts\python.exe scripts\summarize_burst_results.py
.\.venv\Scripts\python.exe scripts\plot_burst_benchmark.py
```

See `docs/EXPERIMENTS_COMPLETE.md`.

## Figures

- `outputs/figures/burst_tour_cost_bar.png` — main bar chart
- Map snapshot: one cross-burst decision (manual / future)
- Uniform vs burst schematic (why signal needs spatial correlation)
