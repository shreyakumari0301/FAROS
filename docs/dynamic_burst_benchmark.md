# Dynamic burst CVRP benchmark (FAROS gate)

## Problem

- **CVRP200**, capacity **80**, reveal horizon **40**, Euclidean travel time.
- **Frozen dispatcher:** nearest-feasible among **already revealed** customers; depot return and idle **wait** when nothing is feasible (unchanged).
- **Learned piece:** at **cross-burst** stops only, choose **wait** or **go** to one of up to **4 spatially separated** feasible customers (min pairwise distance **0.25**).
- **Generator:** `generate_burst_instance` — 4 well-separated wave centers, tight customer noise (**σ=0.015**), overlapping reveal windows. Hidden customers stay hidden from features.

## Splits

| Split | Episodes | Use |
|-------|----------|-----|
| Train | 0–11 | fit scorer |
| Val | 12–15 | early stop |
| Test | 16–19 | report tour cost |

Default seeds: **1234, 1235, 1236** (20 instances per seed).

## Metrics

1. **One-step expected completion cost** — after one deviating action, mean cost over **M=20** resampled future reveal times in cluster windows; then nearest-feasible rollout (`train_burst_wait.py` / `burst_value.json`).
2. **Realized tour cost** — full day on **true** reveal times (`eval_burst_tours.py`, `train_burst_wait.py`).

**Primary pass criterion:** test mean tour cost **below** nearest-feasible by a **clear margin** (not noise on one seed).

## Results (authoritative logs)

**Primary (10 seeds, frozen bank, test 16–19):** see `burst_benchmark_full.json` — nearest **25.53**, lazy MSA **20.00**, hybrid **20.23**, full MSA **21.30**, wait-first **20.89**. Protocol: `docs/EXPERIMENTS_COMPLETE.md`.

| Setting | Nearest | Learned wait+go @ cross-burst | Notes |
|---------|---------|----------------------------------|-------|
| Uniform reveals | 45.06 | **fail** (MLP / DeepSet) | `arrival_value.json`, `set_encoder.json` |
| Burst (legacy 3-seed pre-bank) | 25.83 ± 1.11 | 22.28 ± 0.70 | superseded by 10-seed bank run |
| Burst, wait **oracle** (cheat) | — | ~22.23 | `burst_tours_multiseed.json` |

The **wait oracle** is an upper bound (perfect lookahead at cross-burst); it is not deployable.

## Commands (Windows `.venv`)

```powershell
cd C:\Users\SK\FAROS
.\.venv\Scripts\python.exe scripts\train_burst_wait.py          # train + test tours, 3 seeds
.\.venv\Scripts\python.exe scripts\train_burst_wait.py --probe # smoke
.\.venv\Scripts\python.exe scripts\eval_burst_tours.py
.\.venv\Scripts\python.exe scripts\eval_burst_wait.py
.\.venv\Scripts\python.exe scripts\eval_burst_multiseed.py
.\.venv\Scripts\python.exe scripts\run_burst_benchmark.py
.\.venv\Scripts\python.exe scripts\train_burst_wait.py
.\.venv\Scripts\python.exe scripts\summarize_burst_results.py
.\.venv\Scripts\python.exe scripts\plot_burst_benchmark.py
```

## Paper baselines (literature-aligned)

Run: `.\.venv\Scripts\python.exe scripts\run_paper_baselines.py` → `outputs/logs/paper_baselines_burst.json`

| Policy | Literature analogue |
|--------|---------------------|
| `nearest` | Myopic dynamic insertion |
| `msa_scenario` | Bent scenario consensus — argmin **E[completion cost]** over **M** resampled cluster reveals (same model as training labels) |
| `wait_first` | SS-VRPTW-CR / Mitrović–Laporte — **wait** when a hidden customer shares the current burst window and reveals soon |
| `learned_wait_cross` | Learning-augmented fork (MLP on visible features) |

Positioning paragraph: `docs/LITERATURE_DYNAMIC_VRP.md`.

Export labels: `data/processed/burst_decisions_v1/`. Uniform reveals failed the same gate; do not use static CVRP world models here.
