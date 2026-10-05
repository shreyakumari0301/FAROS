# Experiment package (frozen — for writing phase)

**Status: OR experiments complete.** Do not re-run unless reproducing or extending the benchmark.

## Reproduce (optional)

```powershell
cd C:\Users\SK\FAROS
.\.venv\Scripts\python.exe scripts\run_burst_benchmark.py
.\.venv\Scripts\python.exe scripts\train_burst_wait.py
.\.venv\Scripts\python.exe scripts\summarize_burst_results.py
.\.venv\Scripts\python.exe scripts\plot_burst_benchmark.py
```

- **Instances:** `data/processed/burst_instances_v1/seed_{1234..1243}.pkl`
- **Test:** episodes **16–19** | **Metric:** full-tour cost | **M=20** at forks

## Authoritative OR results (10 seeds)

| Method | Mean tour | Std | Δ vs nearest | Δ vs MSA |
|--------|-----------|-----|--------------|----------|
| Nearest | **25.53** | 1.34 | — | — |
| MSA full | 21.30 | 1.93 | +4.23 | — |
| Wait-first | 20.89 | 1.91 | +4.65 | +0.42 |
| Hybrid | 20.23 | 1.67 | +5.30 | +1.07 |
| Lazy MSA | **20.00** | 1.92 | +5.53 | +1.30 |

**Paired:** 40/40 beat nearest (OR methods). **M ablation:** `ablation_M` in `burst_benchmark_full.json`.

**Compute / seed (mean):** MSA 802 calls, 57k futures scored; hybrid 562 / 36k; lazy 584 full-MSA invocations.

## Secondary

| Track | Result |
|-------|--------|
| Uniform | 45.06 nearest, 45.09 learned — fail (`arrival_value.json`) |
| Learned wait+go (3 seeds, bank) | 24.58 vs 26.06 nearest; go-only **worse** than nearest |
| Wait oracle | ~22.23 (`burst_tours_multiseed.json`) |

## Artifacts

| File | Use |
|------|-----|
| `burst_benchmark_full.json` | Paper table, paired, ablation, compute |
| `BENCHMARK_SUMMARY.json` | One-page merge |
| `outputs/figures/burst_tour_costs.csv` | Figure |

## Writing checklist

- [x] 10-seed OR benchmark
- [x] Bank-aligned learned footnote
- [ ] Draft + figure from CSV
- [ ] Advisor mail (`ADVISOR_EMAIL.txt`)

## Deprecated

`eval_burst_tours.py` — wrong instance stream; do not cite.
