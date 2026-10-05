# Dynamic burst CVRP — paper structure (mirrors CTA-RAG / main2.tex spine)

**Working title:** Decoupled recourse at cross-burst forks in dynamic CVRP with stochastic reveal times

**Target venue type:** OR / transport / optimization (EJOR, COR, EURO JTL, T-ITS) — not vision/ML main track.

**Narrative spine:** Problem → why frozen backbone → where uncertainty matters (cross-burst) → technical gap (monolithic MSA mixes timing and routing) → hybrid/lazy mechanism → why it should work → evidence (tour cost only).

**Every number below is from:** `outputs/logs/burst_benchmark_full.json`, `BENCHMARK_SUMMARY.json`, `arrival_value.json`, `burst_learned_wait.json` (bank-aligned), unless marked PLACEHOLDER.

---

## Abstract (draft)

Dynamic vehicle routing with fixed locations and demands but **stochastic reveal times** requires policies that minimize **realized travel distance**, not one-step prediction accuracy. We keep a **frozen nearest-feasible** backbone and intervene only at **cross-burst** stops, where spatially separated feasible customers and overlapping reveal windows create a wait-vs-route fork. Literature-aligned **scenario recourse** (Bent-style MSA over resampled cluster futures) and **wait-first** heuristics strongly beat myopic insertion on a documented burst benchmark (10 seeds, test episodes 16–19). We propose **decoupled recourse**: **hybrid wait-first + MSA-go** fixes timing without scenarios and runs MSA only over go-actions; **lazy MSA** invokes full scenarios only when wait-first agrees with nearest on go. On mean **full-tour cost**, lazy MSA reaches **20.00** versus nearest **25.53** (+5.53) and full MSA **21.30** (+1.30), with hybrid at **20.23** (+1.07 vs MSA). All OR policies beat nearest on **40/40** paired test days. **Uniform** independent reveals yield a **negative** learning result (tour cost ≈ nearest). A bank-aligned MLP improves tours only modestly (**24.58** vs **26.06** nearest) and remains below OR. We report scenario **call counts** (MSA ~802 vs hybrid ~562 per seed) and **M** ablation. Primary metric throughout is **realized tour cost**; one-step accuracy does not track tour improvement.

**Keywords:** dynamic CVRP, reveal times, recourse, scenario consensus, waiting strategies, benchmark.

---

## 1. Introduction

**SOC / planner analogy (optional one paragraph):** A single vehicle must serve customers that **appear over time**; replanning happens when stopped; waiting is allowed. The **operational objective** is total distance traveled (travel time).

**Failure modes (two types, tour-level):**

1. **Myopic insertion** — serve nearest feasible customer; misses anticipatory wait and inter-burst routing (+~5.5 tour cost on burst).
2. **Monolithic scenario pool** — score wait and all go-actions in one MSA argmin; mixes **timing** and **routing** recourse; extra distractors in context (CWE-sibling analogue: catalog neighbors in RCM paper).

**Gap (structural, not “better retriever”):** Prior dynamic VRP work adapts **reoptimization**, **accept/reject**, or **full policy learning**. We fix a **simple backbone** and ask: at **spatially correlated** forks only, can **decoupling** wait timing from scenario routing beat **full MSA** on the **same** completion-cost functional?

**Contributions (numbered):**

1. **Burst dynamic CVRP benchmark** — frozen instance bank, train/val/test splits, uniform negative control.
2. **Decoupled recourse policies** — hybrid wait + MSA-go; lazy MSA activation.
3. **Unified evaluation** — one driver, paired tour statistics, M ablation, rollout accounting.
4. **Honest learning line** — MLP on rollout labels; tour-eval only; fails on uniform; below OR on burst.

---

## 2. Related work and research gap

| Line | What they adapt | Our contrast |
|------|-----------------|--------------|
| Psaraftis / online reopt | Full route each event | Frozen nearest backbone |
| Bent MSA / SS-VRPTW-CR | Scenario recourse + waiting | Same uncertainty model; we **split** wait vs go scoring |
| Ulmer ADP | Value approximation | No learned value; OR rollouts at forks |
| Neural static CVRP (Kool et al.) | End-to-end routing | Wrong problem (static, full visibility) |
| Uniform reveal + MLP | Features of visible set | **Negative result** — no tour gain |

**Gap:** No prior work in our setting combines (i) **reveal-time-only** dynamics, (ii) **cross-burst fork** intervention only, (iii) **decoupled** wait-first timing + selective MSA on **go**, (iv) **tour cost** as sole success metric with **paired** episode tests.

---

## 3. Problem formulation

**Instance:** CVRP200, capacity 80, horizon 40; Euclidean travel time = distance.

**Dynamics:** Customer \(i\) has fixed \((x_i, d_i)\); reveal time \(r_i\) known only when \(t \ge r_i\).

**Backbone:** When feasible customers exist at a stop, default **go** to nearest feasible; depot returns; idle **wait** for next reveal.

**Intervention set:** **Cross-burst** stops: ≥2 separated feasible customers (min pairwise distance 0.25), time \(<\) horizon, future reveals possible.

**Action at fork:** `wait` or `go` to one of up to **4** separated candidates.

**Objective (Problem statement):**

\[
\min \;\; \mathbb{E}\big[\text{total tour distance on true reveals}\big]
\]

**Not the objective:** one-step argmin match, top-1 on local labels, expected cost of a single deviating action without full rollout.

**Failure taxonomy (tour-level):**

| Outcome | Meaning |
|---------|---------|
| **Improve** | Full tour cost \(<\) nearest |
| **Worsen** | Full tour cost \(>\) nearest |
| **Tie** | Equal cost |

Report **frac_improved**, **frac_worsened** on paired (seed, episode) days — analogue to rejected vs silent, but here all policies output valid tours.

**Design hypothesis:** Decoupling **timing** (wait-first, zero scenarios) from **routing** (MSA on go-only) reduces harmful wait/go competition in MSA and lowers mean tour cost vs full MSA.

---

## 4. Proposed method

### 4.1 Overview (Figure — PLACEHOLDER)

```
Request = full day instance
Router = cross-burst detector (geometric + time, not ML)
Specialists = policies at fork only
  - nearest (no specialist)
  - wait-first
  - MSA full (wait + go in pool)
  - hybrid (wait-first then MSA on go)
  - lazy (full MSA only if wait-first go target == nearest go)
```

### 4.2 Policies (contracts at fork)

| Policy | Timing | Routing | Scenarios |
|--------|--------|---------|-----------|
| Nearest | — | nearest | 0 |
| Wait-first | window heuristic | nearest | 0 |
| MSA full | in argmin | argmin over wait+go | M=20 |
| Hybrid | wait-first | MSA on go only | M on go subset |
| Lazy MSA | wait-first; else maybe full MSA | as MSA when triggered | conditional |

**Scenario model:** Resample unrevealed customers’ reveal times within cluster windows; mean completion cost after one action; nearest-feasible to end (Bent / SS-VRPTW-CR spirit).

### 4.3 Why it should work (pre-experiment)

- Wait-first encodes **Mitrović–Laporte** timing without Monte Carlo.
- Full MSA may **over-score wait** or **distract** with sibling go candidates — same spirit as “catalog siblings hurt RCM.”
- Hybrid removes wait from scenario pool; lazy avoids MSA when wait-first already disagrees with nearest on the discrete wait/go branch.

### 4.4 Properties (structural — not security)

- **P1 (Backbone consistency):** Off-fork behavior identical across all policies.
- **P2 (Same generator worlds):** Frozen bank ⇒ paired comparisons valid.
- **P3 (Compute):** Hybrid/lazy ≤ full MSA scenario calls in expectation (measured).

---

## 5. Experimental design

### 5.1 Evaluation objectives

| # | Question | Section |
|---|----------|---------|
| i | Do OR policies beat nearest on **tour cost**? | Main table |
| ii | Do hybrid/lazy beat **full MSA**? | vs MSA paired |
| iii | Does **M** matter? | ablation_M |
| iv | Is learning necessary? | uniform + MLP footnote |
| v | Compute tradeoff? | msa_calls, futures_scored |

### 5.2 Datasets

| Setting | Seeds | Test eps | n tour days |
|---------|-------|----------|-------------|
| Burst (primary) | 1234–1243 | 16–19 | 40 |
| Burst learned (secondary) | 1234–1236 | 16–19 | 12 |
| Uniform (negative) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

**Bank:** `data/processed/burst_instances_v1/seed_*.pkl`

### 5.3 Compared systems

All share: same instances, same backbone, same cross-burst definition.

- Nearest-feasible  
- Wait-first  
- Full MSA (M=20)  
- Hybrid wait + MSA-go  
- Lazy MSA  
- (Secondary) Learned wait+go MLP — bank-aligned train  

### 5.4 Metrics

- **Primary:** mean tour cost (lower better).  
- **Secondary:** Δ vs nearest, Δ vs MSA; paired mean Δ with bootstrap CI.  
- **Compute:** mean MSA calls / futures scored per seed.  
- **Do not lead with:** one-step top-1, label MSE.

### 5.5 Statistical analysis

- Paired on (seed, episode) for each policy vs nearest and vs MSA.  
- Report `frac_improved`, `frac_worsened` (if worsened > 0).  
- PLACEHOLDER: Holm correction if many contrasts; name test explicitly in final draft.

### 5.6 Reproducibility checklist

- [x] Frozen instance bank  
- [x] `scripts/run_burst_benchmark.py`  
- [ ] Exact generator parameter table in appendix  
- [ ] Git tag / Zenodo for logs  
- [ ] Commit hash in paper  

---

## 6. Results

### 6.1 Main table (10 seeds, M=20, burst)

| System | Mean tour | Std (seeds) | Δ vs nearest | Δ vs MSA |
|--------|-----------|-------------|--------------|----------|
| Nearest | **25.53** | 1.34 | — | — |
| Full MSA | 21.30 | 1.93 | +4.23 | — |
| Wait-first | 20.89 | 1.91 | +4.65 | +0.42 |
| Hybrid | 20.23 | 1.67 | +5.30 | +1.07 |
| **Lazy MSA** | **20.00** | 1.92 | **+5.53** | **+1.30** |

**Paired vs nearest:** 40/40 days improved for MSA, wait-first, hybrid, lazy (`frac_improved = 1.0`).

**Paired vs MSA (approximate):** hybrid ~77.5% days improved; lazy ~85% (see `paired_vs_msa` in log).

### 6.2 Compute

| Policy | Mean MSA calls / seed | Mean futures scored |
|--------|------------------------|---------------------|
| Full MSA | 801.9 | 57,008 |
| Hybrid | 561.5 | 35,582 |
| Lazy | 583.8 full-MSA invocations | (partial scoring) |

### 6.3 M ablation (aggregate)

| M | MSA mean | Hybrid mean | Nearest |
|---|----------|-------------|---------|
| 5 | 22.87 | 20.45 | 25.53 |
| 10 | (in log) | (in log) | 25.53 |
| 20 | 21.30 | 20.23 | 25.53 |

Hybrid beats full MSA at all M in aggregate log.

### 6.4 Uniform negative control

| | Tour / one-step |
|--|-----------------|
| Nearest | ~45.06 |
| Learned | ~45.09 |
| **Conclusion** | No beat; visible set insufficient |

### 6.5 Learning (secondary, 3 seeds, bank)

| | Mean tour |
|--|-----------|
| Nearest | 26.06 |
| Learned wait+go | 24.58 (+1.48) |
| Learned go-only | 26.34 (worse) |

OR lazy **20.00** (10-seed) remains primary.

### 6.6 Item-level utility (PLACEHOLDER — run if needed for paper)

Analogue of CTA-RAG Table `tab:utility`: paired flips vs nearest per episode — partially in `paired_vs_nearest`; extend with **worsened** counts for hybrid/MSA if any exist (currently 0 vs nearest for OR policies).

---

## 7. Claims and evidence (Table — copy to LaTeX)

| Claim | Evidence | Status |
|-------|----------|--------|
| Burst structure enables large tour gains | +5.5 vs nearest | **Supported** (40/40 paired) |
| Decoupled hybrid beats full MSA on mean tour | +1.07 mean; ~78% paired days | **Supported** (point + majority paired) |
| Lazy beats full MSA | +1.30 mean; ~85% paired days | **Supported** |
| Wait-first beats full MSA on mean | +0.42 | Point estimate |
| MLP beats nearest on burst (bank) | +1.48, 3 seeds | **Weak / secondary** |
| MLP beats OR | No | **Not supported** |
| Learning beats nearest on uniform | No | **Not supported** |
| Hybrid reduces scenario calls vs full MSA | 562 vs 802 | **Observed** |
| Method is global DVRP SOTA | Not tested | **Not claimed** |

---

## 8. Discussion

**Where specialization pays:** Only at **cross-burst** forks; burst generator required; uniform fails.

**Objective discipline:** Advisor-aligned — **tour cost only**; one-step accuracy misleading (uniform; go-only MLP).

**vs full MSA:** Timing and routing recourse should be **sequential**, not one argmin pool.

**Deployment:** Policies are deployable rules (no training); lazy saves scenarios.

**Limitations (open):** Synthetic generator; single vehicle; no accept/reject; no pickup-delivery; learned line unstable; no real city instances; no full online reoptimization baseline from literature code.

---

## 9. Limitations (reviewer-facing)

1. **Synthetic benchmark only** — external validity unknown.  
2. **Small test slice** — 4 episodes × 10 seeds = 40 tours (strong paired, but narrow).  
3. **No industry DVRP instance set.**  
4. **MLP not main claim** — do not oversell learning.  
5. **Wait oracle / old multiseed logs** — different harness; cite carefully.  
6. **Component ablation** — index vs wait-first vs MSA depth not fully isolated (PLACEHOLDER E3).  

---

## 10. Conclusion

We study dynamic CVRP with stochastic reveal times and **minimize realized tour distance**. A **burst benchmark** with frozen instances separates **uniform failure** from **burst success**. **Decoupled recourse** (hybrid and lazy MSA) **outperforms full scenario optimization** on mean tour cost while using **fewer** scenario calls. Learning does not replace OR on this testbed. Future work: real instances, sensitivity to burst parameters, abstention when MSA variance is high, optional distillation of MSA for compute.

---

## Appendix mapping to repo

| Artifact | Path |
|----------|------|
| Full benchmark | `outputs/logs/burst_benchmark_full.json` |
| Summary | `outputs/logs/BENCHMARK_SUMMARY.json` |
| Protocol | `docs/EXPERIMENTS_COMPLETE.md` |
| Figure CSV | `outputs/figures/burst_tour_costs.csv` |
| Commands | `scripts/run_burst_benchmark.py` |

---

## Author notes (not for submission — like main2.tex post `\end{document}`)

### A. What is done vs CTA-RAG-level rigor

| CTA-RAG element | Our status |
|-----------------|------------|
| Formal problem + taxonomy | Done in §3 (tour-level) |
| Threat model | N/A (OR sim); use **failure taxonomy** instead |
| Properties / propositions | Informal §4.4; could formalize routing bound if misroute added |
| Main table all baselines | Done (7 systems → we have 5 OR + nearest) |
| Holm-corrected tests | PLACEHOLDER — add if journal requires |
| External validation | **Missing** — need second generator or published instance |
| Rejected vs silent split | Analogue: **parse failures** minimal; all tours complete |
| Item-level utility table | Partial in paired JSON |
| Claims table | §7 |
| Attack experiments | N/A |

### B. Rejection risks (hostile reviewer)

1. **“Toy synthetic generator.”** → Acknowledge; position as **benchmark paper** + OR recourse idea.  
2. **“Only nearest backbone.”** → Intentional; cite learning-augmented algorithms literature.  
3. **“n=40 tours.”** → More seeds/episodes cheap to add.  
4. **“No comparison to Ulmer / published DVRP code.”** → PLACEHOLDER E4.  
5. **“Hybrid is heuristic soup.”** → Decouple is the **idea**; ablate wait-first off.  
6. **“MLP section weak.”** → Keep as negative/secondary.  

### C. Experiments still useful (optional)

- **E1:** More seeds (20) or episodes 16–19 → 16–39.  
- **E2:** Wait-first hyperparameter grid on val only.  
- **E3:** Ablate: hybrid without wait-first (MSA-go only); full MSA at M=5 compute-matched.  
- **E4:** One published dynamic instance or Solomon-style with reveal overlay.  
- **E5:** One cross-burst **map figure** (single decision snapshot).  
- **E6:** Report **frac_worsened** explicitly in `summarize_burst_results.py` for MSA vs hybrid.  
- **E7:** LaTeX elsarticle template + bibliography from `LITERATURE_DYNAMIC_VRP.md`.  

### D. Suggested LaTeX next step

1. Copy elsarticle skeleton from `main2.tex` (packages trimmed).  
2. Paste §1–10 from this file into `\section{}` blocks.  
3. One table: `\ref{tab:main}` = our mean tour table.  
4. One figure: bar chart from CSV.  
5. Bibliography: Saint-Guillain, Bent, Mitrović–Laporte, Psaraftis, Ulmer, CTIBench analogue = none — use Zhang survey instead.

---

*End of structure document.*
