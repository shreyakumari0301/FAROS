# After the deadline — what could improve results (honest priority)

You are already at **~20.0 tour cost (lazy)** vs **25.5 nearest** and **~1.3 below full MSA**. Further gains are **diminishing**; writing the current story is the best ROI.

## Worth trying only if a reviewer asks or you extend the paper

| Idea | Expected gain | Cost | Notes |
|------|---------------|------|--------|
| **Tune wait-first** (IMMINENT_HORIZON, WINDOW_TOL) | Small (0.2–0.5) | Low | Grid on val eps 12–15 only; retest 16–19 once |
| **Higher M at forks** (M=40) | Tiny vs M=20 | Medium | Ablation already shows M=20 ≈ sweet spot for hybrid |
| **Stronger lazy rule** (e.g. MSA when wait-first go ≠ hybrid argmin) | Uncertain | Medium | Risk overfitting one generator |
| **Distill MSA → MLP** (CE on teacher actions) | Unlikely to beat lazy | High | Probe failed; needs architecture + more data |
| **More seeds (20+)** | Tighter CIs only | Low compute | Does not change mean much |
| **Real / published dynamic instance** | Credibility, not synthetic score | High | Different project phase |

## Unlikely to help on *this* benchmark

- **Better one-step accuracy** — advisor point; tours did not follow.
- **Full neural routing** — contradicts frozen-backbone design.
- **Uniform reveals** — structurally hard; keep as negative only.
- **Go-only learned fork** — already **worse** than nearest on bank eval.

## Best “improvement” for the write-up (no new runs)

1. One **cross-burst map** figure (manual snapshot from simulator).
2. Sentence on **wait-first beating full MSA** on mean tour (timing decoupling).
3. **Compute paragraph:** hybrid ~30% fewer scenario calls than full MSA with better tour cost.

## Recommendation

**Do not chase lower tour cost before submitting.** Lead with lazy/hybrid vs MSA + benchmark + uniform negative. Revisit learning only in a **second paper** with distillation or a standard DVRP dataset.
