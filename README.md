# FAROS — Phase 0/1

Clean CVRP foundation + counterfactual branch bank for later FAROS research.

## Phase 0 (complete)

Frozen **MnLP** backbone on CVRP200 with exact cost parity vs LaRP (`20.3511238098`).

```bash
wsl -e bash -lc "cd /mnt/c/Users/SK/FAROS && /home/sk/miniconda3/envs/mnlp/bin/python scripts/parity_check.py"
wsl -e bash -lc "cd /mnt/c/Users/SK/FAROS && /home/sk/miniconda3/envs/mnlp/bin/python scripts/run_greedy.py"
```

## Phase 1 — branch bank

```bash
# 5-instance full trajectory (stride=1); resume-safe shards under data/processed/branch_bank_v1/
wsl -e bash -lc "cd /mnt/c/Users/SK/FAROS && /home/sk/miniconda3/envs/mnlp/bin/python scripts/generate_branch_bank.py --resume"

wsl -e bash -lc "cd /mnt/c/Users/SK/FAROS && /home/sk/miniconda3/envs/mnlp/bin/python scripts/inspect_branch_bank.py --path data/processed/branch_bank_v1 --index 20"
wsl -e bash -lc "cd /mnt/c/Users/SK/FAROS && /home/sk/miniconda3/envs/mnlp/bin/python scripts/analyze_branch_bank.py --path data/processed/branch_bank_v1"
```

Each candidate stores:
- `post_action_state` = exact T(s_t, a)
- `rollout_states[H]` = s_{t+H} after action + (H-1) greedy steps (H=1 equals post-action)

Do **not** train FAROS until Phase 1B/1C analysis identifies which horizon/features carry ranking signal.

## Dynamic burst CVRP (advisor line — current gate)

Independent **uniform** reveals: models fail to beat nearest (`outputs/logs/arrival_value.json`).

**Burst** waves + learned **wait/go** at cross-burst stops: beats nearest on **full tours**
(see `outputs/logs/burst_learned_wait.json`, protocol in `docs/dynamic_burst_benchmark.md`).

```powershell
.\.venv\Scripts\python.exe scripts\summarize_burst_results.py
.\.venv\Scripts\python.exe scripts\train_burst_wait.py
.\.venv\Scripts\python.exe scripts\export_burst_dataset.py --seed 1234
```

Draft advisor mail: `docs/ADVISOR_EMAIL.txt`.

## Phase 2: actual future-state representation probes

This experiment trains ranking probes, not a FAROS future-representation predictor.
The Phase 1 bank contains only structural summaries beyond H=1, so Phase 2 uses
a new `data/processed/probe_bank_v2` bank with full future-state snapshots.
Coordinates and demands are shared per instance; each branch stores its visited
mask, current node, capacity, step, termination flag, and incurred prefix cost at
H=1,2,4,8. The compact NPZ files keep completion-cost targets separate from inputs.

Data: 100 independent synthetic CVRP200 instances, seed 20260916, uniform unit-square
coordinates, integer demands 1..9, capacity 50. The existing benchmark is excluded.
The adapter initially reads one benchmark record to initialize its environment,
then all raw problem/reference arrays are replaced before any generation.
Dummy reference routes only satisfy the environment interface; labels come from
frozen MnLP greedy completions. Split: 0..79 train, 80..89 validation, 90..99 test.

There are eight probes: a small structural MLP and a shared 2-layer, width-32
Transformer architecture independently trained at each horizon. All use pairwise
logistic ranking loss (higher score means lower completion cost), excluding cost
ties within 1e-6 and averaging first within decision groups. Training uses seed 1234,
AdamW at 3e-4, batch size 16, at most 20 epochs and patience 5. Model selection uses
validation mean regret on multi-candidate states that cannot terminate within H=8.
Feature normalization is fitted only to training instances. Test instances do not
participate in checkpoint selection. This first run is one seed, not a definitive
statistical decision on the FAROS hypothesis.

Both models receive matched known-cost context: capacity ratio, cost incurred
since the original decision, termination flag, and normalized step. This makes
total-completion comparisons well-defined; scores never receive completion cost,
regret, candidate rank, or policy probability. A policy-ranking baseline and an
incurred-cost-only baseline help distinguish future-state signal from known cost.
The main comparison excludes states within 8 decisions of termination, identically
for all horizons. JSON results also include all states, multi-candidate states,
and per-instance metrics. Top-1 accepts any cost-tied optimum; predicted pairwise
ties receive half credit. Selected regret is measured in original route-cost units.
These are offline branch-selection metrics, not closed-loop routing improvements.

```bash
cd /mnt/c/Users/SK/FAROS
/home/sk/miniconda3/envs/mnlp/bin/python scripts/test_phase2.py
/home/sk/miniconda3/envs/mnlp/bin/python scripts/run_phase2.py --launch
tmux attach -t faros_phase2
```

The detached `faros_phase2` session generates all 100 instances then trains all
eight probes automatically. Run the same launch command after a stopped process
to resume. An exclusive process lock prevents overlapping pipelines. Generation
validates its manifest and completed NPZ archives, writes temporary shards then
atomically renames them, and restarts only the interrupted instance. Training saves
atomic latest/best checkpoints and resumes at epoch boundaries. Keep configuration
unchanged when resuming; changed experiments require new output directories.

Monitor `data/processed/probe_bank_v2/progress.json`, `outputs/phase2/status.json`,
and `outputs/phase2/pipeline.log`. Final reports: `outputs/phase2/comparison.md`
and `comparison.json`; each probe also has its config, history, and checkpoints.
On the earlier CPU run, generation alone took approximately 27 hours per 100
instances; full-state probe training adds time. Prevent machine sleep during runs.

## Phase 2B: H=8 regret-sensitive probes

Reuse the completed 100-instance bank and the original 80/10/10 split. This is
an exploratory follow-up because Phase 2 test results informed the hypothesis;
confirmation requires fresh held-out instances. No future-state predictor is trained.

The original probe already has a two-layer width-32 node Transformer and an
unweighted pairwise loss. Phase 2B adds four controlled runs: structural weighted,
original full-state weighted, stronger full-state unweighted, and stronger full-state
weighted. The stronger encoder uses three width-64 attention layers, explicit
depot/current-node readouts, remaining-node pooling, and capacity/context-conditioned
attention. All receive the same label-free H=8 inputs and known incurred-cost context.
Node identity comes from state flags, not arbitrary node ordering.

Weighted loss multiplies each better-versus-worse logistic term by its absolute
completion-cost gap, then averages by valid pair count within each decision and
over decisions with nontied pairs. It does not normalize by the sum of cost gaps,
so high-consequence decisions retain greater weight. Training remains seed 1234,
AdamW 3e-4, batch size 16, up to 20 epochs, patience 5, and checkpoint selection
by validation nonterminal mean regret only. Original Phase 2 checkpoints stay intact.

```bash
/home/sk/miniconda3/envs/mnlp/bin/python scripts/test_phase2b.py
/home/sk/miniconda3/envs/mnlp/bin/python scripts/run_phase2b.py --launch
tmux attach -t faros_phase2b
```

Outputs are in `outputs/phase2b/`: `diagnostics.md/json` reproduces and analyzes
the original H=8 results; `comparison.md/json` adds each completed new variant.
Reports include mean, P90 and worst regret, top-1 and pairwise accuracy, disjoint
policy-regret bins, oracle-best policy rank, high-policy-regret (>0.1) states,
per-instance metrics, and each bin's contribution to the overall mean improvement.
Oracle cost ties use the lowest policy rank among tied best candidates. Predictions
are saved for auditing. Paired 2,000-resample instance-cluster bootstrap intervals
compare mean and P90 regret with policy and structural baselines; they are exploratory
and not corrected for multiple comparisons.

The main representation question is whether the stronger weighted encoder improves
on the equally weighted structural baseline, not only on policy. Assess mean regret
and tail behavior together; a tiny point difference is not evidence of unique state
information. Even a positive result remains an actual-state offline result, not a
closed-loop routing improvement or justification by itself for a JEPA-style predictor.
`status.json`, `pipeline.log`, and per-variant histories track progress. Relaunching
the same command resumes at epoch boundaries under an exclusive process lock.

## Phase 2C: fresh-test confirmation

Evaluate the frozen Phase 2B `strong_weighted` and `structural_weighted` best
checkpoints against policy on 32 new synthetic CVRP200 instances (seed 2026091801).
No training, threshold tuning, interim evaluation, or new checkpoint selection occurs.
The final 128-instance benchmark is never opened: adapter initialization uses a
separate dummy synthetic text record. Generation uses the existing branch-bank
format and exact frozen MnLP completion labels; only H=8 is evaluated.

Before generation, `outputs/phase2c/protocol.json` records the hypotheses, sample
size, seed, metrics, checkpoint hashes and source hashes. Checkpoints are copied
to `outputs/phase2c/frozen/`. Resume rejects source or frozen-artifact changes.
The new bank is `data/processed/probe_bank_phase2c`; exact instance fingerprints
are checked for duplicates and overlap with the previous 100 synthetic instances.

Preregistered differences (negative favors full-state): H1 mean full-state minus
policy, H2 P90 full-state minus policy, H3 mean full-state minus weighted structural.
Use 10,000 paired whole-instance bootstrap resamples, seed 2026091802, and two-sided
95% percentile intervals. Each hypothesis requires its upper interval endpoint
below zero; full confirmation requires all three. Intervals are pointwise, not
simultaneous. Report worst regret and ranking accuracies as secondary metrics.
All methods use the same nonterminal multi-candidate states as Phase 2B.

```bash
/home/sk/miniconda3/envs/mnlp/bin/python scripts/test_phase2c.py
/home/sk/miniconda3/envs/mnlp/bin/python scripts/run_phase2c.py --launch
/home/sk/miniconda3/envs/mnlp/bin/python scripts/run_phase2c.py --status
```

The detached `faros_phase2c` session generates all 32 instances, evaluates the
frozen models, and writes `outputs/phase2c/comparison.md`, `results.json`, and
auditable prediction archives. Monitor `status.json`, `pipeline.log`, and the
bank's `progress.json`. Relaunch the same command after interruption to resume
completed generation shards. This run does not train the FAROS future predictor.

## Final FAROS: Future-Aware Ranking for Optimization Search

The final method predicts relative branch outcomes from exact H=1 post-action
states. It does not predict future states. `models/faros_outcome.py` reuses the
width-64 three-layer state-encoder architecture, initialized from scratch, and
adds one small candidate-set attention layer plus nonnegative regret and risk-logit
heads. There are no candidate positional embeddings. Invalid candidates are excluded
from state encoding, masked in candidate attention and losses, and cannot be selected.

Targets are `r_i = C_i - min_valid(C)` and `y_i = 1[r_i > delta]`. The loss combines
listwise KL, regret-weighted Huber with weight `1 + alpha*r_i`, and BCE on risk logits.
The sole fixed loss preset is tau=0.1, alpha=1, lambda=1, gamma=0.1, delta=0.1;
there is no loss-preset search. Validation jointly selects beta from {0, 0.05, 0.1}
and the checkpoint by mean regret; ties prefer smaller beta then earlier epoch.
Selection is exactly `argmin(r_hat + beta*sigmoid(risk_logit))`, with no confidence
gate. Exact score ties use the existing policy candidate order.

Training uses the original 80 instances, validation 10, seed 1234, AdamW 3e-4,
batch size 16, at most 20 epochs and patience 5. Only exact H=1 nodes/context enter
the model. Completion costs are targets, not features. The old ten-instance test
comparison against policy and actual H8 probes is explicitly exploratory and runs
only after checkpoint and beta are frozen. Validation label prevalence is recorded;
it does not trigger additional hyperparameter searches.

The decisive comparison is Greedy vs FAROS vs Explicit Rollout on 32 identical new
synthetic CVRP200 instances, seed 2026091901. Explicit Rollout performs full greedy
completion for each current top-five candidate at each decision; it is not SGBS.
Method order cycles through all six permutations. Runtime includes exact branching
and FAROS inference, with model loading and shared training-instance warmup excluded.
Each route is independently checked for customer coverage, capacity and cost.
Report mean route cost, paired cost difference from greedy with 10,000 instance
bootstrap resamples, runtime and policy-forward counts. No reference gap is claimed
because these synthetic instances have no reference solutions. The final 128-instance
benchmark is not opened. Fresh instance fingerprints must be disjoint from both
previous synthetic banks.

Training labels take one candidate then revert to greedy; deployment repeatedly
uses FAROS. Therefore offline branch regret is a surrogate and closed-loop route
performance is decisive. This is one final model; no gates or architecture changes
are introduced in response to fresh-test results.

```bash
/home/sk/miniconda3/envs/mnlp/bin/python scripts/test_faros_final.py
/home/sk/miniconda3/envs/mnlp/bin/python scripts/run_faros_final.py --launch
```

Outputs live in `outputs/faros_final/`: preregistered `protocol.json`, training
`history.json`, `status.json`, `pipeline.log`, selected `frozen.pt` and
`frozen_config.json`, exploratory `offline.md`, and final `comparison.md`.
The `faros_final` tmux session automatically runs training, freezing, offline
evaluation and all fresh routes. Relaunch to resume at epoch boundaries or at the
next unfinished method/instance route. Source/data hashes prevent silent changes
to a resumed experiment.
