"""Frozen-checkpoint confirmation on 32 new CVRP200 instances; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.probe_data import ProbeDataset
from scripts.generate_branch_bank import load_config
from scripts.generate_probe_bank import atomic_json, generate
from scripts.run_phase2b import make_model, predict, diagnostics, regrets, save_predictions

OUT = ROOT/'outputs/phase2c'
BANK = ROOT/'data/processed/probe_bank_phase2c'
SEED = 2026091801
COUNT = 32
MODELS = {'strong_weighted': 'strong', 'structural_weighted': 'structural'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initializer(path):
    """A dummy synthetic record for adapter initialization, never benchmark data."""
    rng = np.random.default_rng(871234)
    xy = rng.random((201, 2))
    fields = ['depot', *xy[0], 'customer', *xy[1:].ravel(), 'capacity', 50,
              'demand', 0, *([1]*200), 'cost', 0, 'node_flag', *range(1, 201), *([1]*200)]
    path.write_text(','.join(map(str, fields))+'\n', encoding='utf-8')


def freeze(out, bank):
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    source_files = [Path(__file__), ROOT/'scripts/generate_probe_bank.py',
                    ROOT/'scripts/generate_branch_bank.py', ROOT/'scripts/run_phase2b.py',
                    ROOT/'scripts/train_ranking_probes.py', ROOT/'models/regret_state_encoder.py',
                    ROOT/'models/structural_probe.py', ROOT/'data/probe_data.py',
                    ROOT/'policy/mnlp_adapter.py', ROOT/'policy/backbone.py',
                    ROOT/'planner/rollout.py', ROOT/'config.yaml',
                    Path(cfg['policy']['mnlp_root'])/'CVRP/VRPEnv.py',
                    Path(cfg['policy']['mnlp_root'])/'CVRP/VRPModel.py']
    spec = dict(version=1, instances=COUNT, seed=SEED, horizon=8, step_stride=1,
                top_k=5, customers=200, capacity=50, bank=str(bank),
                generation='Independent uniform unit-square coordinates, integer demands 1..9; frozen MnLP completions',
                no_training=True, no_interim_evaluation=True, final_128_benchmark='not opened or evaluated',
                selection='Phase 2B best.pt selected by original validation mean regret; no reselection',
                population='All multi-candidate decisions with original step + 8 < 200',
                weighting='Pooled decision-level mean; resample whole instances with replacement',
                bootstrap=dict(draws=10000, seed=2026091802, interval='two-sided percentile 95%, pointwise'),
                hypotheses={'H1': 'mean(strong_weighted)-mean(policy) < 0',
                            'H2': 'P90(strong_weighted)-P90(policy) < 0',
                            'H3': 'mean(strong_weighted)-mean(structural_weighted) < 0'},
                decision='Each hypothesis supported only if its 95% CI upper bound is below zero; '
                         'full confirmation requires all three. Pointwise intervals are not simultaneous.',
                secondary=['worst regret', 'top1', 'pairwise', 'high-policy-regret >0.1',
                           'policy-regret bins', 'oracle-best rank', 'per-instance metrics'],
                frozen_checkpoints={name: sha(ROOT/'outputs/phase2b'/name/'best.pt') for name in MODELS},
                policy_checkpoint=dict(path=cfg['policy']['checkpoint'], sha256=sha(cfg['policy']['checkpoint'])),
                source_sha256={str(p): sha(p) for p in source_files})
    protocol = out/'protocol.json'
    if protocol.exists():
        saved = json.loads(protocol.read_text())
        if saved['spec'] != spec:
            raise ValueError('Frozen protocol/checkpoints/source changed; refusing to resume')
    else:
        if bank.exists() and any(bank.glob('instance_*.npz')):
            raise ValueError('Cannot preregister after fresh instance shards already exist')
        (out/'frozen').mkdir(exist_ok=True)
        for name in MODELS:
            shutil.copyfile(ROOT/'outputs/phase2b'/name/'best.pt', out/'frozen'/(name+'.pt'))
        initializer(out/'initializer.txt')
        atomic_json(protocol, dict(registered_unix=time.time(), spec=spec,
                                  initializer_sha256=sha(out/'initializer.txt')))
    saved = json.loads(protocol.read_text())
    if sha(out/'initializer.txt') != saved['initializer_sha256']:
        raise ValueError('Initializer changed')
    for name, digest in spec['frozen_checkpoints'].items():
        if sha(out/'frozen'/(name+'.pt')) != digest:
            raise ValueError('Frozen checkpoint changed: '+name)
    return saved


def check_disjoint(bank, old_bank):
    def identities(path):
        with np.load(path/'instances.npz') as f:
            return [hashlib.sha256(x.tobytes()+d.tobytes()).hexdigest()
                    for x, d in zip(f['coords'], f['demands'])]
    fresh, old = identities(bank), identities(old_bank)
    if len(set(fresh)) != len(fresh) or set(fresh) & set(old):
        raise ValueError('Duplicate or reused confirmation instance')
    return dict(fresh_instances=len(fresh), previous_instances=len(old), overlaps=0,
                fresh_instance_sha256=fresh)


def hypotheses(predictions, draws=10000, seed=2026091802):
    p = predictions['strong_weighted']
    for other in predictions.values():
        for key in ('costs', 'valid', 'instance', 'nonterminal'):
            if not np.array_equal(p[key], other[key]):
                raise ValueError('Unpaired prediction records')
    mask = p['nonterminal'] & (p['valid'].sum(1)>1)
    ids = np.unique(p['instance'][mask])
    clusters = [np.flatnonzero(mask & (p['instance'] == i)) for i in ids]
    r = {name: regrets(value).astype(np.float64) for name, value in predictions.items()}
    def differences(ix):
        full, policy, structural = (r[name][ix] for name in ('strong_weighted', 'policy', 'structural_weighted'))
        return [(full-policy).mean(), np.quantile(full, .9)-np.quantile(policy, .9),
                (full-structural).mean()]
    rng = np.random.default_rng(seed)
    bootstrap = np.array([differences(np.concatenate([clusters[i] for i in rng.integers(len(ids), size=len(ids))]))
                          for _ in range(draws)])
    point = differences(np.flatnonzero(mask))
    ci = np.quantile(bootstrap, [.025, .975], axis=0)
    return {name: dict(delta=float(point[i]), ci95=ci[:, i].tolist(),
                       supported=bool(ci[1, i] < 0), instances=len(ids), draws=draws)
            for i, name in enumerate(('H1', 'H2', 'H3'))}


def evaluate(out, bank, count=COUNT, draws=10000):
    dataset = ProbeDataset(bank, range(count), 8)
    predictions = {}
    for name, kind in MODELS.items():
        checkpoint = torch.load(out/'frozen'/(name+'.pt'), map_location='cpu', weights_only=False)
        model = make_model(kind)
        model.load_state_dict(checkpoint['model'])
        predictions[name] = predict(model, dataset, kind, checkpoint['mean'], checkpoint['std'])
        save_predictions(out/(name+'_predictions.npz'), predictions[name])
    base = predictions['strong_weighted']
    predictions['policy'] = dict(base, scores=np.broadcast_to(-np.arange(5, dtype=float), base['scores'].shape))
    save_predictions(out/'policy_predictions.npz', predictions['policy'])
    results = dict(hypotheses=hypotheses(predictions, draws=draws),
                   metrics={name: diagnostics(p) for name, p in predictions.items()})
    results['all_three_supported'] = all(h['supported'] for h in results['hypotheses'].values())
    atomic_json(out/'results.json', results)
    lines = ['# Phase 2C: frozen H8 confirmation', '',
             'Fresh synthetic instances; frozen Phase 2B checkpoints; no retraining or interim model selection.', '',
             '| Model | Decisions | Mean regret | P90 regret | Worst regret | Top-1 | Pairwise |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, d in results['metrics'].items():
        m = d['overall']
        lines.append('| %s | %d | %.6f | %.6f | %.6f | %.4f | %.4f |' %
                     (name, m['groups'], m['mean_regret'], m['p90_regret'], m['worst_regret'],
                      m['top1_accuracy'], m['pairwise_accuracy']))
    lines += ['', 'Negative differences favor the full-state probe. Paired whole-instance percentile bootstrap.', '',
              '| Hypothesis | Delta | 95% CI | Supported |', '|---|---:|---|---|']
    for name, h in results['hypotheses'].items():
        lines.append('| %s | %.6f | [%.6f, %.6f] | %s |' %
                     (name, h['delta'], *h['ci95'], h['supported']))
    lines += ['', 'H1: mean full-state minus policy. H2: P90 full-state minus policy. '
              'H3: mean full-state minus weighted structural.',
              'All three supported: '+str(results['all_three_supported']),
              'Intervals are pointwise, not simultaneous. The protocol requires all three for full confirmation.',
              'Worst-case regret is reported separately; favorable mean/P90 does not establish extreme-tail safety.',
              'JSON includes regret bins, oracle-rank conditioning and per-instance metrics.',
              'These are offline actual-future-state results, not a learned future predictor or route-cost gains.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return results


def run():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/'pipeline.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **extra):
            atomic_json(OUT/'status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra))
            print('PHASE2C:', stage, extra, flush=True)
        try:
            freeze(OUT, BANK)
            if (OUT/'results.json').exists() and (OUT/'comparison.md').exists():
                status('complete', comparison=str(OUT/'comparison.md'))
                return
            status('generating', instances=COUNT, seed=SEED, bank=str(BANK))
            generate(BANK, count=COUNT, seed=SEED, initializer_data_path=OUT/'initializer.txt')
            audit = check_disjoint(BANK, ROOT/'data/processed/probe_bank_v2')
            if audit['fresh_instances'] != COUNT:
                raise ValueError('Incorrect confirmation sample size')
            atomic_json(OUT/'data_audit.json', audit)
            freeze(OUT, BANK)  # Detect code/checkpoint drift during a long generation run.
            status('evaluating', instances=COUNT)
            evaluate(OUT, BANK)
            status('complete', comparison=str(OUT/'comparison.md'))
        except Exception:
            status('failed', error=traceback.format_exc())
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--status', action='store_true')
    a = p.parse_args()
    if a.status:
        for path in (OUT/'status.json', BANK/'progress.json'):
            if path.exists():
                print(path, path.read_text())
        if (OUT/'status.json').exists():
            state = json.loads((OUT/'status.json').read_text())
            proc = Path('/proc')/str(state.get('pid', 0))/'cmdline'
            print('live:', proc.exists() and b'run_phase2c.py' in proc.read_bytes())
        return
    if a.launch:
        import shlex
        OUT.mkdir(parents=True, exist_ok=True)
        command = ' '.join(map(shlex.quote, [sys.executable, '-u', str(Path(__file__).resolve())]))
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'faros_phase2c',
                        command+' >> '+shlex.quote(str(OUT/'pipeline.log'))+' 2>&1'], check=True)
        print('Launched faros_phase2c:', OUT/'pipeline.log')
        return
    torch.set_num_threads(2)
    run()


if __name__ == '__main__':
    main()
