"""Final FAROS: train once, freeze on validation, offline then fresh closed-loop test."""
import argparse
import hashlib
import itertools
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
from torch.utils.data import DataLoader
from data.probe_data import ProbeDataset
from models.faros_outcome import FAROSOutcome, outcome_loss, selection_score
from scripts.generate_probe_bank import atomic_json, install_synthetic
from scripts.generate_branch_bank import load_config
from scripts.train_ranking_probes import atomic_torch
from scripts.run_phase2b import diagnostics, save_predictions
from scripts.run_phase2c import initializer, sha
from scripts.faros_routing import route
from policy.backbone import load_policy

OUT = ROOT/'outputs/faros_final'
BANK = ROOT/'data/processed/probe_bank_v2'
SPLITS = dict(train=list(range(80)), validation=list(range(80, 90)), test=list(range(90, 100)))
SETTINGS = dict(tau=.1, alpha=1., **{'lambda': 1., 'gamma': .1}, delta=.1,
                beta_grid=[0., .05, .1], seed=1234, epochs=20, patience=5,
                batch_size=16, learning_rate=3e-4, weight_decay=1e-4,
                fresh_instances=32, fresh_seed=2026091901, bootstrap_seed=2026091902,
                bootstrap_draws=10000, horizon=1, top_k=5, threads=2)


def status(out, stage, **extra):
    atomic_json(out/'status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra))


def protocol(out):
    cfg = load_config()
    paths = [ROOT/p for p in ('models/faros_outcome.py', 'models/regret_state_encoder.py',
             'scripts/run_faros_final.py', 'scripts/faros_routing.py', 'data/probe_data.py',
             'scripts/run_phase2b.py', 'scripts/run_phase2c.py', 'scripts/train_ranking_probes.py',
             'scripts/generate_probe_bank.py', 'scripts/generate_branch_bank.py',
             'planner/rollout.py', 'policy/mnlp_adapter.py', 'policy/backbone.py', 'config.yaml')]
    paths += [Path(cfg['policy']['mnlp_root'])/'CVRP'/name for name in ('VRPEnv.py', 'VRPModel.py')]
    spec = dict(settings=SETTINGS, splits=SPLITS, source_sha256={str(p): sha(p) for p in paths},
                bank_sha256={p.name: sha(p) for p in sorted(BANK.glob('instance_*.npz'))},
                policy_sha256=sha(cfg['policy']['checkpoint']),
                model='Width64 three-layer state encoder, one-layer candidate-set Transformer; initialized from scratch',
                target='Within-decision r_i=C_i-min_valid(C); y_i=1[r_i>delta]',
                loss='KL(softmax(-r/tau)||softmax(-r_hat/tau)) + lambda*weighted Huber + gamma*BCE',
                weights='Huber weight 1+alpha*r; losses average valid candidates then decisions',
                fixed_loss_preset='One preset, no loss hyperparameter search; delta=.1 fixed before validation and test',
                selection='Checkpoint and beta jointly minimize validation mean regret on nonterminal multi-candidate states; '
                          'beta ties prefer smallest beta, epoch ties earliest. No gate.',
                input='Exact H=1 nodes/context only; no policy rank, probability, H8 state or completion label inputs',
                deployment='argmin(r_hat + beta*sigmoid(risk_logits)); padded candidates masked; exact ties prefer policy order',
                mismatch='Labels revert to greedy after one candidate; deployment repeatedly chooses FAROS',
                old_test='Original ten-instance test is exploratory; never used for selection',
                fresh_test='32 new CVRP200 synthetic instances; final128 benchmark not opened; no reference solver available',
                comparator='Explicit rollout: full greedy completion for each current top5 candidate at every step, no beam search',
                timing='CPU two threads, common policy/model loading excluded, all per-route compute included; '
                       'method order cycles through six permutations; warmup uses training instance 0',
                primary='Paired FAROS-minus-greedy route cost, two-sided 95% whole-instance bootstrap CI; negative favors FAROS',
                secondary='Mean route cost, percent difference, runtime, policy forward passes, FAROS inference time',
                stop_rule='One final method only; no architecture changes or gate after fresh results')
    path = out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['spec'] != spec:
            raise ValueError('Protocol/source/data changed; refusing resume')
    else:
        atomic_json(path, dict(registered_unix=time.time(), spec=spec))
    return spec


@torch.no_grad()
def predict(model, dataset, batch_size=16):
    model.eval()
    stores = {k: [] for k in ('regret', 'risk_logits', 'costs', 'valid', 'nonterminal', 'instance')}
    for b in DataLoader(dataset, batch_size=batch_size):
        r, risk = model(b['nodes'], b['context'], b['valid'])
        stores['regret'].append(r.numpy())
        stores['risk_logits'].append(risk.numpy())
        for key in ('costs', 'valid', 'nonterminal', 'instance'):
            stores[key].append(b[key].numpy())
    return {key: np.concatenate(value) for key, value in stores.items()}


def scored(p, beta):
    scores = -selection_score(torch.from_numpy(p['regret']), torch.from_numpy(p['risk_logits']),
                              torch.from_numpy(p['valid']), beta).numpy()
    # Metric functions ignore padded pairs; use finite padding to avoid inf-inf warnings.
    scores[~p['valid']] = 0.
    return dict(p, scores=scores)


def train(out, datasets, cfg):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    model = FAROSOutcome()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start, best, stale, history = 0, float('inf'), 0, []
    if (out/'last.pt').exists():
        c = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        start, best, stale, history = c['epoch']+1, c['best'], c['stale'], c['history']
    for epoch in range(start, cfg['epochs']):
        if stale >= cfg['patience']:
            break
        torch.manual_seed(cfg['seed']+epoch)
        loader = DataLoader(datasets['train'], batch_size=cfg['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        model.train()
        started, total = time.time(), 0.
        status(out, 'training', epoch=epoch+1, batches_completed=0, batches=len(loader))
        for i, b in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            r, logits = model(b['nodes'], b['context'], b['valid'])
            loss = outcome_loss(r, logits, b['costs'], b['valid'], cfg)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
            if (i+1) % 100 == 0:
                status(out, 'training', epoch=epoch+1, batches_completed=i+1, batches=len(loader))
        p = predict(model, datasets['validation'], cfg['batch_size'])
        trials = [(diagnostics(scored(p, beta))['overall'], beta) for beta in cfg['beta_grid']]
        metrics, beta = min(trials, key=lambda x: (x[0]['mean_regret'], x[1]))
        if metrics['mean_regret'] < best:
            best, stale = metrics['mean_regret'], 0
            atomic_torch(out/'best.pt', dict(model=model.state_dict(), epoch=epoch+1, beta=beta, settings=cfg))
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, loss=total/len(loader), beta=beta, validation=metrics,
                            seconds=time.time()-started,
                            beta_candidates=[dict(beta=b, validation=m) for m, b in trials]))
        atomic_json(out/'history.json', history)
        atomic_torch(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, best=best, stale=stale, history=history))
        print('FAROS epoch', history[-1], flush=True)
    return torch.load(out/'best.pt', map_location='cpu', weights_only=False)


def freeze(out, checkpoint):
    path = out/'frozen_config.json'
    if not path.exists():
        shutil.copyfile(out/'best.pt', out/'frozen.pt')
        atomic_json(path, dict(frozen_unix=time.time(), checkpoint_sha256=sha(out/'frozen.pt'),
                    selected_epoch=checkpoint['epoch'], beta=checkpoint['beta'], settings=checkpoint['settings'],
                    protocol_sha256=sha(out/'protocol.json'), selection='validation only', no_gate=True))
    frozen = json.loads(path.read_text())
    if sha(out/'frozen.pt') != frozen['checkpoint_sha256'] or sha(out/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Frozen model/config mismatch')
    return frozen


def offline(out, model, beta):
    p = scored(predict(model, ProbeDataset(BANK, SPLITS['test'], 1)), beta)
    save_predictions(out/'offline_predictions.npz', p)
    predictions = {'faros_h1': p,
                   'policy': dict(p, scores=np.broadcast_to(-np.arange(5, dtype=float), p['scores'].shape))}
    for name in ('structural_weighted', 'strong_weighted'):
        with np.load(ROOT/'outputs/phase2b'/name/'predictions.npz') as f:
            other = {k: f[k] for k in f.files}
        for key in ('costs', 'valid', 'instance', 'nonterminal'):
            if not np.array_equal(p[key], other[key]):
                raise ValueError('Offline baseline alignment mismatch')
        predictions[name+'_h8'] = other
    results = {name: diagnostics(value) for name, value in predictions.items()}
    atomic_json(out/'offline_results.json', results)
    lines = ['# Exploratory offline comparison', '', 'Original reused ten test instances; H8 baselines receive actual future states.', '',
             '| Method | Mean regret | P90 | Worst | Top-1 |', '|---|---:|---:|---:|---:|']
    for name, d in results.items():
        m = d['overall']
        lines.append('| %s | %.6f | %.6f | %.6f | %.4f |' %
                     (name, m['mean_regret'], m['p90_regret'], m['worst_regret'], m['top1_accuracy']))
    (out/'offline.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def instance_fingerprints(coords, demands):
    return [hashlib.sha256(x.tobytes()+d.tobytes()).hexdigest() for x, d in zip(coords, demands)]


def summarize_routes(out, count, draws=10000):
    methods = ('greedy', 'faros', 'explicit_rollout')
    rows = {m: [json.loads((out/'routes'/('%03d_%s.json' % (i, m))).read_text()) for i in range(count)] for m in methods}
    costs = {m: np.array([r['route_cost'] for r in rs]) for m, rs in rows.items()}
    rng = np.random.default_rng(SETTINGS['bootstrap_seed'])
    ix = rng.integers(count, size=(draws, count))
    report = {}
    for method in methods:
        delta = costs[method]-costs['greedy']
        ci = np.quantile(delta[ix].mean(1), [.025, .975]).tolist()
        report[method] = dict(instances=count, mean_route_cost=float(costs[method].mean()),
                             mean_minus_greedy=float(delta.mean()), ci95=ci,
                             percent_vs_greedy=float(100*(costs[method].mean()/costs['greedy'].mean()-1)),
                             mean_seconds=float(np.mean([r['seconds'] for r in rows[method]])),
                             mean_policy_forward_passes=float(np.mean([r['policy_forward_passes'] for r in rows[method]])),
                             mean_faros_inference_seconds=float(np.mean([r['faros_inference_seconds'] for r in rows[method]])),
                             reference_gap=None)
    atomic_json(out/'closed_loop_results.json', report)
    lines = ['# Final FAROS closed-loop comparison', '',
             '32 fresh synthetic CVRP200 instances. Negative paired cost differences favor the method.',
             'All methods use identical instances; explicit rollout is not SGBS. No reference solver was run, so no reference/optimality gap is claimed.', '',
             '| Method | Mean cost | Minus greedy | 95% CI | Mean seconds | Policy forwards | FAROS inference seconds |',
             '|---|---:|---:|---|---:|---:|---:|']
    for name, m in report.items():
        lines.append('| %s | %.6f | %.6f | [%.6f, %.6f] | %.3f | %.1f | %.3f |' %
                     (name, m['mean_route_cost'], m['mean_minus_greedy'], *m['ci95'],
                      m['mean_seconds'], m['mean_policy_forward_passes'], m['mean_faros_inference_seconds']))
    lines += ['', 'Intervals use 10,000 paired whole-instance bootstrap samples. Total time includes exact branching and FAROS inference.',
              'Training labels follow greedy completion after one action; deployment repeatedly follows FAROS.',
              'This is the final method; no test-driven gates or architecture changes.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def closed_loop(out, model, frozen):
    cfg = load_config()
    init = out/'initializer.txt'
    if not init.exists():
        initializer(init)
    policy = load_policy(name='mnlp', checkpoint=cfg['policy']['checkpoint'],
                         mnlp_root=cfg['policy']['mnlp_root'], data_path=str(init), device='cpu', max_episodes=1)
    # Warm up only on an existing training problem, outside measured routes.
    coords, demands = install_synthetic(policy, 100, 20260916)
    route(policy, 0, 'greedy')
    route(policy, 0, 'faros', model, frozen['beta'])
    count = SETTINGS['fresh_instances']
    coords, demands = install_synthetic(policy, count, SETTINGS['fresh_seed'])
    fingerprints = instance_fingerprints(coords, demands)
    for old in (BANK/'instances.npz', ROOT/'data/processed/probe_bank_phase2c/instances.npz'):
        with np.load(old) as f:
            if set(fingerprints) & set(instance_fingerprints(f['coords'], f['demands'])):
                raise ValueError('Fresh instance overlaps previously used problems')
    if len(set(fingerprints)) != count:
        raise ValueError('Duplicate fresh instance')
    atomic_json(out/'fresh_data_audit.json', dict(instances=count, seed=SETTINGS['fresh_seed'],
                fingerprints=fingerprints, overlaps=0, benchmark_opened=False))
    save_predictions(out/'fresh_instances.npz', dict(coords=coords, demands=demands))
    folder = out/'routes'
    folder.mkdir(exist_ok=True)
    orders = list(itertools.permutations(('greedy', 'faros', 'explicit_rollout')))
    for instance in range(count):
        for method in orders[instance % len(orders)]:
            path = folder/('%03d_%s.json' % (instance, method))
            if path.exists():
                row = json.loads(path.read_text())
                if row['instance_sha256'] != fingerprints[instance] or row['checkpoint_sha256'] != frozen['checkpoint_sha256']:
                    raise ValueError('Saved route provenance mismatch')
                continue
            status(out, 'closed_loop', active_instance=instance+1, instances=count, method=method,
                   completed_routes=len(list(folder.glob('*.json'))))
            row = route(policy, instance, method, model, frozen['beta'])
            row.update(instance_sha256=fingerprints[instance], checkpoint_sha256=frozen['checkpoint_sha256'])
            atomic_json(path, row)
            print('ROUTE', instance+1, method, row['route_cost'], row['seconds'], flush=True)
    summarize_routes(out, count, SETTINGS['bootstrap_draws'])


def run(out=OUT):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'pipeline.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protocol(out)
            if (out/'frozen_config.json').exists():
                frozen = freeze(out, None)
            else:
                datasets = {name: ProbeDataset(BANK, ids, 1) for name, ids in SPLITS.items() if name != 'test'}
                # Record validation label prevalence before training; it does not alter the fixed preset.
                costs = np.concatenate([a['costs'] for a in datasets['validation'].arrays])
                valid = np.concatenate([a['valid'] for a in datasets['validation'].arrays])
                r = costs-np.where(valid, costs, np.inf).min(1)[:, None]
                atomic_json(out/'validation_setup.json', dict(settings=SETTINGS,
                            harmful_candidate_fraction=float((r[valid]>SETTINGS['delta']).mean())))
                checkpoint = train(out, datasets, SETTINGS)
                frozen = freeze(out, checkpoint)
                del datasets
            model = FAROSOutcome().eval()
            model.load_state_dict(torch.load(out/'frozen.pt', map_location='cpu', weights_only=False)['model'])
            status(out, 'offline_evaluation')
            offline(out, model, frozen['beta'])
            protocol(out)
            closed_loop(out, model, frozen)
            status(out, 'complete', comparison=str(out/'comparison.md'))
        except Exception:
            status(out, 'failed', error=traceback.format_exc())
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    if a.launch:
        import shlex
        OUT.mkdir(parents=True, exist_ok=True)
        command = ' '.join(map(shlex.quote, [sys.executable, '-u', str(Path(__file__).resolve())]))
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'faros_final',
                        command+' >> '+shlex.quote(str(OUT/'pipeline.log'))+' 2>&1'], check=True)
        print('Launched faros_final:', OUT/'pipeline.log')
        return
    torch.set_num_threads(SETTINGS['threads'])
    run()


if __name__ == '__main__':
    main()
