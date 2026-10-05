"""Resumable H8 diagnostics and controlled regret-sensitive probe experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data.probe_data import ProbeDataset
from models.future_state_encoder import FutureStateEncoder
from models.regret_state_encoder import RegretStateEncoder
from models.structural_probe import StructuralProbe
from scripts.generate_probe_bank import atomic_json
from scripts.train_ranking_probes import atomic_torch, metric_summary, ranking_loss, score_batch, TIE_TOL

SPLITS = dict(train=list(range(80)), validation=list(range(80, 90)), test=list(range(90, 100)))
VARIANTS = (('structural_weighted', 'structural', True),
            ('full_state_weighted', 'full_state', True),
            ('strong_unweighted', 'strong', False),
            ('strong_weighted', 'strong', True))


def weighted_loss(scores, costs, valid):
    gaps = costs[:, None, :] - costs[:, :, None]
    pairs = (gaps > TIE_TOL) & valid[:, :, None] & valid[:, None, :]
    counts = pairs.sum((1, 2))
    active = counts > 0
    if not active.any():
        return scores.sum() * 0
    losses = F.softplus(-(scores[:, :, None] - scores[:, None, :]))
    # Divide by pair count, NOT gap sum: large-cost decisions retain higher weight.
    return ((losses * gaps.clamp_min(0) * pairs).sum((1, 2)) /
            counts.clamp_min(1))[active].mean()


def make_model(kind):
    return {'structural': StructuralProbe, 'full_state': FutureStateEncoder,
            'strong': RegretStateEncoder}[kind]()


@torch.no_grad()
def predict(model, dataset, kind, mean, std, batch_size=16):
    model.eval()
    stores = {k: [] for k in ('scores', 'costs', 'valid', 'nonterminal', 'instance')}
    for b in DataLoader(dataset, batch_size=batch_size):
        stores['scores'].append(score_batch(model, b, kind, 'cpu', mean, std).numpy())
        for k in stores:
            if k != 'scores':
                stores[k].append(b[k].numpy())
    return {k: np.concatenate(v) for k, v in stores.items()}


def regrets(p):
    costs, valid = p['costs'], p['valid']
    chosen = np.where(valid, p['scores'], -np.inf).argmax(1)
    return costs[np.arange(len(costs)), chosen] - np.where(valid, costs, np.inf).min(1)


def metrics(p, mask):
    m = metric_summary(p['scores'], p['costs'], p['valid'], mask)
    if m['groups']:
        r = regrets(p)[mask]
        m.update(p90_regret=float(np.quantile(r, .9)), worst_regret=float(r.max()))
    return m


def diagnostics(p):
    costs, valid = p['costs'], p['valid']
    mask = p['nonterminal'] & (valid.sum(1) > 1)
    best = np.where(valid, costs, np.inf).min(1)
    policy_r = costs[:, 0] - best
    # Disjoint bins: [0,.01], (.01,.05], (.05,.10], (.10,.25], (.25,inf).
    bounds = (0.01, .05, .1, .25, np.inf)
    labels = ('0-0.01', '0.01-0.05', '0.05-0.10', '0.10-0.25', '>0.25')
    bins = {}
    lower = -np.inf
    for label, upper in zip(labels, bounds):
        select = mask & (policy_r > lower) & (policy_r <= upper)
        m = metrics(p, select)
        if select.any():
            improvement = policy_r[select] - regrets(p)[select]
            m['total_regret_saved'] = float(improvement.sum())
            m['contribution_to_overall_mean_improvement'] = float(improvement.sum()/mask.sum())
        bins[label] = m
        lower = upper
    # Cost ties are assigned to the best (lowest numerical) policy rank.
    oracle_rank = (valid & ((costs - best[:, None]) <= TIE_TOL)).argmax(1) + 1
    return dict(overall=metrics(p, mask), policy_regret_bins=bins,
                oracle_rank={str(i): metrics(p, mask & (oracle_rank == i)) for i in range(1, 6)},
                high_regret=metrics(p, mask & (policy_r > .1)),
                per_instance={str(i): metrics(p, mask & (p['instance'] == i))
                              for i in np.unique(p['instance'])})


def paired_ci(p, reference, draws=2000):
    # Paired instance-cluster bootstrap, retaining all decisions per sampled instance.
    mask = p['nonterminal'] & (p['valid'].sum(1) > 1)
    ids = np.unique(p['instance'][mask])
    rr, pr = regrets(reference), regrets(p)
    rng = np.random.default_rng(20260918)
    deltas, tails = [], []
    for _ in range(draws):
        ix = np.concatenate([np.flatnonzero(mask & (p['instance'] == i))
                             for i in rng.choice(ids, len(ids), replace=True)])
        deltas.append(float((rr[ix] - pr[ix]).mean()))
        tails.append(float(np.quantile(rr[ix], .9) - np.quantile(pr[ix], .9)))
    return dict(mean_regret_reduction=float((rr[mask]-pr[mask]).mean()),
                mean_reduction_ci95=np.quantile(deltas, [.025, .975]).tolist(),
                p90_reduction=float(np.quantile(rr[mask], .9)-np.quantile(pr[mask], .9)),
                p90_reduction_ci95=np.quantile(tails, [.025, .975]).tolist(),
                bootstrap='2000 paired instance-cluster resamples; exploratory, no multiplicity correction')


def save_predictions(path, p):
    with open(str(path)+'.tmp', 'wb') as f:
        np.savez_compressed(f, **p)
    Path(str(path)+'.tmp').replace(path)


def report(out, predictions, filename):
    report_data = {name: diagnostics(p) for name, p in predictions.items()}
    for name, p in predictions.items():
        report_data[name]['vs_policy'] = paired_ci(p, predictions['policy'])
        if name != 'structural_original':
            report_data[name]['vs_structural_original'] = paired_ci(p, predictions['structural_original'])
        if name.startswith('strong') and 'structural_weighted' in predictions:
            report_data[name]['vs_structural_weighted'] = paired_ci(p, predictions['structural_weighted'])
    atomic_json(out/(filename+'.json'), report_data)
    lines = ['# Phase 2B H=8', '',
             'Exploratory reuse of the Phase 2 test set (10 instances); a fresh holdout is required for confirmation.',
             'All comparisons use the same nonterminal, multi-candidate decisions. Lower regret is better.', '',
             '| Model | N | Mean regret | P90 regret | Worst regret | Top-1 | Pairwise |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, d in report_data.items():
        m = d['overall']
        lines.append('| %s | %d | %.6f | %.6f | %.6f | %.4f | %.4f |' %
                     (name, m['groups'], m['mean_regret'], m['p90_regret'], m['worst_regret'],
                      m['top1_accuracy'], m['pairwise_accuracy']))
    for section in ('policy_regret_bins', 'oracle_rank'):
        lines += ['', '## '+section, '', '| Bin/rank | Model | N | Mean regret | P90 | Worst | Top-1 | Pairwise |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for key in report_data['policy'][section]:
            for name, d in report_data.items():
                m = d[section][key]
                if m['groups']:
                    lines.append('| %s | %s | %d | %.6f | %.6f | %.6f | %.4f | %s |' %
                                 (key, name, m['groups'], m['mean_regret'], m['p90_regret'],
                                  m['worst_regret'], m['top1_accuracy'], str(m['pairwise_accuracy'])))
    lines += ['', '## High policy regret (>0.1)', '', '| Model | N | Mean regret | P90 | Worst |',
              '|---|---:|---:|---:|---:|']
    for name, d in report_data.items():
        m = d['high_regret']
        lines.append('| %s | %d | %.6f | %.6f | %.6f |' %
                     (name, m['groups'], m['mean_regret'], m['p90_regret'], m['worst_regret']))
    lines += ['', '## Paired uncertainty', '', 'Positive reduction favors the model. Intervals cluster by instance.', '']
    for name, d in report_data.items():
        for ref in ('vs_policy', 'vs_structural_original', 'vs_structural_weighted'):
            if ref in d:
                m = d[ref]
                lines.append('- %s %s: mean reduction %.6f, 95%% CI %s; P90 reduction %.6f, 95%% CI %s.' %
                             (name, ref, m['mean_regret_reduction'], m['mean_reduction_ci95'],
                              m['p90_reduction'], m['p90_reduction_ci95']))
    lines += ['', 'Bins are [0,.01], (.01,.05], (.05,.10], (.10,.25], (.25,infinity).',
              'Oracle rank uses the lowest policy rank among cost-tied optima (tolerance 1e-6).',
              'High-regret bins use oracle labels for retrospective diagnosis only, never as model inputs.',
              'JSON includes per-instance metrics and each bin contribution to overall regret reduction.',
              'These are actual-future-state probes, not predicted future representations or closed-loop gains.']
    (out/(filename+'.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report_data


def train_variant(out, datasets, name, kind, weighted, epochs, batch_size, seed, manifest):
    folder = out/name
    folder.mkdir(exist_ok=True)
    config = dict(name=name, kind=kind, weighted=weighted, horizon=8, epochs=epochs,
                  batch_size=batch_size, seed=seed, splits=SPLITS, manifest=manifest,
                  lr=3e-4, patience=5, loss_normalization='valid pair count per decision',
                  selection='validation nonterminal mean regret', code_version=1)
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text()) != config:
        raise ValueError('Configuration mismatch: '+name)
    atomic_json(folder/'config.json', config)
    if (folder/'predictions.npz').exists():
        with np.load(folder/'predictions.npz') as f:
            return {k: f[k] for k in f.files}
    torch.manual_seed(seed)
    model = make_model(kind)
    feat = np.concatenate([a['structural'][:, :, 3][a['valid']] for a in datasets['train'].arrays])
    mean, std = torch.tensor(feat.mean(0)), torch.tensor(feat.std(0).clip(.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    start, best, stale, history = 0, float('inf'), 0, []
    if (folder/'last.pt').exists():
        c = torch.load(folder/'last.pt', map_location='cpu')
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        start, best, stale, history = c['epoch']+1, c['best'], c['stale'], c['history']
    for epoch in range(start, epochs):
        if stale >= 5:
            break
        torch.manual_seed(seed+epoch)
        model.train()
        started, total, count = time.time(), 0., 0
        loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed+epoch))
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = score_batch(model, b, kind, 'cpu', mean, std)
            loss = (weighted_loss if weighted else ranking_loss)(scores, b['costs'], b['valid'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        p = predict(model, datasets['validation'], kind, mean, std, batch_size)
        m = metrics(p, p['nonterminal'] & (p['valid'].sum(1)>1))
        if m['mean_regret'] < best:
            best, stale = m['mean_regret'], 0
            atomic_torch(folder/'best.pt', dict(model=model.state_dict(), mean=mean, std=std,
                                               epoch=epoch, config=config))
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, loss=total/count, validation=m, seconds=time.time()-started))
        atomic_json(folder/'history.json', history)
        atomic_torch(folder/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, best=best, stale=stale, history=history))
        atomic_json(out/'status.json', dict(stage='training', variant=name, epoch=epoch+1,
                    validation=m, updated_unix=time.time(), pid=os.getpid()))
        print(name, history[-1], flush=True)
    c = torch.load(folder/'best.pt', map_location='cpu')
    model.load_state_dict(c['model'])
    p = predict(model, datasets['test'], kind, c['mean'], c['std'], batch_size)
    atomic_json(folder/'results.json', dict(best_epoch=c['epoch']+1, test=diagnostics(p)))
    save_predictions(folder/'predictions.npz', p)
    return p


def run(out, bank, epochs=20, batch_size=16, seed=1234, diagnostics_only=False):
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(out/'pipeline.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out/'status.json', dict(stage='diagnostics', pid=os.getpid(), updated_unix=time.time()))
        datasets = {k: ProbeDataset(bank, v, 8) for k, v in SPLITS.items()}
        predictions = {}
        for kind in ('structural', 'full_state'):
            path = ROOT/'outputs/phase2'/('%s_h8' % kind)/'best.pt'
            c = torch.load(path, map_location='cpu')
            model = make_model(kind)
            model.load_state_dict(c['model'])
            p = predict(model, datasets['test'], kind, c['mean'], c['std'], batch_size)
            predictions[kind+'_original'] = p
            save_predictions(out/(kind+'_original.npz'), p)
            # Check reproduced metrics against the frozen original evaluation.
            old = json.loads((path.parent/'results.json').read_text())['test']['nonterminal']
            now = diagnostics(p)['overall']
            for key in ('mean_regret', 'top1_accuracy', 'pairwise_accuracy'):
                if abs(old[key]-now[key]) > 1e-6:
                    raise ValueError('Original metric reproduction failed: '+kind+' '+key)
        base = predictions['structural_original']
        predictions['policy'] = dict(base, scores=np.broadcast_to(-np.arange(5, dtype=float), base['scores'].shape))
        context = np.concatenate([a['context'][:, :, 3, 1] for a in datasets['test'].arrays])
        predictions['incurred_cost'] = dict(base, scores=-context)
        report(out, predictions, 'diagnostics')
        print('Original H8 diagnostics complete:', out/'diagnostics.md', flush=True)
        if diagnostics_only:
            atomic_json(out/'status.json', dict(stage='diagnostics_complete', updated_unix=time.time()))
            return
        manifest = json.loads((bank/'generation_manifest.json').read_text())
        atomic_json(out/'protocol.json', dict(splits=SPLITS, variants=VARIANTS, seed=seed,
                    epochs=epochs, batch_size=batch_size, horizon=8,
                    exploratory_test_reuse=True, selection='validation mean regret only',
                    primary='mean regret', secondary=['p90 regret', 'worst regret', 'high policy regret >0.1'],
                    interpretation='Compare weighted strong vs weighted structural; report paired instance CIs. '
                    'Confirm any positive finding on fresh held-out instances before a FAROS predictor.',
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), ROOT/'models/regret_state_encoder.py')}))
        for name, kind, weighted in VARIANTS:
            predictions[name] = train_variant(out, datasets, name, kind, weighted,
                                               epochs, batch_size, seed, manifest)
            report(out, predictions, 'comparison')
        atomic_json(out/'status.json', dict(stage='complete', updated_unix=time.time(),
                    comparison=str(out/'comparison.md')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--diagnostics-only', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'outputs/phase2b')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.launch:
        import shlex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--out', str(a.out),
                   '--threads', str(a.threads), '--epochs', str(a.epochs), '--batch-size', str(a.batch_size)]
        if a.diagnostics_only:
            command.append('--diagnostics-only')
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'faros_phase2b',
                        ' '.join(map(shlex.quote, command))+' >> '+shlex.quote(str(a.out/'pipeline.log'))+' 2>&1'], check=True)
        print('Launched faros_phase2b:', a.out/'pipeline.log')
        return
    torch.set_num_threads(a.threads)
    try:
        run(a.out, ROOT/'data/processed/probe_bank_v2', a.epochs, a.batch_size,
            diagnostics_only=a.diagnostics_only)
    except BlockingIOError:
        raise SystemExit('Another Phase 2B process already holds the lock')
    except Exception:
        atomic_json(a.out/'status.json', dict(stage='failed', error=traceback.format_exc(), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
