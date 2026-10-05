"""Actual future-state probes: instance-disjoint ranking, no FAROS prediction."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.probe_data import ProbeDataset, HORIZONS
from models.future_state_encoder import FutureStateEncoder
from models.structural_probe import StructuralProbe
from scripts.generate_probe_bank import atomic_json

TIE_TOL = 1e-6


def ranking_loss(scores, costs, valid):
    # Each decision has equal weight, independent of its number of valid pairs.
    better = (costs[:, :, None] + TIE_TOL < costs[:, None, :])
    pairs = better & valid[:, :, None] & valid[:, None, :]
    losses = F.softplus(-(scores[:, :, None] - scores[:, None, :]))
    counts = pairs.sum((1, 2))
    active = counts > 0
    if not active.any():
        return scores.sum() * 0
    return ((losses * pairs).sum((1, 2)) / counts.clamp_min(1))[active].mean()


def score_batch(model, batch, kind, device, mean=None, std=None):
    b, k = batch['valid'].shape
    if kind == 'structural':
        x = batch['structural'].to(device).reshape(b*k, -1)
        return model((x-mean)/std).reshape(b, k)
    nodes = batch['nodes'].to(device).reshape(b*k, 201, 8)
    ctx = batch['context'].to(device).reshape(b*k, 4)
    return model(nodes, ctx).reshape(b, k)


def metric_summary(scores, costs, valid, select):
    scores, costs, valid = scores[select], costs[select], valid[select]
    if len(scores) == 0:
        return dict(groups=0)
    pred = scores.copy()
    pred[~valid] = -np.inf
    chosen = pred.argmax(1)
    best = np.where(valid, costs, np.inf).min(1)
    regret = costs[np.arange(len(costs)), chosen] - best
    better = (costs[:, :, None] + TIE_TOL < costs[:, None, :])
    pairs = better & valid[:, :, None] & valid[:, None, :]
    diff = scores[:, :, None] - scores[:, None, :]
    correct = (diff > 1e-8).astype(float) + .5 * (np.abs(diff) <= 1e-8)
    counts = pairs.sum((1, 2))
    active = counts > 0
    return dict(groups=len(costs), pairs=int(pairs.sum()),
                pairwise_accuracy=float((correct*pairs).sum()/pairs.sum()) if pairs.any() else None,
                pairwise_group_accuracy=float(((correct*pairs).sum((1, 2))[active]/counts[active]).mean()) if active.any() else None,
                top1_accuracy=float((regret <= TIE_TOL).mean()),
                mean_regret=float(regret.mean()),
                mean_selected_cost=float(costs[np.arange(len(costs)), chosen].mean()),
                mean_improvement_vs_greedy=float((costs[:, 0]-costs[np.arange(len(costs)), chosen]).mean()))


def summarize(scores, costs, valid, nonterminal, instances):
    informative = valid.sum(1) > 1
    result = dict(all=metric_summary(scores, costs, valid, np.ones(len(costs), bool)),
                  multi_candidate=metric_summary(scores, costs, valid, informative),
                  nonterminal=metric_summary(scores, costs, valid, nonterminal & informative))
    result['per_instance'] = {str(i): metric_summary(scores, costs, valid, (instances == i) & nonterminal & informative)
                              for i in np.unique(instances)}
    return result


@torch.no_grad()
def evaluate(model, loader, kind, device, mean=None, std=None, baseline=None):
    model.eval()
    stores = [[] for _ in range(5)]
    for batch in loader:
        if baseline == 'policy':
            scores = -torch.arange(5).float().expand(len(batch['valid']), -1)
        elif baseline == 'incurred_cost':
            scores = -batch['context'][:, :, 1]
        else:
            scores = score_batch(model, batch, kind, device, mean, std).cpu()
        for store, val in zip(stores, (scores, batch['costs'], batch['valid'], batch['nonterminal'], batch['instance'])):
            store.append(val.numpy())
    # Costs and masks are retained separately from model inputs.
    return summarize(*(np.concatenate(s) for s in stores))


def atomic_torch(path, obj):
    temp = str(path) + '.tmp'
    torch.save(obj, temp)
    Path(temp).replace(path)


def train_one(bank, out, splits, kind, horizon, epochs=20, batch_size=16, seed=1234, device='cpu'):
    out = Path(out) / ('%s_h%d' % (kind, horizon))
    out.mkdir(parents=True, exist_ok=True)
    config = dict(kind=kind, horizon=horizon, epochs=epochs, batch_size=batch_size, seed=seed,
                  learning_rate=3e-4, patience=5, split=splits,
                  manifest=json.loads((Path(bank)/'generation_manifest.json').read_text()))
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Training configuration changed; choose a new output directory')
    atomic_json(out/'config.json', config)
    if (out/'results.json').exists():
        return json.loads((out/'results.json').read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train = ProbeDataset(bank, splits['train'], horizon)
    val = ProbeDataset(bank, splits['validation'], horizon)
    test = ProbeDataset(bank, splits['test'], horizon)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    model = (StructuralProbe() if kind == 'structural' else FutureStateEncoder()).to(device)
    # Standardization statistics are computed exclusively from training instances.
    features = np.concatenate([a['structural'][:, :, HORIZONS.index(horizon)][a['valid']] for a in train.arrays])
    mean = torch.tensor(features.mean(0), device=device)
    std = torch.tensor(features.std(0).clip(.01), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    start, best, stale, history = 0, float('inf'), 0, []
    if (out/'last.pt').exists():
        ckpt = torch.load(out/'last.pt', map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start, best, stale, history = ckpt['epoch']+1, ckpt['best'], ckpt['stale'], ckpt['history']
    for epoch in range(start, epochs):
        if stale >= 5:
            break
        torch.manual_seed(seed + epoch)
        loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed+epoch))
        model.train()
        total, batches = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = score_batch(model, batch, kind, device, mean, std)
            loss = ranking_loss(scores, batch['costs'].to(device), batch['valid'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ranking loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        metrics = evaluate(model, val_loader, kind, device, mean, std)
        value = metrics['nonterminal']['mean_regret']
        improved = value < best
        if improved:
            best, stale = value, 0
            atomic_torch(out/'best.pt', dict(model=model.state_dict(), mean=mean, std=std,
                                           config=config, epoch=epoch))
        else:
            stale += 1
        row = dict(epoch=epoch+1, train_loss=total/max(batches, 1), validation=metrics)
        history.append(row)
        atomic_json(out/'history.json', history)
        atomic_torch(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, best=best, stale=stale, history=history))
        print('%s H=%d epoch=%d loss=%.5f val_regret=%.6f' %
              (kind, horizon, epoch+1, row['train_loss'], value), flush=True)
    ckpt = torch.load(out/'best.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    results = dict(kind=kind, horizon=horizon, best_epoch=ckpt['epoch']+1,
                   validation=evaluate(model, val_loader, kind, device, mean, std),
                   test=evaluate(model, test_loader, kind, device, mean, std),
                   policy_baseline=evaluate(model, test_loader, kind, device, baseline='policy'),
                   incurred_cost_baseline=evaluate(model, test_loader, kind, device, baseline='incurred_cost'))
    atomic_json(out/'results.json', results)
    return results


def run_all(bank, out, epochs=20, batch_size=16, seed=1234, splits=None):
    splits = splits or dict(train=list(range(80)), validation=list(range(80, 90)), test=list(range(90, 100)))
    sets = [set(v) for v in splits.values()]
    assert all(sets) and all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'splits.json', splits)
    results = []
    for kind in ('structural', 'full_state'):
        for horizon in HORIZONS:
            results.append(train_one(bank, out, splits, kind, horizon, epochs, batch_size, seed))
            atomic_json(out/'comparison.json', results)
    rows = ['# Phase 2 actual-state probes', '',
            'Held-out synthetic instances. Main table excludes states that can terminate within H=8.', '',
            '| Probe | H | Pairwise accuracy | Top-1 accuracy | Mean regret |',
            '|---|---:|---:|---:|---:|']
    for r in results:
        for name, key in ((r['kind'], 'test'), ('policy', 'policy_baseline'), ('incurred_cost', 'incurred_cost_baseline')):
            m = r[key]['nonterminal']
            rows.append('| %s | %d | %.4f | %.4f | %.6f |' %
                        (name, r['horizon'], m['pairwise_accuracy'], m['top1_accuracy'], m['mean_regret']))
    (out/'comparison.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', default=str(ROOT/'data/processed/probe_bank_v2'))
    p.add_argument('--out', default=str(ROOT/'outputs/phase2'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run_all(a.bank, a.out, a.epochs, a.batch_size)
