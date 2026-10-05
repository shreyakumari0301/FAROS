"""Scientific invariants and resumable real-bank smoke training for Phase 2B."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_phase2b as b


def main():
    torch.set_num_threads(2)
    scores = torch.tensor([[0., 0., 0.]], requires_grad=True)
    costs = torch.tensor([[0., .1, 2.]])
    valid = torch.tensor([[True, True, False]])
    loss = b.weighted_loss(scores, costs, valid)
    assert torch.allclose(loss, torch.tensor(.1*np.log(2), dtype=loss.dtype), atol=1e-7)
    assert torch.allclose(b.weighted_loss(scores, costs*10, valid), loss*10)
    loss.backward()
    assert scores.grad[0, 0] < 0 and scores.grad[0, 1] > 0 and scores.grad[0, 2] == 0
    assert b.weighted_loss(scores, torch.ones_like(costs), valid) == 0
    assert b.weighted_loss(scores, costs, torch.zeros_like(valid)) == 0
    print('PASS: cost weighting, direction, invalid candidates, ties', flush=True)

    ds = b.ProbeDataset(ROOT/'data/processed/probe_bank_v2', [90], 8)
    item = ds[0]
    model = b.RegretStateEncoder().eval()
    nodes, context = item['nodes'], item['context']
    perm = torch.randperm(201)  # Includes the depot: readout uses flags, not array position.
    with torch.no_grad():
        assert torch.allclose(model(nodes, context), model(nodes[:, perm], context), atol=2e-6)
        terminal = nodes.clone()
        terminal[:, :, 3] = 1
        assert torch.isfinite(model(terminal, context)).all()
    model.train()
    model(nodes, context).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('PASS: node permutation invariance, terminal attention, finite gradients', flush=True)

    p = dict(scores=np.array([[0., 1., -5.], [2., 1., -5.]]),
             costs=np.array([[.3, 0., 9.], [0., 0., 9.]]),
             valid=np.array([[True, True, False]]*2), nonterminal=np.ones(2, bool),
             instance=np.array([90, 91]))
    d = b.diagnostics(p)
    assert d['overall']['mean_regret'] == 0
    assert d['oracle_rank']['1']['groups'] == 1 and d['oracle_rank']['2']['groups'] == 1
    assert sum(m['groups'] for m in d['policy_regret_bins'].values()) == 2
    assert d['high_regret']['groups'] == 1
    # Float32 addition of 1e-6 near route cost 30 rounds to a larger tolerance.
    near = dict(p, costs=np.array([[30.+2e-6, 30., 99.], [30., 30., 99.]], np.float32))
    assert b.diagnostics(near)['oracle_rank']['2']['groups'] == 1
    ci = b.paired_ci(p, p, draws=20)
    assert ci['mean_reduction_ci95'] == [0., 0.]
    print('PASS: bins, tied oracle ranks, tail metrics, paired bootstrap', flush=True)

    datasets = {}
    for key, instance in [('train', 0), ('validation', 80), ('test', 90)]:
        source = b.ProbeDataset(ROOT/'data/processed/probe_bank_v2', [instance], 8)
        small = Subset(source, range(4))
        small.arrays = source.arrays
        datasets[key] = small
    with tempfile.TemporaryDirectory(prefix='phase2b_test_') as tmp:
        out = Path(tmp)
        for name, kind, weighted in b.VARIANTS:
            pred = b.train_variant(out, datasets, name, kind, weighted, 1, 2, 1234, {'smoke': True})
            assert np.isfinite(pred['scores']).all()
            checkpoint = out/name/'last.pt'
            stamp = checkpoint.stat().st_mtime_ns
            # Remove only disposable smoke outputs to test epoch-boundary resume.
            (out/name/'predictions.npz').unlink()
            (out/name/'results.json').unlink()
            again = b.train_variant(out, datasets, name, kind, weighted, 1, 2, 1234, {'smoke': True})
            assert checkpoint.stat().st_mtime_ns == stamp
            assert np.array_equal(pred['scores'], again['scores'])
    print('PASS: all four variants train, evaluate, and resume identically', flush=True)
    print('ALL PHASE 2B CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
