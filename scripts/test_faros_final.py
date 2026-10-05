"""Set invariants, H1 parity, resumable training and real-policy routing smoke tests."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.faros_outcome import FAROSOutcome, selection_score, outcome_loss
from scripts.run_faros_final import SETTINGS, train, predict, scored, freeze
from scripts.generate_probe_bank import atomic_json, install_synthetic
from scripts.generate_branch_bank import load_config
from scripts.run_phase2c import initializer
from scripts.faros_routing import route, exact_inputs
from policy.backbone import load_policy
from policy.mnlp_adapter import topk_feasible_mnlp
from data.probe_data import ProbeDataset


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    bank = ROOT/'data/processed/probe_bank_v2'
    item = ProbeDataset(bank, [0], 1)[0]
    nodes, ctx, valid = item['nodes'][None], item['context'][None], item['valid'][None]
    model = FAROSOutcome().eval()
    perm = torch.tensor([4, 2, 0, 3, 1])
    with torch.no_grad():
        r, risk = model(nodes, ctx, valid)
        rp, qp = model(nodes[:, perm], ctx[:, perm], valid[:, perm])
        assert torch.allclose(r[:, perm], rp, atol=2e-6)
        assert torch.allclose(risk[:, perm], qp, atol=2e-6)
        original = selection_score(r, risk, valid, .1).argmin(1)
        reordered = perm[selection_score(rp, qp, valid[:, perm], .1).argmin(1)]
        assert torch.equal(original, reordered)
        padded_nodes = torch.cat((nodes, torch.full_like(nodes[:, :1], float('nan'))), 1)
        padded_ctx = torch.cat((ctx, torch.full_like(ctx[:, :1], float('nan'))), 1)
        padded_valid = torch.cat((valid, torch.zeros((1, 1), dtype=torch.bool)), 1)
        rpad, qpad = model(padded_nodes, padded_ctx, padded_valid)
        assert torch.allclose(r, rpad[:, :5], atol=2e-6)
        assert torch.allclose(risk, qpad[:, :5], atol=2e-6)
        assert selection_score(rpad, qpad, padded_valid, .1)[0, -1].isinf()
        costs = item['costs'][None]
        l = outcome_loss(r, risk, costs, valid, SETTINGS)
        lp = outcome_loss(rpad, qpad, torch.cat((costs, torch.tensor([[float('nan')]])), 1), padded_valid, SETTINGS)
        assert torch.allclose(l, lp, atol=2e-6)
        assert torch.allclose(l, outcome_loss(r, risk, costs+100, valid, SETTINGS), atol=2e-5)
    model.train()
    r, risk = model(padded_nodes, padded_ctx, padded_valid)
    loss = outcome_loss(r, risk, torch.cat((costs, torch.tensor([[float('nan')]])), 1), padded_valid, SETTINGS)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    try:
        model(nodes, ctx, torch.zeros_like(valid))
        raise AssertionError('Empty candidate set accepted')
    except ValueError:
        pass
    print('PASS: candidate equivariance, selection, NaN padding isolation, masked loss and gradients, relative cost targets', flush=True)

    with tempfile.TemporaryDirectory(prefix='faros_final_test_') as temp:
        out = Path(temp)
        datasets = {name: Subset(ProbeDataset(bank, [i], 1), range(4))
                    for name, i in [('train', 0), ('validation', 80)]}
        cfg = dict(SETTINGS, epochs=1, batch_size=2)
        c = train(out, datasets, cfg)
        stamp = (out/'last.pt').stat().st_mtime_ns
        resumed = train(out, datasets, cfg)
        assert (out/'last.pt').stat().st_mtime_ns == stamp
        assert resumed['beta'] == c['beta']
        assert all(torch.equal(c['model'][k], resumed['model'][k]) for k in c['model'])
        atomic_json(out/'protocol.json', {'smoke': True})
        f = freeze(out, c)
        assert freeze(out, None) == f
        model.load_state_dict(c['model'])
        model.eval()
        assert np.isfinite(scored(predict(model, datasets['validation'], 2), c['beta'])['scores']).all()
        print('PASS: training, validation-only beta/checkpoint, deterministic resume, freeze and inference', flush=True)
        initializer(out/'initializer.txt')
        cfg = load_config()
        policy = load_policy(name='mnlp', checkpoint=cfg['policy']['checkpoint'],
                             mnlp_root=cfg['policy']['mnlp_root'], data_path=str(out/'initializer.txt'),
                             device='cpu', max_episodes=1)
        install_synthetic(policy, 100, 20260916)
        policy.reset_episode(0)
        candidates = topk_feasible_mnlp(policy.logits(), policy, 5)
        xn, xc, xv = exact_inputs(policy, candidates)
        assert torch.allclose(xn, nodes[:, :len(candidates)], atol=1e-6)
        assert torch.allclose(xc, ctx[:, :len(candidates)], atol=1e-6)
        assert policy._current_step == 0 and not policy._done
        print('PASS: deployment H1 inputs match stored training features exactly', flush=True)
        # Tiny synthetic routing smoke; no final evaluation seed is used.
        install_synthetic(policy, 1, 98761923, customers=8)
        results = {method: route(policy, 0, method, model, c['beta'])
                   for method in ('greedy', 'faros', 'explicit_rollout')}
        assert results['greedy']['policy_forward_passes'] == 8
        assert results['faros']['policy_forward_passes'] == 8
        assert results['faros']['faros_forward_passes'] == 8
        assert results['explicit_rollout']['policy_forward_passes'] > 8
        assert results['explicit_rollout']['route_cost'] <= results['greedy']['route_cost']+1e-4
        assert all(r['seconds'] >= r['faros_inference_seconds'] for r in results.values())
        print('PASS: three closed-loop methods, route feasibility/cost verification, compute accounting', flush=True)
    print('ALL FINAL FAROS TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
