"""Scientific invariants plus real-policy end-to-end smoke test."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.future_state_encoder import FutureStateEncoder
from scripts.generate_probe_bank import generate, install_synthetic
from scripts.generate_branch_bank import load_config, generate_episode
from scripts.train_ranking_probes import ranking_loss, metric_summary, run_all
from policy.backbone import load_policy
from planner.rollout import snapshot_env, restore_env, prefix_cost_now


def main():
    torch.set_num_threads(2)
    costs = torch.tensor([[1., 2., 3., 0., 0.]])
    valid = torch.tensor([[True, True, True, False, False]])
    good = torch.tensor([[3., 2., 1., 0., 0.]], requires_grad=True)
    assert ranking_loss(good, costs, valid) < ranking_loss(-good, costs, valid)
    loss = ranking_loss(good, costs, valid)
    loss.backward()
    assert good.grad[0, 0] < 0 and good.grad[0, 2] > 0
    equal = torch.ones_like(costs)
    assert ranking_loss(good, equal, valid) == 0
    m = metric_summary(good.detach().numpy(), costs.numpy(), valid.numpy(), np.array([True]))
    assert m['pairwise_accuracy'] == 1 and m['top1_accuracy'] == 1 and m['mean_regret'] == 0
    m = metric_summary(np.zeros((1, 5)), costs.numpy(), valid.numpy(), np.array([True]))
    assert m['pairwise_accuracy'] == .5
    model = FutureStateEncoder().eval()
    nodes, context = torch.rand(2, 201, 8), torch.rand(2, 4)
    nodes[:, :, 3] = 0
    perm = torch.cat((torch.tensor([0]), torch.randperm(200)+1))
    with torch.no_grad():
        assert torch.allclose(model(nodes, context), model(nodes[:, perm], context), atol=1e-5)
    print('PASS: pairwise direction, masks, ties, regret, node permutation invariance', flush=True)

    cfg = load_config()
    policy = load_policy(name='mnlp', checkpoint=cfg['policy']['checkpoint'],
                         mnlp_root=cfg['policy']['mnlp_root'], data_path=cfg['policy']['data_path'],
                         device='cpu', max_episodes=1)
    install_synthetic(policy, 1, 98765)
    groups = generate_episode(policy, 0, 5, [1, 2, 4, 8], step_stride=199, full_states=True)
    # Independent replay checks the recorded full futures, including terminal clamping.
    for g in groups:
        for c in g.candidates:
            policy.reset_episode(0)
            for _ in range(g.step):
                policy.step_joint(policy.greedy_action())
            policy.step_joint(c.action)
            for h in range(1, 9):
                if h > 1 and not policy._done:
                    policy.step_joint(policy.greedy_action())
                if str(h) in c.future_states:
                    saved, live = c.future_states[str(h)], policy.snapshot()
                    assert saved['visited'] == live['visited'].astype(int).tolist()
                    assert saved['current_node'] == live['current_node']
                    assert abs(saved['remaining_capacity']-live['remaining_capacity']) < 1e-6
                    assert abs(saved['prefix_cost']-prefix_cost_now(policy)) < 1e-6
            while not policy._done:
                policy.step_joint(policy.greedy_action())
            assert abs(policy.tour_cost()-c.completion_cost) < 1e-6
    print('PASS: full future-state replay, terminal clamping, completion label replay', flush=True)
    bank = ROOT/'data/processed/phase2_smoke'
    out = ROOT/'outputs/phase2_smoke'
    generate(bank, count=3, seed=98766, stride=100)
    before = {p.name: p.stat().st_mtime_ns for p in bank.glob('instance_*.npz')}
    generate(bank, count=3, seed=98766, stride=100)
    assert before == {p.name: p.stat().st_mtime_ns for p in bank.glob('instance_*.npz')}
    try:
        generate(bank, count=3, seed=98767, stride=100)
        raise AssertionError('Manifest mismatch was accepted')
    except ValueError as e:
        assert 'Manifest mismatch' in str(e)
    splits = dict(train=[0], validation=[1], test=[2])
    run_all(bank, out, epochs=1, batch_size=2, splits=splits)
    run_all(bank, out, epochs=1, batch_size=2, splits=splits)
    results = json.loads((out/'comparison.json').read_text())
    assert len(results) == 8
    assert all(np.isfinite(r['test']['all']['mean_regret']) for r in results)
    print('PASS: atomic sharding, resume, manifest rejection, eight probe training/evaluation runs', flush=True)
    print('ALL PHASE 2 CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
