"""Generate independent synthetic CVRP200 data with atomic per-instance shards."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_branch_bank import generate_episode, load_config
from policy.backbone import load_policy
from data.probe_data import pack_groups


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def install_synthetic(policy, count, seed, customers=200):
    """Use independent uniform instances, never benchmark coordinates/solutions.

    Dummy feasible reference routes only satisfy VRPEnv's interface; they never
    supply branch labels. Labels always come from frozen-policy completions.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, (count, customers + 1, 2)).astype(np.float32)
    demands = rng.integers(1, 10, (count, customers + 1)).astype(np.float32)
    demands[:, 0] = 0
    env = policy.env
    env.raw_data_nodes = torch.tensor(coords, device=policy.device)
    env.raw_data_demand = torch.tensor(demands, device=policy.device)
    env.raw_data_capacity = torch.full((count,), 50.0, device=policy.device)
    solution = torch.zeros((count, customers, 2), dtype=torch.long, device=policy.device)
    solution[:, :, 0] = torch.arange(1, customers + 1, device=policy.device)
    solution[:, :, 1] = 1
    env.raw_data_node_flag = solution
    env.raw_data_cost = torch.zeros(count, device=policy.device)
    return coords, demands


def generate(bank, count=100, seed=20260916, stride=1, initializer_data_path=None):
    bank = Path(bank)
    bank.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    checkpoint = Path(cfg['policy']['checkpoint'])
    manifest = dict(schema=2, n_instances=count, seed=seed, customers=200,
                    capacity=50, coords='independent uniform [0,1)^2', demands='integer 1..9',
                    top_k=5, horizons=[1, 2, 4, 8], step_stride=stride,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source='synthetic; benchmark excluded')
    if count < 1 or stride < 1:
        raise ValueError('Instance count and stride must be positive')
    mf = bank / 'generation_manifest.json'
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise ValueError('Manifest mismatch: use a new bank directory')
    atomic_json(mf, manifest)
    torch.manual_seed(seed)
    policy = load_policy(name='mnlp', checkpoint=str(checkpoint),
                         mnlp_root=cfg['policy']['mnlp_root'],
                         data_path=str(initializer_data_path) if initializer_data_path else cfg['policy']['data_path'],
                         device='cpu', max_episodes=1)
    coords, demands = install_synthetic(policy, count, seed)
    np.savez_compressed(bank / 'instances.npz', coords=coords, demands=demands, capacity=50)
    completed = []
    for ep in range(count):
        shard = bank / ('instance_%04d.npz' % ep)
        if shard.exists():
            # Read the entire archive: truncated/invalid files must not be skipped.
            with np.load(shard) as f:
                values = {k: f[k] for k in f.files}
            if (len(values['steps']) != len(range(0, 200, stride))
                    or not np.array_equal(values['coords'], coords[ep])
                    or not np.array_equal(values['demands'], demands[ep])):
                raise ValueError('Incomplete shard: ' + str(shard))
            completed.append(ep)
            print('resume: verified instance', ep, flush=True)
            continue
        started = time.time()
        atomic_json(bank / 'progress.json', dict(stage='generation', active_instance=ep,
                    completed=completed, target=count, updated_unix=time.time()))
        print('Generating instance %d/%d' % (ep + 1, count), flush=True)
        groups = generate_episode(policy, ep, 5, [1, 2, 4, 8], stride, full_states=True)
        packed = [g.to_dict() for g in groups]
        if len(groups) != len(range(0, 200, stride)):
            raise ValueError('Trajectory stopped early')
        pack_groups(packed, shard)
        completed.append(ep)
        atomic_json(bank / 'progress.json', dict(stage='generation', completed=completed,
                    target=count, last_instance_seconds=time.time()-started, updated_unix=time.time()))
        print('Saved %s: %d groups, %.1fs' % (shard.name, len(groups), time.time()-started), flush=True)
    atomic_json(bank / 'progress.json', dict(stage='complete', completed=completed,
                target=count, updated_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', default=str(ROOT / 'data/processed/probe_bank_v2'))
    p.add_argument('--n-instances', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260916)
    p.add_argument('--step-stride', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    generate(a.bank, a.n_instances, a.seed, a.step_stride)
