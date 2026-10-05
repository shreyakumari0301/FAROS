"""Compact, label-separated arrays for grouped ranking probes."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (1, 2, 4, 8)
FEATURES = ('remaining_customer_ratio', 'remaining_demand_ratio',
            'remaining_capacity_ratio', 'capacity_pressure', 'fragmentation',
            'depot_distance', 'routes_lower_bound')


def pack_groups(groups, path):
    first = groups[0]['state']
    size, k, nh, nodes = len(groups), 5, 4, len(first['coords'])
    a = dict(coords=np.asarray(first['coords'], np.float32),
             demands=np.asarray(first['demands'], np.float32),
             capacity=np.float32(first['capacity']),
             visited=np.ones((size, k, nh, nodes), bool),
             current=np.zeros((size, k, nh), np.int64),
             context=np.zeros((size, k, nh, 4), np.float32),
             structural=np.zeros((size, k, nh, 11), np.float32),
             costs=np.zeros((size, k), np.float32),
             valid=np.zeros((size, k), bool),
             steps=np.asarray([g['step'] for g in groups], np.int64))
    for gi, group in enumerate(groups):
        for ci, c in enumerate(group['candidates']):
            a['valid'][gi, ci] = True
            a['costs'][gi, ci] = c['completion_cost']
            base_prefix = c['post_action_state']['structural']['prefix_cost'] - c['immediate_cost']
            for hi, h in enumerate(HORIZONS):
                s, f = c['future_states'][str(h)], c['rollout_states'][str(h)]
                assert s['step'] == min(group['step'] + h, nodes - 1)
                assert sum(s['visited']) == s['step'] + 1
                assert 0 <= s['remaining_capacity'] <= first['capacity'] + 1e-5
                if h == 1:
                    assert s['visited'] == c['post_action_state']['visited']
                    assert s['current_node'] == c['post_action_state']['current_node']
                a['visited'][gi, ci, hi] = s['visited']
                a['current'][gi, ci, hi] = s['current_node']
                # Known incurred cost is necessary to compare total completion costs.
                # No final cost, regret, policy score, or rank enters the inputs.
                ctx = [s['remaining_capacity'] / first['capacity'],
                       s['prefix_cost'] - base_prefix, float(s['done']),
                       s['step'] / float(nodes - 1)]
                a['context'][gi, ci, hi] = ctx
                feat = [f[key] for key in FEATURES]
                feat[-1] /= 20.0
                a['structural'][gi, ci, hi] = feat + ctx
    with open(str(path) + '.tmp', 'wb') as out:
        np.savez_compressed(out, **a)
    Path(str(path) + '.tmp').replace(path)


class ProbeDataset(Dataset):
    def __init__(self, bank, instances, horizon):
        self.hi = HORIZONS.index(horizon)
        self.instances = list(instances)
        self.arrays, self.index = [], []
        for instance in self.instances:
            with np.load(Path(bank) / ('instance_%04d.npz' % instance)) as f:
                arr = {k: f[k] for k in f.files}
            self.arrays.append(arr)
            self.index.extend((len(self.arrays)-1, i) for i in range(len(arr['steps'])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ai, gi = self.index[idx]
        a, hi = self.arrays[ai], self.hi
        k, n = a['visited'].shape[1], len(a['coords'])
        coords = np.broadcast_to(a['coords'], (k, n, 2))
        visited = a['visited'][gi, :, hi].astype(np.float32)
        current = a['current'][gi, :, hi]
        rel = coords - a['coords'][current, None, :]
        demand = np.broadcast_to(a['demands'] / a['capacity'], (k, n))
        depot = np.zeros((k, n), np.float32)
        depot[:, 0] = 1
        is_current = np.zeros((k, n), np.float32)
        is_current[np.arange(k), current] = 1
        nodes = np.concatenate((coords, demand[..., None], visited[..., None],
                                depot[..., None], is_current[..., None], rel), -1).copy()
        return dict(nodes=torch.from_numpy(nodes),
                    context=torch.from_numpy(a['context'][gi, :, hi].copy()),
                    structural=torch.from_numpy(a['structural'][gi, :, hi].copy()),
                    costs=torch.from_numpy(a['costs'][gi].copy()),
                    valid=torch.from_numpy(a['valid'][gi].copy()),
                    nonterminal=torch.tensor(a['steps'][gi] + max(HORIZONS) < n - 1),
                    instance=torch.tensor(self.instances[ai]))
