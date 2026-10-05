"""Closed-loop greedy, FAROS and explicit full greedy-completion rollout."""
import time
import numpy as np
import torch
from models.faros_outcome import selection_score
from planner.rollout import snapshot_env, restore_env, prefix_cost_now, greedy_complete
from policy.mnlp_adapter import topk_feasible_mnlp


def restore(policy, snapshot, step, done=False):
    restore_env(policy.env, snapshot)
    policy._step_state = policy.env.step_state
    policy._current_step, policy._done = step, done


def exact_inputs(policy, candidates):
    """Same eight node features/four context features as ProbeDataset(H=1)."""
    snapshot, step = snapshot_env(policy.env), policy._current_step
    before = prefix_cost_now(policy)
    nodes, contexts = [], []
    for candidate in candidates:
        restore(policy, snapshot, step)
        state = policy.step_joint(candidate['action'])
        xy = state['coords'].astype(np.float32)
        n = len(xy)
        depot, current = np.zeros(n, np.float32), np.zeros(n, np.float32)
        depot[0], current[state['current_node']] = 1., 1.
        demand = state['demands'].astype(np.float32)/state['capacity']
        nodes.append(np.column_stack((xy, demand, state['visited'].astype(np.float32),
                     depot, current, xy-xy[state['current_node']])).astype(np.float32))
        contexts.append([state['remaining_capacity']/state['capacity'],
                         prefix_cost_now(policy)-before, float(state['done']), state['step']/200.])
    restore(policy, snapshot, step)
    return (torch.tensor(np.stack(nodes))[None], torch.tensor(contexts, dtype=torch.float32)[None],
            torch.ones((1, len(candidates)), dtype=torch.bool))


def verify_route(state, reported_cost):
    nodes, flags = state['selected_nodes'][0], state['selected_flags'][0]
    n = state['n_customers']
    if len(nodes) != n or sorted(nodes) != list(range(1, n+1)) or not state['done']:
        raise ValueError('Route does not visit each customer exactly once')
    xy, demands, capacity = state['coords'], state['demands'], state['capacity']
    prev, load, cost = 0, 0., 0.
    for node, flag in zip(nodes, flags):
        if flag:
            cost += np.linalg.norm(xy[prev]-xy[0])
            prev, load = 0, 0.
        load += demands[node]
        if load > capacity+1e-5:
            raise ValueError('Capacity violation')
        cost += np.linalg.norm(xy[prev]-xy[node])
        prev = node
    cost += np.linalg.norm(xy[prev]-xy[0])
    if abs(cost-reported_cost) > 1e-4:
        raise ValueError('Independent route cost mismatch')
    return float(cost)


@torch.no_grad()
def route(policy, instance, method, model=None, beta=0.):
    calls, inference_calls, inference_seconds = 0, 0, 0.
    original = policy.logits
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    policy.logits = counted
    started = time.perf_counter()
    try:
        policy.reset_episode(instance)
        while not policy._done:
            if method == 'greedy':
                action = policy.greedy_action()
            else:
                candidates = topk_feasible_mnlp(policy.logits(), policy, k=5)
                if not candidates:
                    raise ValueError('No feasible candidate')
                if method == 'faros':
                    nodes, context, valid = exact_inputs(policy, candidates)
                    tick = time.perf_counter()
                    r, logits = model(nodes, context, valid)
                    scores = selection_score(r, logits, valid, beta)
                    inference_seconds += time.perf_counter()-tick
                    inference_calls += 1
                    chosen = int(scores.argmin(1).item())
                elif method == 'explicit_rollout':
                    snapshot, step = snapshot_env(policy.env), policy._current_step
                    costs = []
                    for candidate in candidates:
                        restore(policy, snapshot, step)
                        policy.step_joint(candidate['action'])
                        costs.append(greedy_complete(policy))
                    restore(policy, snapshot, step)
                    chosen = int(np.argmin(costs))
                else:
                    raise ValueError(method)
                action = candidates[chosen]['action']
            policy.step_joint(action)
        cost, state = policy.tour_cost(), policy.snapshot()
        seconds = time.perf_counter()-started
        independent = verify_route(state, cost)
        return dict(instance=instance, method=method, route_cost=cost, seconds=seconds,
                    policy_forward_passes=calls, faros_forward_passes=inference_calls,
                    faros_inference_seconds=inference_seconds, independently_computed_cost=independent,
                    nodes=state['selected_nodes'][0], depot_flags=state['selected_flags'][0],
                    reference_gap=None)
    finally:
        policy.logits = original
