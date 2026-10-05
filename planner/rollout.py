"""Greedy / horizon rollouts for Phase 1 branch generation (MnLP-backed)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from data.structural_labels import structural_labels
from policy.mnlp_adapter import MnLPPolicy


def _visited_from_selected(selected_nodes: torch.Tensor, n_customers: int) -> np.ndarray:
    visited = np.zeros(n_customers + 1, dtype=bool)
    visited[0] = True
    if selected_nodes.numel() > 0:
        for node in selected_nodes.view(-1).tolist():
            if int(node) > 0:
                visited[int(node)] = True
    return visited


def labels_from_env(policy: MnLPPolicy, prefix_cost: float) -> Dict[str, float]:
    problems = policy.env.problems.detach().cpu().numpy()[0]
    visited = _visited_from_selected(policy.env.selected_student_list[0], policy.n_customers)
    return structural_labels(
        coords=problems[:, :2],
        demands=problems[:, 2],
        visited=visited,
        remaining_capacity=float(problems[0, 3]),
        capacity=float(policy.env.raw_data_capacity.ravel()[0].item()),
        prefix_cost=prefix_cost,
    )


def prefix_cost_now(policy: MnLPPolicy) -> float:
    from cvrp_cost import prefix_cost_from_lists

    return float(
        prefix_cost_from_lists(
            policy.env.problems,
            policy.env.selected_student_list,
            policy.env.selected_student_flag,
        )
    )


def greedy_complete(policy: MnLPPolicy) -> float:
    """Continue greedy decoding from the live env until done; return tour cost."""
    while not policy._done:
        action = policy.greedy_action()
        policy.step_joint(action)
    return policy.tour_cost()


def rollout_to_horizons(
    policy: MnLPPolicy,
    horizons: Sequence[int],
    full_states=None,
) -> Dict[int, Dict[str, float]]:
    """Label futures at absolute horizons H from the decision.

    Caller must already be at the exact post-action state s_{t+1} = T(s_t, a).

    Convention:
      H=1 -> current post-action state (0 extra greedy steps)
      H=h -> take (h-1) additional greedy steps, then label
    """
    targets: Dict[int, Dict[str, float]] = {}
    def capture():
        labels = labels_from_env(policy, prefix_cost_now(policy))
        if full_states is not None:
            s = policy.snapshot()
            full_states.update({"latest": {
                "visited": s["visited"].astype(int).tolist(),
                "current_node": s["current_node"],
                "remaining_capacity": s["remaining_capacity"],
                "done": s["done"], "step": s["step"],
                "prefix_cost": labels["prefix_cost"],
            }})
        return labels

    def save(h, labels):
        targets[h] = labels
        if full_states is not None:
            full_states[str(h)] = dict(full_states["latest"])
    horizon_set = sorted(set(int(h) for h in horizons))
    if not horizon_set:
        return targets

    max_h = max(horizon_set)
    # h_extra = number of greedy steps beyond the already-applied action
    # After k extra steps we are at absolute horizon H = k+1
    if 1 in horizon_set:
        save(1, capture())

    for k in range(1, max_h):  # k extra steps -> absolute H = k+1
        abs_h = k + 1
        if policy._done:
            lab = capture()
            for hh in horizon_set:
                if hh >= abs_h and hh not in targets:
                    save(hh, lab)
            break

        action = policy.greedy_action()
        policy.step_joint(action)
        if abs_h in horizon_set:
            save(abs_h, capture())

    for hh in horizon_set:
        if hh not in targets:
            save(hh, capture())
    if full_states is not None:
        full_states.pop("latest", None)
    return targets


def snapshot_env(env) -> dict:
    return {
        "problems": env.problems.clone(),
        "selected_count": env.selected_count,
        "selected_node_list": env.selected_node_list.clone(),
        "selected_teacher_flag": env.selected_teacher_flag.clone(),
        "selected_student_list": env.selected_student_list.clone(),
        "selected_student_flag": env.selected_student_flag.clone(),
    }


def restore_env(env, snap: dict) -> None:
    from VRPEnv import Step_State

    env.problems = snap["problems"].clone()
    env.selected_count = snap["selected_count"]
    env.selected_node_list = snap["selected_node_list"].clone()
    env.selected_teacher_flag = snap["selected_teacher_flag"].clone()
    env.selected_student_list = snap["selected_student_list"].clone()
    env.selected_student_flag = snap["selected_student_flag"].clone()
    env.step_state = Step_State(problems=env.problems)
