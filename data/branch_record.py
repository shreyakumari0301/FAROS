from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class StructuralSnapshot:
    remaining_customer_ratio: float
    remaining_demand_ratio: float
    remaining_capacity_ratio: float
    capacity_slack: float
    capacity_pressure: float
    fragmentation: float
    depot_distance: float
    routes_lower_bound: float
    remaining_customers: int
    remaining_demand: float
    prefix_cost: float
    phi: List[float] = field(default_factory=list)

    @classmethod
    def from_labels(cls, labels: Dict[str, float]) -> "StructuralSnapshot":
        from data.structural_labels import phi_vector

        return cls(
            remaining_customer_ratio=float(labels["remaining_customer_ratio"]),
            remaining_demand_ratio=float(labels["remaining_demand_ratio"]),
            remaining_capacity_ratio=float(labels["remaining_capacity_ratio"]),
            capacity_slack=float(labels.get("capacity_slack", labels["remaining_capacity_ratio"])),
            capacity_pressure=float(labels.get("capacity_pressure", 1.0 - labels["remaining_capacity_ratio"])),
            fragmentation=float(labels["fragmentation"]),
            depot_distance=float(labels.get("depot_distance", 0.0)),
            routes_lower_bound=float(labels.get("routes_lower_bound", 0.0)),
            remaining_customers=int(labels["remaining_customers"]),
            remaining_demand=float(labels.get("remaining_demand", 0.0)),
            prefix_cost=float(labels["prefix_cost"]),
            phi=phi_vector(labels),
        )


# Backward-compatible alias
HorizonSnapshot = StructuralSnapshot


@dataclass
class BranchRecord:
    instance_id: int
    step: int

    action: int
    node: int
    flag: int
    policy_score: float
    candidate_rank: int

    immediate_cost: float
    completion_cost: float

    regret: float = 0.0
    advantage_vs_greedy: float = 0.0

    # Exact immediate future: s_{t+1} = T(s_t, a)
    post_action_state: Dict[str, Any] = field(default_factory=dict)

    # Rollout futures: s_{t+H} after action + (H-1) greedy steps. H=1 == post_action.
    rollout_states: Dict[str, StructuralSnapshot] = field(default_factory=dict)
    future_states: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "step": self.step,
            "action": self.action,
            "node": self.node,
            "flag": self.flag,
            "policy_score": self.policy_score,
            "candidate_rank": self.candidate_rank,
            "immediate_cost": self.immediate_cost,
            "completion_cost": self.completion_cost,
            "regret": self.regret,
            "advantage_vs_greedy": self.advantage_vs_greedy,
            "post_action_state": self.post_action_state,
            "future_states": self.future_states,
            "rollout_states": {
                str(h): (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                for h, v in self.rollout_states.items()
            },
            # Alias kept so older inspect scripts still work.
            "horizons": {
                str(h): (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                for h, v in self.rollout_states.items()
            },
        }


@dataclass
class BranchGroup:
    instance_id: int
    step: int
    state: Dict[str, Any]
    greedy_action: int
    greedy_node: int
    greedy_flag: int
    candidates: List[BranchRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "step": self.step,
            "state": self.state,
            "greedy_action": self.greedy_action,
            "greedy_node": self.greedy_node,
            "greedy_flag": self.greedy_flag,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def assign_relative_labels(group: BranchGroup) -> None:
    if not group.candidates:
        return
    best_cost = min(c.completion_cost for c in group.candidates)
    greedy_cost = group.candidates[0].completion_cost
    for c in group.candidates:
        c.regret = float(c.completion_cost - best_cost)
        c.advantage_vs_greedy = float(greedy_cost - c.completion_cost)
