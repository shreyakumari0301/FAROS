from .backbone import NearestNeighborPolicy, RoutingPolicy, load_policy
from .topk import topk_feasible

__all__ = [
    "RoutingPolicy",
    "NearestNeighborPolicy",
    "load_policy",
    "topk_feasible",
]


def __getattr__(name: str):
    if name in ("MnLPPolicy", "topk_feasible_mnlp", "decode_joint_action"):
        from policy import mnlp_adapter as _m

        return getattr(_m, name)
    raise AttributeError(name)
