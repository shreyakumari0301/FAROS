"""Frozen MnLP (LEHD) routing backbone adapter for FAROS.

Wraps the existing LaRP/MnLP checkpoint and VRPEnv. Does not train anything.

Action space matches MnLP decoder indices:
  idx in [0, N)     -> customer (idx+1), flag=0  (direct)
  idx in [N, 2N)    -> customer (idx-N+1), flag=1 (via depot)

where N = number of customers (200 for CVRP200).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from policy.backbone import RoutingPolicy


DEFAULT_MNLP_ROOT = Path(r"C:\Users\SK\LaRP\MnLP")
DEFAULT_CHECKPOINT = DEFAULT_MNLP_ROOT / "checkpoints" / "cvrp_mnlp.pt"
DEFAULT_DATA = DEFAULT_MNLP_ROOT / "CVRP" / "data" / "vrp200_test_lkh.txt"

MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
    "mtp_depth": 4,
    "mtp_weight": 0.3,
}


def _to_local_path(path: Union[Path, str]) -> Path:
    """Map Windows paths to WSL mount paths when running under WSL."""
    p = str(path).replace("\\", "/")
    if os.name != "nt" and len(p) >= 3 and p[1] == ":" and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)

MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
    "mtp_depth": 4,
    "mtp_weight": 0.3,
}


def _to_local_path(path: Path | str) -> Path:
    """Map Windows paths to WSL mount paths when running under WSL."""
    p = str(path).replace("\\", "/")
    if os.name != "nt" and len(p) >= 3 and p[1] == ":" and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)


def _ensure_mnlp_on_path(mnlp_root: Path) -> None:
    mnlp_root = _to_local_path(mnlp_root)
    root = str(mnlp_root.resolve())
    cvrp = str((mnlp_root / "CVRP").resolve())
    for path in (root, cvrp):
        if path not in sys.path:
            sys.path.insert(0, path)


def decode_joint_action(idx: int, n_customers: int) -> Tuple[int, int]:
    """Map joint decoder index -> (customer_node, depot_flag)."""
    split_line = n_customers
    if idx >= split_line:
        return idx - split_line + 1, 1
    return idx + 1, 0


def encode_joint_action(node: int, flag: int, n_customers: int) -> int:
    """Map (customer_node, depot_flag) -> joint decoder index."""
    split_line = n_customers
    if flag == 1:
        return split_line + (node - 1)
    return node - 1


class MnLPPolicy(RoutingPolicy):
    """Frozen MnLP policy. Holds a live VRPEnv session for greedy decoding."""

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        mnlp_root: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        max_episodes: int = 128,
    ):
        super().__init__(model=None)
        self.mnlp_root = _to_local_path(mnlp_root if mnlp_root else DEFAULT_MNLP_ROOT)
        self.checkpoint = _to_local_path(checkpoint if checkpoint else DEFAULT_CHECKPOINT)
        self.data_path = _to_local_path(data_path if data_path else DEFAULT_DATA)
        self.device_str = device
        self.max_episodes = int(max_episodes)

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"MnLP checkpoint not found: {self.checkpoint}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"CVRP data not found: {self.data_path}")

        _ensure_mnlp_on_path(self.mnlp_root)
        self._torch = __import__("torch")

        if device == "cpu":
            self.device = self._torch.device("cpu")
            self._torch.set_default_tensor_type("torch.FloatTensor")
        else:
            if ":" in device:
                idx = int(device.split(":", 1)[1])
            else:
                idx = 0
            self._torch.cuda.set_device(idx)
            self.device = self._torch.device("cuda", idx)
            self._torch.set_default_tensor_type("torch.cuda.FloatTensor")

        from VRPEnv import VRPEnv
        from VRPModel import VRPModel

        self.env = VRPEnv(
            mode="test",
            data_path=str(self.data_path),
            sub_path=False,
            RRC_budget=0,
        )
        self.model = VRPModel(**MODEL_PARAMS).to(self.device)
        self.model.mode = "test"
        self.model.eval()

        ckpt = self._torch.load(str(self.checkpoint), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model  # frozen
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.env.load_raw_data(self.max_episodes)

        self._episode: Optional[int] = None
        self._current_step: int = 0
        self._done: bool = True
        self._n_customers: int = 0
        self._last_probs = None
        self._split_line: int = 0

    @property
    def n_customers(self) -> int:
        return self._n_customers

    def reset_episode(self, episode_idx: int = 0) -> Dict[str, Any]:
        """Load one CVRP instance and reset MnLP env to the start of decoding."""
        self.env.load_problems(int(episode_idx), 1)
        self.env.reset("test")
        state, _, _, done = self.env.pre_step()
        self._episode = int(episode_idx)
        self._current_step = 0
        self._done = bool(done)
        self._n_customers = int(self.env.problems.shape[1] - 1)
        self._split_line = self._n_customers
        self._step_state = state
        self._lkh_solution = self.env.solution.clone()
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Serializable-ish view of the current MnLP decision state."""
        problems = self.env.problems.detach().cpu().numpy()[0]
        coords = problems[:, :2].astype(np.float64)
        demands = problems[:, 2].astype(np.float64)
        rem_cap = float(problems[0, 3])
        visited = np.zeros(self._n_customers + 1, dtype=bool)
        visited[0] = True
        if self.env.selected_student_list.shape[1] > 0:
            for node in self.env.selected_student_list[0].tolist():
                if int(node) > 0:
                    visited[int(node)] = True
        current = 0
        if self.env.selected_student_list.shape[1] > 0:
            current = int(self.env.selected_student_list[0, -1].item())
            last_flag = int(self.env.selected_student_flag[0, -1].item())
            # After a via-depot step, vehicle is at the customer (not depot).
            # current node for distance is always last selected customer.
            _ = last_flag
        return {
            "episode": self._episode,
            "step": self._current_step,
            "done": self._done,
            "coords": coords,
            "demands": demands,
            "visited": visited,
            "current_node": current,
            "remaining_capacity": rem_cap,
            "capacity": float(self.env.raw_data_capacity.ravel()[0].item()),
            "n_customers": self._n_customers,
            "selected_nodes": self.env.selected_student_list.detach().cpu().numpy().tolist(),
            "selected_flags": self.env.selected_student_flag.detach().cpu().numpy().tolist(),
        }

    def logits(self, state=None) -> np.ndarray:
        """Return decoder scores over the joint MnLP action space [2N].

        `state` is unused: the live env session is the source of truth, matching
        the original MnLP greedy decoder.
        """
        if self._done:
            raise RuntimeError("Episode is done; call reset_episode() first")

        torch = self._torch
        with torch.no_grad():
            probs, split_line = self.model.get_action_probs(
                self._step_state,
                self.env.selected_node_list,
                self.env.raw_data_capacity,
                self._current_step,
            )
        self._last_probs = probs
        self._split_line = int(split_line)
        # Use log-probs as scores for top-k (monotonic with probs).
        scores = torch.log(probs.clamp_min(1e-12))[0].detach().cpu().numpy()
        return scores.astype(np.float64)

    def step_joint(self, action_idx: int) -> Dict[str, Any]:
        """Apply one joint decoder action to the live MnLP env."""
        if self._done:
            raise RuntimeError("Cannot step a done episode")

        torch = self._torch
        node, flag = decode_joint_action(int(action_idx), self._n_customers)
        selected = torch.tensor([node], dtype=torch.long, device=self.device)
        selected_flag = torch.tensor([flag], dtype=torch.long, device=self.device)

        if self._current_step == 0:
            selected_flag = torch.ones(1, dtype=torch.long, device=self.device)

        self._step_state, _, _, done = self.env.step(
            selected, selected, selected_flag, selected_flag
        )
        self._current_step += 1
        self._done = bool(done)
        return self.snapshot()

    def greedy_action(self) -> int:
        scores = self.logits()
        return int(np.argmax(scores))

    def tour_cost(self) -> float:
        """MnLP tour length for the completed student solution (same as evaluate)."""
        if not self._done:
            raise RuntimeError("Tour incomplete")
        problems = self.env.problems
        solution = torch_cat_solution(self.env)
        return float(self.env._get_travel_distance_2(problems, solution).item())

    def greedy_rollout_cost(self, episode_idx: int = 0) -> float:
        """Full greedy decode on one episode; returns tour cost."""
        self.reset_episode(episode_idx)
        while not self._done:
            action = self.greedy_action()
            self.step_joint(action)
        return self.tour_cost()

    def instance_meta(self, episode_idx: int = 0) -> Dict[str, Any]:
        self.env.load_problems(int(episode_idx), 1)
        problems = self.env.problems.detach().cpu().numpy()[0]
        return {
            "episode": int(episode_idx),
            "n_customers": int(problems.shape[0] - 1),
            "capacity": float(self.env.raw_data_capacity.ravel()[0].item()),
            "lkh_cost": float(self.env.raw_data_cost[episode_idx].item()),
            "coords": problems[:, :2],
            "demands": problems[:, 2],
        }


def torch_cat_solution(env):
    import torch

    return torch.cat(
        (
            env.selected_student_list.reshape(1, -1, 1),
            env.selected_student_flag.reshape(1, -1, 1),
        ),
        dim=2,
    )


def topk_feasible_mnlp(
    logits: np.ndarray,
    policy: MnLPPolicy,
    k: int = 5,
    prob_eps: float = 1e-6,
) -> List[Dict[str, float]]:
    """Top-K feasible joint actions using MnLP visit masking (matches LaRP)."""
    _ensure_mnlp_on_path(policy.mnlp_root)
    import torch
    from rollout_planner import filter_feasible_candidates

    probs = torch.from_numpy(np.exp(logits)).float().unsqueeze(0)
    # Renormalize for numerical safety
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    nodes, flags, valid = filter_feasible_candidates(
        probs,
        policy._split_line,
        policy.env.selected_node_list,
        int(k),
        prob_eps=prob_eps,
    )
    out: List[Dict[str, float]] = []
    for i in range(nodes.shape[1]):
        if not bool(valid[0, i].item()):
            continue
        node = int(nodes[0, i].item())
        flag = int(flags[0, i].item())
        idx = encode_joint_action(node, flag, policy.n_customers)
        out.append(
            {
                "action": idx,
                "node": node,
                "flag": flag,
                "score": float(logits[idx]),
            }
        )
    return out
