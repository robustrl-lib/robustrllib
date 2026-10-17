"""Evaluation loader for RORL: the deterministic action ``tanh(mean)`` of the released policy."""
from __future__ import annotations

import numpy as np
import torch

from baselines.rorl.agent import make_policy


class RORLPolicy:
    def __init__(self, policy, device="cpu"):
        self.device = torch.device(device)
        self.policy = policy.eval()

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs) -> "RORLPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        state_dim, action_dim = blob["dims"]
        policy = make_policy(int(state_dim), int(action_dim), blob["obs_mean"], blob["obs_std"], blob["hparams"], device)
        policy.load_state_dict(blob["policy"])
        return cls(policy, device)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs.to(self.device), deterministic=True)[0]

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, -1)
        return np.clip(self.actor(obs)[0].cpu().numpy(), -1.0, 1.0)
