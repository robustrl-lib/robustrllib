"""Evaluation loader for the IQL-backbone methods (IQL, SynthER, ATLA-IQL, RSC-IQL, ROMB, FWM, PLR-PVL).

Every one of them deploys an IQL Gaussian policy; the checkpoint carries the acting
network and the observation normalisation it was trained with.
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.iql.agent import GaussianPolicy


class IQLPolicy:
    def __init__(self, policy: GaussianPolicy, mean, std, device="cpu"):
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device).reshape(-1)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=self.device).reshape(-1)

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs) -> "IQLPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        state_dim, action_dim, max_action = blob["dims"]
        hp = blob["hparams"]
        policy = GaussianPolicy(int(state_dim), int(action_dim), float(max_action),
                                int(hp.get("hidden", 256)), float(hp.get("dropout", 0.0)))
        policy.load_state_dict(blob["acting_policy"])
        return cls(policy, blob["state_mean"], blob["state_std"], device)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action of a ``[B, obs_dim]`` tensor, differentiable in ``obs``."""
        return self.policy.mean_action((obs.to(self.device) - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.actor(obs).cpu().numpy().reshape(-1)
