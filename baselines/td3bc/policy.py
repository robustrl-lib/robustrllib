"""Evaluation loader for TD3+BC: the deterministic actor on normalised states."""
from __future__ import annotations

import numpy as np
import torch

from baselines.td3bc.agent import Actor


class TD3BCPolicy:
    def __init__(self, actor: Actor, mean, std, device="cpu"):
        self.device = torch.device(device)
        self.net = actor.to(self.device).eval()
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device).reshape(-1)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=self.device).reshape(-1)

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs) -> "TD3BCPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        state_dim, action_dim, max_action = blob["dims"]
        actor = Actor(int(state_dim), int(action_dim), float(max_action))
        actor.load_state_dict(blob["actor"])
        return cls(actor, blob["state_mean"], blob["state_std"], device)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs.to(self.device) - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.actor(obs).cpu().numpy().reshape(-1)
