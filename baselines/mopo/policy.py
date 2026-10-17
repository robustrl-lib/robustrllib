"""Evaluation loader for the SAC policies of MOPO and RAMBO: the deterministic action
``tanh(mean)`` of the policy, on observations normalised the way the policy was trained."""
from __future__ import annotations

import numpy as np
import torch

from baselines.mopo.sac import TanhGaussianPolicy


def save_policy(path, agent, hparams, obs_dim, act_dim, mean=None, std=None) -> None:
    torch.save({"hparams": dict(hparams), "dims": [obs_dim, act_dim, agent.policy.max_action],
                "state_mean": None if mean is None else np.asarray(mean, np.float32),
                "state_std": None if std is None else np.asarray(std, np.float32),
                "policy": agent.policy.state_dict(),
                "critics": {"q1": agent.q1.state_dict(), "q2": agent.q2.state_dict()}}, path)


class SACPolicy:
    def __init__(self, net: TanhGaussianPolicy, mean, std, device="cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        dim = net.net[0].in_features
        self.mean = torch.as_tensor(np.zeros(dim) if mean is None else mean, dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(np.ones(dim) if std is None else std, dtype=torch.float32, device=self.device)

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs) -> "SACPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        obs_dim, act_dim, max_action = blob["dims"]
        net = TanhGaussianPolicy(int(obs_dim), int(act_dim), float(max_action))
        net.load_state_dict(blob["policy"])
        return cls(net, blob["state_mean"], blob["state_std"], device)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net.mean_action((obs.to(self.device) - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.actor(obs).cpu().numpy().reshape(-1)
