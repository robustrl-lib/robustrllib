"""Load a trained RARL protagonist for evaluation."""
from __future__ import annotations

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


class RARLPolicy:
    """The protagonist of a RARL run, with its frozen observation normalisation."""

    def __init__(self, model, vecnorm=None):
        self.model = model
        self.vecnorm = vecnorm
        self.discrete = isinstance(model.action_space, gym.spaces.Discrete)
        if vecnorm is not None:
            rms = vecnorm.obs_rms
            self._mean = torch.as_tensor(rms.mean, dtype=torch.float32, device=model.device)
            self._std = torch.sqrt(torch.as_tensor(rms.var + vecnorm.epsilon, dtype=torch.float32,
                                                   device=model.device))

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kw):
        base = (config or {}).get("algorithm", {}).get("hparams", {}).get("base", "ppo")
        if base == "trpo":
            from sb3_contrib import TRPO as agent
        else:
            from stable_baselines3 import PPO as agent
        model = agent.load(str(path), device=device)
        stats = Path(path).parent / "vecnorm.pkl"
        vecnorm = None
        if stats.exists():
            with open(stats, "rb") as f:
                vecnorm = pickle.load(f)
            vecnorm.training = False
        return cls(model, vecnorm)

    def predict(self, obs, deterministic=True):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0]) if self.discrete else action

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Differentiable map from a batch of raw observations to the deterministic action."""
        obs = obs.to(self.model.device)
        if self.vecnorm is not None:
            clip = float(self.vecnorm.clip_obs)
            obs = torch.clamp((obs - self._mean) / self._std, -clip, clip)
        dist = self.model.policy.get_distribution(obs).distribution
        return dist.probs if self.discrete else dist.mean
