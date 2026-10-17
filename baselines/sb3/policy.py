"""Load a PPO, SAC or DR-SAC checkpoint written by ``baselines/sb3/train.py``."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC


class SB3Policy:
    """Deterministic actions of a Stable-Baselines3 PPO or SAC policy.

    ``predict`` takes one observation as ``robustrllib.tasks.make_env`` produces it and
    first applies the observation statistics stored in the checkpoint (``normalize``).
    ``actor`` is the same map as a differentiable torch function of a ``[1, obs_dim]``
    tensor, used by the adversarial observation shift; it is ``None`` for image
    observations and discrete actions.
    """

    def __init__(self, model):
        self.model = model
        model.policy.set_training_mode(False)
        stats = getattr(model, "obs_normalization", None)
        self.stats = None
        if stats:
            self.stats = {
                "mean": np.asarray(stats["mean"], dtype=np.float64),
                "std": np.sqrt(np.asarray(stats["var"], dtype=np.float64) + float(stats["epsilon"])),
                "clip": float(stats["clip"]),
            }

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs):
        name = ((config or {}).get("algorithm") or {}).get("name", "")
        algorithm = PPO if name == "ppo" else SAC
        return cls(algorithm.load(str(path), device=device))

    def _normalize(self, obs):
        if self.stats is None:
            return obs
        return np.clip((obs - self.stats["mean"]) / self.stats["std"], -self.stats["clip"], self.stats["clip"])

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(self._normalize(np.asarray(obs)), deterministic=deterministic)
        return action

    @property
    def actor(self):
        continuous = isinstance(self.model.action_space, gym.spaces.Box)
        vector = len(self.model.observation_space.shape) == 1
        return self._actor if continuous and vector else None

    def _actor(self, obs):
        device = self.model.device
        obs = obs.to(device=device, dtype=torch.float32)
        if self.stats is not None:
            mean = torch.as_tensor(self.stats["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(self.stats["std"], dtype=torch.float32, device=device)
            obs = torch.clamp((obs - mean) / std, -self.stats["clip"], self.stats["clip"])
        action = self.model.policy._predict(obs, deterministic=True)
        if self.model.policy.squash_output:
            # SAC acts in [-1, 1]; map to the action space as ``predict`` does
            low = torch.as_tensor(self.model.action_space.low, dtype=torch.float32, device=device)
            high = torch.as_tensor(self.model.action_space.high, dtype=torch.float32, device=device)
            action = low + 0.5 * (action + 1.0) * (high - low)
        return action
