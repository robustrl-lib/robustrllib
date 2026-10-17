"""Load a trained RSC-SAC policy for evaluation."""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC


class RSCPolicy:
    """Deterministic actions of an RSC-SAC checkpoint (a Stable-Baselines3 SAC file).

    The causal model is used only in training; acting needs the SAC policy alone.
    """

    def __init__(self, model: SAC):
        self.model = model
        space = model.action_space
        self._low = torch.as_tensor(space.low, dtype=torch.float32, device=model.device)
        self._high = torch.as_tensor(space.high, dtype=torch.float32, device=model.device)

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kw):
        return cls(SAC.load(str(path), device=device))

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(np.asarray(obs), deterministic=deterministic)
        return action

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Differentiable deterministic action, in the units of the action space."""
        squashed = self.model.actor(obs.to(self.model.device), deterministic=True)
        return self._low + 0.5 * (squashed + 1.0) * (self._high - self._low)
