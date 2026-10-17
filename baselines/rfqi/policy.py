"""Evaluation loader for RFQI: the best of ``policy_action_samples`` perturbed VAE actions under Q1."""
from __future__ import annotations

import numpy as np
import torch

from baselines.rfqi.core import RFQI


class RFQIPolicy:
    def __init__(self, agent: RFQI):
        self.agent = agent
        for net in (agent.actor, agent.critic, agent.vae):
            net.eval()

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs) -> "RFQIPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        state_dim, action_dim = blob["dims"]
        agent = RFQI(int(state_dim), int(action_dim), blob["max_action"], device, **blob["hparams"])
        agent.load_state_dict(blob["agent"])
        return cls(agent)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """The selected candidate action of a ``[1, obs_dim]`` tensor, differentiable in ``obs``
        through the VAE decoder and the perturbation actor (the selection itself is held fixed)."""
        actions, best = self.agent.candidates(obs.to(self.agent.device))
        return actions[best].reshape(1, -1)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        return self.agent.act(obs)
