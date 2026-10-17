"""Load a trained ATLA-PPO or ATLA-SA-PPO policy for evaluation.

    policy = ATLAPolicy.load("runs/atla_hopper/seed0/ckpt/step_4999168.pt")
    policy.reset()                          # before each episode (LSTM state)
    action = policy.predict(obs)            # obs as produced by robustrllib.tasks.make_env

The observation normaliser is frozen at its final training statistics; the
deterministic action is the mean of the Gaussian policy (vector observations) or
the most likely action (images). ``actor`` is the differentiable map from a raw
observation tensor (1, obs_dim) to the action, given for the feed-forward policy
and ``None`` for the LSTM and image policies.
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.atla.models import CtsLSTMPolicy, CtsPolicy


class ATLAPolicy:
    def __init__(self, net, obs_filter=None, recurrent=False, image=False, discrete=False, device="cpu"):
        self.net = net
        self.recurrent = recurrent
        self.image = image
        self.discrete = discrete
        self.device = device
        self.mean = self.std = None
        self.clip = None
        if obs_filter is not None:
            n = int(obs_filter["n"])
            mean = obs_filter["mean"].double().numpy()
            var = obs_filter["sum_sq"].double().numpy() / (n - 1) if n > 1 else np.square(mean)
            self.mean, self.std = mean, np.sqrt(var)
            self.clip = float(obs_filter["clip"]) or None
        self.actor = None if (recurrent or image) else self._actor

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kw):
        ckpt = torch.load(path, map_location="cpu")
        if ckpt["backend"] == "pixel":
            from stable_baselines3.common.policies import ActorCriticCnnPolicy

            if env is None:
                from robustrllib.tasks import make_env

                env = make_env(config["task"])
            net = ActorCriticCnnPolicy(env.observation_space, env.action_space, lambda _: 0.0, use_sde=False)
            net.load_state_dict(ckpt["policy"])
            net.set_training_mode(False)
            net.to(device)
            return cls(net, image=True, discrete=hasattr(env.action_space, "n"), device=device)
        recurrent = int(ckpt["hparams"]["history_length"]) > 0
        net = (CtsLSTMPolicy if recurrent else CtsPolicy)(ckpt["obs_dim"], ckpt["act_dim"])
        net.load_state_dict(ckpt["policy"])
        net.eval()
        return cls(net, obs_filter=ckpt["obs_filter"], recurrent=recurrent)

    def reset(self):
        if self.recurrent:
            self.net.reset()

    def _normalize(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
            if self.clip:
                x = np.clip(x, -self.clip, self.clip)
        return x

    def predict(self, obs, deterministic=True):
        if self.image:
            action, _ = self.net.predict(np.asarray(obs), deterministic=deterministic)
            return int(action) if self.discrete else action
        with torch.no_grad():
            x = torch.as_tensor(self._normalize(obs), dtype=torch.float32).unsqueeze(0)
            mean, std = self.net(x)
            action = mean if deterministic else mean + torch.randn_like(mean) * std
        return action[0].numpy()

    def _actor(self, obs):
        x = obs
        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=obs.dtype, device=obs.device)
            std = torch.as_tensor(self.std, dtype=obs.dtype, device=obs.device)
            x = (x - mean) / (std + 1e-8)
            if self.clip:
                x = x.clamp(-self.clip, self.clip)
        return self.net(x)[0]
