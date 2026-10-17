"""A minimal Gaussian-policy REINFORCE, written to be copied when adding an algorithm.

It shows the whole contract: a network, an algorithm body, and the four
:class:`Algo` methods. An algorithm card names it with
``class: robustrllib.algos.template_reinforce:Reinforce`` and
``entry: baselines/train_algo.py``; no edit to the library is needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from robustrllib.algos.base import Algo


class _Policy(nn.Module):  # <-- your network
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mu = self.mu(self.body(obs))
        return torch.distributions.Normal(mu, self.log_std.exp())


class Reinforce(Algo):  # <-- your algorithm body
    def __init__(self, lr=3e-4, gamma=0.99, hidden=(64, 64), device="cpu", **_):
        super().__init__()
        self.lr, self.gamma, self.hidden, self.device = lr, gamma, tuple(hidden), device
        self.policy = None

    def _build(self, env):
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.policy = _Policy(obs_dim, act_dim, self.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def train(self, env, total_timesteps, seed=None):
        self._build(env)
        env.action_space.seed(seed)
        steps, episode = 0, 0
        while steps < total_timesteps:
            # seed the first reset only, so that every episode starts from a new state
            obs, _ = env.reset(seed=seed if episode == 0 else None)
            episode += 1
            logps, rewards = [], []
            done = False
            while not done:
                t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
                dist = self.policy.dist(t)
                a = dist.sample()
                logps.append(dist.log_prob(a).sum())
                obs, r, term, trunc, _ = env.step(a.cpu().numpy())
                rewards.append(r); steps += 1; done = term or trunc
            ret, returns = 0.0, []
            for r in reversed(rewards):
                ret = r + self.gamma * ret; returns.insert(0, ret)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            loss = -(torch.stack(logps) * returns).sum()
            self.opt.zero_grad(); loss.backward(); self.opt.step()

    def predict(self, obs, deterministic=True):
        t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.policy.dist(t)
            a = dist.mean if deterministic else dist.sample()
        return a.cpu().numpy()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    @classmethod
    def load(cls, path, env=None, device="cpu", config=None, **kwargs):
        hparams = dict((config or {}).get("algorithm", {}).get("hparams") or {})
        algo = cls(device=device, **hparams)
        algo._build(env)
        algo.policy.load_state_dict(torch.load(path, map_location=algo.device))
        return algo
