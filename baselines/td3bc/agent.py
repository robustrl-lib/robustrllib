"""TD3+BC (Fujimoto & Gu, 2021), following github.com/sfujim/TD3_BC.

TD3 with a behaviour-cloning term in the actor loss,
``-lambda * Q(s, pi(s)) + (pi(s) - a)^2`` with ``lambda = alpha / mean|Q|``, on states
normalised by the dataset mean and standard deviation (+1e-3).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, action_dim), nn.Tanh())
        self.max_action = float(max_action)

    def forward(self, state):
        return self.max_action * self.net(state)


def _q_net(in_dim):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1, self.q2 = _q_net(state_dim + action_dim), _q_net(state_dim + action_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


class TD3BCAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, *, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, alpha=2.5, lr=3e-4, device="cpu", **_):
        self.device = torch.device(device)
        self.state_dim, self.action_dim, self.max_action = int(state_dim), int(action_dim), float(max_action)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau, self.alpha = float(gamma), float(tau), float(alpha)
        self.policy_noise, self.noise_clip, self.policy_freq = float(policy_noise), float(noise_clip), int(policy_freq)
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)
        self.steps = 0

    def set_normalization(self, mean, std):
        self.state_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device).reshape(-1)
        self.state_std = torch.as_tensor(std, dtype=torch.float32, device=self.device).reshape(-1)

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _norm(self, s):
        return (s - self.state_mean) / self.state_std

    def update(self, batch) -> dict:
        self.steps += 1
        state, action = self._norm(self._t(batch.states)), self._t(batch.actions)
        next_state, reward, not_done = self._norm(self._t(batch.next_states)), self._t(batch.rewards), 1.0 - self._t(batch.dones)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target = reward + not_done * self.gamma * torch.min(*self.critic_target(next_state, next_action))
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()
        stats = {"critic_loss": critic_loss.item()}
        if self.steps % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.q1(torch.cat([state, pi], 1))
            actor_loss = -self.alpha / q.abs().mean().detach() * q.mean() + F.mse_loss(pi, action)
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            with torch.no_grad():
                for net, target_net in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                    for p, tp in zip(net.parameters(), target_net.parameters()):
                        tp.mul_(1.0 - self.tau).add_(self.tau * p)
            stats["actor_loss"] = actor_loss.item()
        return stats

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        return self.actor(self._norm(self._t(obs).reshape(1, -1))).cpu().numpy().reshape(-1)

    def save(self, path, hparams) -> None:
        torch.save({"hparams": dict(hparams), "dims": [self.state_dim, self.action_dim, self.max_action],
                    "state_mean": self.state_mean.cpu(), "state_std": self.state_std.cpu(),
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
