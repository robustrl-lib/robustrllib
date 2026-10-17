"""Soft actor-critic used by MOPO and RAMBO: (256, 256) networks, twin critics, automatic
temperature towards ``target_entropy`` (capped at ``max_alpha``), and an optional
behaviour-cloning term on dataset transitions (RAMBO)."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action=1.0, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.max_action = float(max_action)

    def forward(self, obs):
        h = self.net(obs)
        return self.mean(h), self.log_std(h).clamp(-20.0, 2.0)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        eps = torch.randn_like(mean)
        a = torch.tanh(mean + log_std.exp() * eps)
        logp = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a * self.max_action, logp

    def mean_action(self, obs):
        return torch.tanh(self.forward(obs)[0]) * self.max_action


def q_net(obs_dim, act_dim, hidden=256):
    return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class SACAgent:
    def __init__(self, obs_dim, act_dim, max_action=1.0, gamma=0.99, tau=5e-3, critic_lr=3e-4,
                 actor_lr=3e-4, alpha_lr=3e-4, target_entropy=-3.0, max_alpha=10.0, device="cpu"):
        self.device = torch.device(device)
        self.gamma, self.tau, self.target_entropy = float(gamma), float(tau), float(target_entropy)
        self.max_log_alpha = math.log(float(max_alpha))
        self.policy = TanhGaussianPolicy(obs_dim, act_dim, max_action).to(self.device)
        self.q1, self.q2 = q_net(obs_dim, act_dim).to(self.device), q_net(obs_dim, act_dim).to(self.device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def q(self, s, a):
        return torch.min(self.q1(torch.cat([s, a], -1)), self.q2(torch.cat([s, a], -1)))

    def update(self, s, a, r, s2, d, bc_obs=None, bc_act=None, bc_weight=0.0) -> dict:
        """One SAC step; with ``bc_weight > 0`` the actor loss adds
        ``bc_weight * mean|Q| * MSE(pi(bc_obs), bc_act)``."""
        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)
            q_next = torch.min(self.q1_target(torch.cat([s2, a2], -1)), self.q2_target(torch.cat([s2, a2], -1)))
            target = r + (1.0 - d) * self.gamma * (q_next - self.alpha * logp2)
        q1, q2 = self.q1(torch.cat([s, a], -1)), self.q2(torch.cat([s, a], -1))
        q1_loss, q2_loss = F.mse_loss(q1, target), F.mse_loss(q2, target)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        a_new, logp = self.policy.sample(s)
        q_new = self.q(s, a_new)
        policy_loss = (self.alpha.detach() * logp - q_new).mean()
        if bc_weight > 0.0:
            bc = F.mse_loss(self.policy.mean_action(bc_obs), bc_act)
            policy_loss = policy_loss + bc_weight * q_new.abs().mean().detach() * bc
        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(max=self.max_log_alpha)
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)
        return {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": self.alpha.item(), "q1_mean": q1.mean().item()}

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, -1)
        return self.policy.mean_action(obs).cpu().numpy().reshape(-1)
