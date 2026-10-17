"""Implicit Q-Learning (Kostrikov et al., 2022).

Expectile regression of a state value V towards the twin target critics, one-step
critic targets through V (no out-of-dataset action is ever queried), and a policy
extracted by advantage-weighted regression onto the dataset actions.

The update is split into the steps the IQL-backbone mechanisms (ATLA-IQL, RSC-IQL,
ROMB, FWM, PLR-PVL) recombine: ``value_step``, ``critic_step``, ``awr_weights`` and
``actor_step``.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian; dropout regularises the hidden layers when ``dropout > 0``."""

    def __init__(self, state_dim, action_dim, max_action, hidden=256, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.max_action = float(max_action)

    def forward(self, state):
        h = self.net(state)
        return self.mean(h), self.log_std(h).clamp(-20.0, 2.0)

    def mean_action(self, state):
        return torch.tanh(self.forward(state)[0]) * self.max_action

    @staticmethod
    def _gaussian_log_prob(pre, mean, log_std, a):
        logp = -0.5 * ((pre - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        return logp.sum(-1, keepdim=True) - torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)

    def log_prob(self, state, action):
        """log pi(a|s) of a given action, including the tanh change of variables."""
        mean, log_std = self.forward(state)
        a = (action / self.max_action).clamp(-0.999999, 0.999999)
        return self._gaussian_log_prob(torch.atanh(a), mean, log_std, a)

    def sample(self, state):
        """A reparameterised action and its log-probability."""
        mean, log_std = self.forward(state)
        pre = mean + log_std.exp() * torch.randn_like(mean)
        a = torch.tanh(pre)
        return a * self.max_action, self._gaussian_log_prob(pre, mean, log_std, a)


def mlp(in_dim, hidden=256, out_dim=1):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class IQLAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, *, gamma=0.99, tau=0.005,
                 expectile=0.7, beta=3.0, adv_clip=100.0, lr=3e-4, hidden=256, dropout=0.0,
                 device="cpu", seed=0, **_):
        self.device = torch.device(device)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.gamma, self.tau = float(gamma), float(tau)
        self.expectile, self.beta, self.adv_clip = float(expectile), float(beta), float(adv_clip)
        self.seed = int(seed)

        self.policy = GaussianPolicy(state_dim, action_dim, max_action, hidden, dropout).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.value = mlp(state_dim, hidden).to(self.device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        # observation normalisation; the identity unless set_normalization is called
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)

    # ----------------------------------------------------------------- helpers
    def set_normalization(self, mean, std) -> None:
        self.state_mean = self.tensor(mean).reshape(-1)
        self.state_std = self.tensor(std).reshape(-1)

    def normalize(self, state):
        return (state - self.state_mean) / self.state_std

    def tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def unpack(self, batch):
        """(normalized s, a, r, normalized s', done) as tensors."""
        return (self.normalize(self.tensor(batch.states)), self.tensor(batch.actions),
                self.tensor(batch.rewards), self.normalize(self.tensor(batch.next_states)),
                self.tensor(batch.dones))

    def min_q(self, s, a, target=False):
        if target:
            return torch.min(self.q1_target(s, a), self.q2_target(s, a))
        return torch.min(self.q1(s, a), self.q2(s, a))

    # ------------------------------------------------------------------- steps
    def value_step(self, s, a):
        """Expectile regression of V(s) towards min_i Q_target,i(s, a)."""
        with torch.no_grad():
            target = self.min_q(s, a, target=True)
        diff = target - self.value(s)
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        loss = (weight * diff.pow(2)).mean()
        self.value_opt.zero_grad(); loss.backward(); self.value_opt.step()
        return loss

    def critic_step(self, s, a, target, weight=None):
        """Regress both critics onto ``target`` (optionally with per-row weights)."""
        w = 1.0 if weight is None else weight
        q1_loss = (w * (self.q1(s, a) - target).pow(2)).mean()
        q2_loss = (w * (self.q2(s, a) - target).pow(2)).mean()
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()
        return q1_loss, q2_loss

    @torch.no_grad()
    def awr_weights(self, s, a):
        adv = self.min_q(s, a) - self.value(s)
        return torch.exp(self.beta * adv).clamp(max=self.adv_clip), adv

    def actor_step(self, s, a, weights):
        loss = (weights * -self.policy.log_prob(s, a)).mean()
        self.policy_opt.zero_grad(); loss.backward(); self.policy_opt.step()
        return loss

    @torch.no_grad()
    def soft_update(self):
        for net, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

    def update(self, batch) -> dict:
        s, a, r, s2, d = self.unpack(batch)
        value_loss = self.value_step(s, a)
        with torch.no_grad():
            target = r + (1.0 - d) * self.gamma * self.value(s2)
        q1_loss, q2_loss = self.critic_step(s, a, target)
        weights, adv = self.awr_weights(s, a)
        policy_loss = self.actor_step(s, a, weights)
        self.soft_update()
        return {"value_loss": value_loss.item(), "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
                "policy_loss": policy_loss.item(), "adv_mean": adv.mean().item()}

    # -------------------------------------------------------------- deployment
    def acting_policy(self) -> nn.Module:
        """The network that acts at evaluation time."""
        return self.policy

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        policy = self.acting_policy()
        policy.eval()
        a = policy.mean_action(self.normalize(self.tensor(obs).reshape(1, -1)))
        policy.train()
        return a.cpu().numpy().reshape(-1)

    def state_dict(self) -> dict:
        return {name: getattr(self, name).state_dict()
                for name in ("policy", "q1", "q2", "value", "q1_target", "q2_target")}

    def load_state_dict(self, state: dict) -> None:
        for name in ("policy", "q1", "q2", "value", "q1_target", "q2_target"):
            getattr(self, name).load_state_dict(state[name])

    def save(self, path, hparams) -> None:
        """Checkpoint: the acting policy with its normalisation (what the loader needs)
        and the full agent state."""
        torch.save({
            "hparams": dict(hparams),
            "dims": [self.state_dim, self.action_dim, self.policy.max_action],
            "state_mean": self.state_mean.cpu(), "state_std": self.state_std.cpu(),
            "acting_policy": self.acting_policy().state_dict(),
            "agent": self.state_dict(),
        }, path)
