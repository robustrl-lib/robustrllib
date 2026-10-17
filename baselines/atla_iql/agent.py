"""ATLA-IQL: IQL trained against a learned state adversary (Zhang et al., 2021, offline form).

A state adversary proposes a bounded perturbation ``delta = eps * tanh(raw)`` of the
(normalized) observation and is trained at every update to lower the critic's value of
the policy's action at the perturbed state. ``eps`` grows linearly from 0 to ``adv_eps``
over the first ``adv_eps_ramp_steps`` updates. Every ``robust_every``-th update the actor
loss becomes ``(1 - actor_adv_ratio) * clean + actor_adv_ratio * adversarial``, the
adversarial term being the same advantage-weighted regression at the perturbed states;
value and critic updates always use clean states. An exponential moving average of the
policy (decay ``ema_decay``) is the policy that acts.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.iql.agent import IQLAgent


class StateAdversary(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(state_dim, hidden), nn.Linear(hidden, hidden)
        self.mean, self.log_std = nn.Linear(hidden, state_dim), nn.Linear(hidden, state_dim)

    def forward(self, state):
        h = F.relu(self.fc2(F.relu(self.fc1(state))))
        return self.mean(h), self.log_std(h).clamp(-5.0, 2.0).exp()

    def perturbation(self, state, eps, deterministic=False):
        mean, std = self.forward(state)
        raw = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
        return torch.tanh(raw) * float(eps)

    def entropy(self, state):
        _, std = self.forward(state)
        return (torch.log(std) + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(dim=1)


class ATLAIQLAgent(IQLAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, *, adv_eps=0.025,
                 adv_eps_ramp_steps=15000, adv_steps=1, adv_ratio=0.5, actor_adv_ratio=0.1,
                 adv_lr=3e-4, adv_l2_coeff=0.02, adv_entropy_coeff=0.0, robust_every=10,
                 ema_decay=0.995, **kwargs):
        super().__init__(state_dim, action_dim, max_action, **kwargs)
        self.adv_eps_target, self.adv_eps_ramp_steps = float(adv_eps), int(adv_eps_ramp_steps)
        self.adv_steps, self.adv_ratio = int(adv_steps), float(adv_ratio)
        self.actor_adv_ratio = float(actor_adv_ratio)
        self.adv_l2_coeff, self.adv_entropy_coeff = float(adv_l2_coeff), float(adv_entropy_coeff)
        self.robust_every, self.ema_decay = int(robust_every), float(ema_decay)
        self.adversary = StateAdversary(state_dim).to(self.device)
        self.adversary_opt = torch.optim.Adam(self.adversary.parameters(), lr=float(adv_lr))
        self.ema_policy = copy.deepcopy(self.policy).requires_grad_(False)
        self.steps = 0
        self.adv_eps = 0.0

    def _perturb(self, s, ratio, deterministic):
        delta = self.adversary.perturbation(s, self.adv_eps, deterministic)
        if ratio < 1.0:
            delta = delta * (torch.rand(s.shape[0], 1, device=s.device) < ratio).to(s.dtype)
        return s + delta, delta

    def _adversary_step(self, s) -> dict:
        if self.adv_steps <= 0 or self.adv_eps <= 0.0:
            return {}
        nets = (self.policy, self.q1, self.q2, self.value)
        for net in nets:
            net.requires_grad_(False)
        for _ in range(self.adv_steps):
            adv_s, delta = self._perturb(s, self.adv_ratio, deterministic=False)
            q_pi = self.min_q(adv_s, self.policy.mean_action(adv_s))
            loss = (q_pi.mean() - self.adv_entropy_coeff * self.adversary.entropy(s).mean()
                    + self.adv_l2_coeff * delta.pow(2).mean())
            self.adversary_opt.zero_grad(); loss.backward(); self.adversary_opt.step()
        for net in nets:
            net.requires_grad_(True)
        return {"adv_loss": loss.item(), "adv_eps": self.adv_eps}

    def update(self, batch) -> dict:
        self.steps += 1
        ramp = 1.0 if self.adv_eps_ramp_steps <= 0 else min(1.0, self.steps / self.adv_eps_ramp_steps)
        self.adv_eps = self.adv_eps_target * ramp
        robust = self.robust_every > 0 and self.steps % self.robust_every == 0

        s, a, r, s2, d = self.unpack(batch)
        value_loss = self.value_step(s, a)
        with torch.no_grad():
            target = r + (1.0 - d) * self.gamma * self.value(s2)
        q1_loss, q2_loss = self.critic_step(s, a, target)

        weights, _ = self.awr_weights(s, a)
        mix = self.actor_adv_ratio if (robust and self.adv_eps > 0) else 0.0
        policy_loss = (weights * -self.policy.log_prob(s, a)).mean()
        if mix > 0:
            with torch.no_grad():
                adv_s, _ = self._perturb(s, 1.0, deterministic=True)
            policy_loss = (1.0 - mix) * policy_loss + mix * (weights * -self.policy.log_prob(adv_s, a)).mean()
        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

        self.soft_update()
        with torch.no_grad():
            for p, ep in zip(self.policy.parameters(), self.ema_policy.parameters()):
                ep.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * p)
        stats = {"value_loss": value_loss.item(), "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
                 "policy_loss": policy_loss.item()}
        stats.update(self._adversary_step(s.detach()))
        return stats

    def acting_policy(self):
        return self.ema_policy

    def state_dict(self) -> dict:
        return {**super().state_dict(), "ema_policy": self.ema_policy.state_dict(),
                "adversary": self.adversary.state_dict()}
