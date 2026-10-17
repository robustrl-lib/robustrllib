"""Robust Fitted Q-Iteration (Panaganti et al., 2022).

Adapted from https://github.com/zaiyan-x/RFQI (commit 3ace2cb7e6396478203462c6f800825ff5080885,
MIT licence, see LICENSE in this folder): networks and update equations as in its ``rfqi.py``.

A BCQ-style policy (VAE behaviour model + bounded perturbation actor) and twin critics
regress a robust target under a total-variation uncertainty set of radius ``rho``: the
dual variable eta(s, a) is fitted per batch by a fresh network maximising
``-sum max(eta - V', 0) + (1 - rho) sum eta``, and the critic target is
``r - gamma * max(eta - V', 0) + (1 - rho) * gamma * eta``.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(state_dim + action_dim, 400), nn.Linear(400, 300), nn.Linear(300, action_dim)
        self.max_action, self.phi = max_action, phi

    def forward(self, state, action):
        a = F.relu(self.l2(F.relu(self.l1(torch.cat([state, action], 1)))))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(state_dim + action_dim, 400), nn.Linear(400, 300), nn.Linear(300, 1)
        self.l4, self.l5, self.l6 = nn.Linear(state_dim + action_dim, 400), nn.Linear(400, 300), nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def q1(self, state, action):
        return self.l3(F.relu(self.l2(F.relu(self.l1(torch.cat([state, action], 1))))))


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super().__init__()
        self.e1, self.e2 = nn.Linear(state_dim + action_dim, 750), nn.Linear(750, 750)
        self.mean, self.log_std = nn.Linear(750, latent_dim), nn.Linear(750, latent_dim)
        self.d1, self.d2, self.d3 = nn.Linear(state_dim + latent_dim, 750), nn.Linear(750, 750), nn.Linear(750, action_dim)
        self.max_action, self.latent_dim = max_action, latent_dim

    def forward(self, state, action):
        z = F.relu(self.e2(F.relu(self.e1(torch.cat([state, action], 1)))))
        mean, std = self.mean(z), self.log_std(z).clamp(-4, 15).exp()
        return self.decode(state, mean + std * torch.randn_like(std)), mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim), device=state.device).clamp(-0.5, 0.5)
        a = F.relu(self.d2(F.relu(self.d1(torch.cat([state, z], 1)))))
        return self.max_action * torch.tanh(self.d3(a))


class ETA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc_1, self.fc_2, self.fc_out = nn.Linear(state_dim + action_dim, 64), nn.Linear(64, 64), nn.Linear(64, 1)

    def forward(self, state, action):
        return self.fc_out(F.relu(self.fc_2(F.relu(self.fc_1(torch.cat([state, action], 1)))))).squeeze(dim=1)


class RFQI:
    def __init__(self, state_dim, action_dim, max_action, device, *, adam_lr=1e-3, adam_eps=1e-6,
                 actor_lr=3e-4, critic_lr=8e-4, vae_lr=1e-3, gamma=0.99, tau=0.005, lmbda=0.75, phi=0.1,
                 rho=0.5, eta_tol=1e-3, eta_max_steps=10_000, target_action_samples=10,
                 policy_action_samples=100, **_):
        self.device = torch.device(device)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.max_action = torch.as_tensor(max_action, dtype=torch.float32, device=self.device)
        self.gamma, self.tau, self.lmbda, self.rho = float(gamma), float(tau), float(lmbda), float(rho)
        self.adam_lr, self.adam_eps = float(adam_lr), float(adam_eps)
        self.eta_tol, self.eta_max_steps = float(eta_tol), int(eta_max_steps)
        self.target_action_samples, self.policy_action_samples = int(target_action_samples), int(policy_action_samples)
        self.fused = {"fused": True} if self.device.type == "cuda" else {}
        self.actor = Actor(state_dim, action_dim, self.max_action, phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), eps=self.adam_eps, lr=actor_lr, **self.fused)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), eps=self.adam_eps, lr=critic_lr, **self.fused)
        self.vae = VAE(state_dim, action_dim, action_dim * 2, self.max_action).to(self.device)
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=vae_lr, **self.fused)
        self.eta_high = 1.0 / (self.rho * (1.0 - self.gamma))

    def candidates(self, state):
        """The perturbed VAE actions of ``state`` [1, d] and the index of the best under Q1."""
        states = state.reshape(1, -1).repeat(self.policy_action_samples, 1)
        with torch.no_grad():
            z = torch.randn((states.shape[0], self.vae.latent_dim), device=self.device).clamp(-0.5, 0.5)
        actions = self.actor(states, self.vae.decode(states, z))
        return actions, self.critic.q1(states, actions).argmax(0)

    @torch.no_grad()
    def act(self, state) -> np.ndarray:
        actions, best = self.candidates(torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device))
        return torch.clamp(actions[best], -self.max_action, self.max_action).cpu().numpy().flatten()

    def _dual(self, next_value, state, action):
        eta_net = ETA(self.state_dim, self.action_dim).to(self.device)
        opt = torch.optim.Adam(eta_net.parameters(), lr=self.adam_lr, eps=self.adam_eps, maximize=True, **self.fused)
        previous = torch.tensor([float("inf")], device=self.device)
        for step in range(self.eta_max_steps):
            eta = eta_net(state, action)
            value = -torch.clamp(eta - next_value, min=0).sum() + (1.0 - self.rho) * eta.sum()
            opt.zero_grad(); value.backward(); opt.step()
            if torch.norm(value - previous) < self.eta_tol:
                break
            previous = value
        return eta_net(state, action).detach().clamp(0.0, self.eta_high).unsqueeze(1), step + 1

    def update(self, state, action, next_state, reward) -> dict:
        recon, mean, std = self.vae(state, action)
        vae_loss = F.mse_loss(recon, action) + 0.5 * (-0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean())
        self.vae_opt.zero_grad(); vae_loss.backward(); self.vae_opt.step()

        with torch.no_grad():
            nxt = torch.repeat_interleave(next_state, self.target_action_samples, 0)
            q1, q2 = self.critic_target(nxt, self.actor_target(nxt, self.vae.decode(nxt)))
            target_q = self.lmbda * torch.min(q1, q2) + (1.0 - self.lmbda) * torch.max(q1, q2)
            target_q = target_q.reshape(state.shape[0], -1).max(1)[0].reshape(-1, 1)
        etas, eta_steps = self._dual(target_q, state, action)
        with torch.no_grad():
            target = (reward - self.gamma * torch.clamp(etas - target_q, min=0)
                      + (1.0 - self.rho) * etas * self.gamma)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic.q1(state, self.actor(state, self.vae.decode(state))).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        with torch.no_grad():
            for net, target_net in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                for p, tp in zip(net.parameters(), target_net.parameters()):
                    tp.copy_(self.tau * p + (1.0 - self.tau) * tp)
        return {"vae_loss": vae_loss.item(), "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "eta_steps": eta_steps}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "vae": self.vae.state_dict()}

    def load_state_dict(self, state) -> None:
        for name in ("actor", "critic", "vae"):
            getattr(self, name).load_state_dict(state[name])
