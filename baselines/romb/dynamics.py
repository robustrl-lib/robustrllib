"""Probabilistic ensemble world model of ROMB: Gaussian p(delta state, reward | state, action)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, in_dim, out_dim, members, weight_decay=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(members, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(members, 1, out_dim))
        nn.init.trunc_normal_(self.weight, std=1.0 / (2.0 * in_dim ** 0.5))
        self.weight_decay = float(weight_decay)

    def forward(self, x):
        pattern = "bi,eio->ebo" if x.ndim == 2 else "ebi,eio->ebo"
        return torch.einsum(pattern, x, self.weight) + self.bias

    def decay_loss(self):
        return 0.5 * self.weight_decay * self.weight.pow(2).sum()


class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims, members, elites, weight_decays, device):
        super().__init__()
        self.obs_dim, self.members, self.num_elites = obs_dim, members, elites
        self.device = torch.device(device)
        dims = (obs_dim + act_dim, *hidden_dims)
        self.layers = nn.ModuleList(EnsembleLinear(dims[i], dims[i + 1], members, weight_decays[i])
                                    for i in range(len(dims) - 1))
        self.output = EnsembleLinear(dims[-1], 2 * (obs_dim + 1), members, weight_decays[-1])
        self.max_logvar = nn.Parameter(torch.ones(obs_dim + 1) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(obs_dim + 1) * -10.0)
        self.register_buffer("elites", torch.arange(elites))
        self.to(self.device)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        for layer in self.layers:
            x = layer(x)
            x = x * torch.sigmoid(x)
        mean, logvar = torch.chunk(self.output(x), 2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def regularization(self):
        decay = sum(layer.decay_loss() for layer in (*self.layers, self.output))
        return decay + 0.001 * self.max_logvar.sum() - 0.001 * self.min_logvar.sum()


def _nll(mean, logvar, target):
    return (((mean - target).pow(2) * torch.exp(-logvar)).mean(dim=(1, 2)) + logvar.mean(dim=(1, 2))).sum()


class GaussianDynamics:
    """The ensemble with its input scaler and the task's termination rule."""

    def __init__(self, obs_dim, act_dim, termination, *, hidden_dims, members, elites, weight_decays, lr, device):
        self.model = EnsembleModel(obs_dim, act_dim, hidden_dims, members, elites, weight_decays, device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.termination = termination
        self.mu = self.std = None

    def scale(self, x):
        if torch.is_tensor(x):
            return (x - torch.as_tensor(self.mu, dtype=x.dtype, device=x.device)) / torch.as_tensor(
                self.std, dtype=x.dtype, device=x.device)
        return (x - self.mu) / self.std

    def fit(self, data, epochs, batch_size, rng) -> dict:
        """Maximum likelihood on the dataset with a bootstrap per member; the members with
        the lowest holdout error become the elites."""
        inputs = np.concatenate([data.states, data.actions], -1).astype(np.float32)
        targets = np.concatenate([data.next_states - data.states, data.rewards], -1).astype(np.float32)
        perm = rng.permutation(len(inputs))
        n_hold = min(int(0.2 * len(inputs)), 1000)
        hold_in, hold_tg, train_in, train_tg = (inputs[perm[:n_hold]], targets[perm[:n_hold]],
                                                inputs[perm[n_hold:]], targets[perm[n_hold:]])
        self.mu = train_in.mean(0, keepdims=True)
        self.std = train_in.std(0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        train_in, hold_in = self.scale(train_in), self.scale(hold_in)
        members, n = self.model.members, len(train_in)
        idx = rng.integers(n, size=(members, n))
        for epoch in range(int(epochs)):
            idx = idx[np.arange(members)[:, None], np.argsort(rng.uniform(size=idx.shape), axis=-1)]
            for start in range(0, n, int(batch_size)):
                b = idx[:, start:start + int(batch_size)]
                mean, logvar = self.model(train_in[b])
                target = torch.as_tensor(train_tg[b], device=self.model.device)
                loss = _nll(mean, logvar, target) + self.model.regularization()
                self.optim.zero_grad(); loss.backward(); self.optim.step()
            print(f"[dynamics] epoch {epoch + 1}/{epochs} loss {loss.item():.4f}", flush=True)
        with torch.no_grad():
            mean, _ = self.model(hold_in)
            hold = ((mean - torch.as_tensor(hold_tg, device=self.model.device)) ** 2).mean(dim=(1, 2)).cpu().numpy()
        elites = np.argsort(hold)[:self.model.num_elites]
        self.model.elites = torch.as_tensor(elites, device=self.model.device)
        return {"holdout_mse": float(np.sort(hold)[:len(elites)].mean()), "elites": elites.tolist()}

    @torch.no_grad()
    def step(self, obs, act, rng):
        """One model transition from one random elite per row, with its log-density."""
        mean, logvar = self.model(self.scale(np.concatenate([obs, act], -1).astype(np.float32)))
        mean, std = mean.cpu().numpy(), np.sqrt(np.exp(logvar.cpu().numpy()))
        samples = mean + rng.normal(size=mean.shape).astype(np.float32) * std
        log_probs = -0.5 * (((samples - mean) / (std + 1e-8)) ** 2 + 2.0 * np.log(std + 1e-8) + np.log(2.0 * np.pi))
        rows = np.arange(obs.shape[0])
        members = rng.choice(self.model.elites.cpu().numpy(), size=obs.shape[0])
        chosen = samples[members, rows]
        next_obs = obs + chosen[:, :-1]
        done = self.termination(obs, act, next_obs).reshape(-1).astype(np.float32)
        return (next_obs.astype(np.float32), chosen[:, -1].astype(np.float32), done,
                {"log_prob": log_probs[members, rows].sum(-1).astype(np.float32),
                 "sample": chosen.astype(np.float32), "member": members.astype(np.int64)})

    def log_prob(self, obs, act, sample, members):
        mean, logvar = self.model(self.scale(torch.cat([obs, act], -1)))
        rows = torch.arange(obs.shape[0], device=obs.device)
        mean, logvar = mean[members, rows], logvar[members, rows]
        return -0.5 * ((sample - mean).pow(2) / torch.exp(logvar) + logvar + np.log(2.0 * np.pi)).sum(-1)

    def supervised_loss(self, obs, act, next_obs, rew):
        mean, logvar = self.model(self.scale(torch.cat([obs, act], -1)))
        target = torch.cat([next_obs - obs, rew.reshape(-1, 1)], -1).unsqueeze(0).expand_as(mean)
        return _nll(mean, logvar, target) + self.model.regularization()


# --------------------------------------------------------------- termination rules
def _hopper(obs, act, next_obs):
    height, angle = next_obs[:, 0], next_obs[:, 1]
    alive = (np.isfinite(next_obs).all(-1) * np.abs(next_obs[:, 1:] < 100).all(-1)
             * (height > 0.7) * (np.abs(angle) < 0.2))
    return ~alive[:, None]


def _walker2d(obs, act, next_obs):
    height, angle = next_obs[:, 0], next_obs[:, 1]
    alive = (np.all(next_obs > -100, -1) * np.all(next_obs < 100, -1)
             * (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0))
    return ~alive[:, None]


def _halfcheetah(obs, act, next_obs):
    return ~(np.all(next_obs > -100, -1) & np.all(next_obs < 100, -1))[:, None]


def _lunarlander(obs, act, next_obs):
    awake = (np.abs(next_obs[:, 2]) > 0.01) | (np.abs(next_obs[:, 3]) > 0.01)
    landed = (next_obs[:, 6] > 0.5) & (next_obs[:, 7] > 0.5) & ~awake
    return ((np.abs(next_obs[:, 0]) >= 1.0) | landed | ~np.isfinite(next_obs).all(-1))[:, None]


def _never(obs, act, next_obs):
    return np.zeros((len(obs), 1), dtype=bool)


TERMINATION = {"Hopper": _hopper, "Walker2d": _walker2d, "HalfCheetah": _halfcheetah,
               "LunarLander": _lunarlander, "AdroitHandDoor": _never, "PointMaze": _never}


def termination_fn(env_id: str):
    """The termination rule of a task: model rollouts stop where the real episode would."""
    for prefix, fn in TERMINATION.items():
        if env_id.startswith(prefix):
            return fn
    raise KeyError(f"no termination rule for {env_id}")
