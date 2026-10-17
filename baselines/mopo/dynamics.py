"""Probabilistic ensemble dynamics of MOPO (also used by RAMBO), its model environment and
rollout pool, and the termination rules of the tasks.

The ensemble predicts a Gaussian over [reward, delta state] per member: 4 x 200 swish
layers, per-layer weight decay, learnable soft bounds on the log-variance, an input scaler,
a bootstrap resample per member, and early stopping on a holdout set; each member keeps
its best holdout epoch and the members with the lowest holdout error are the elites.
State dimensions that never change in the dataset are left out of the fit and predicted
as constant.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, members, in_dim, out_dim, weight_decay):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(members, 1, out_dim))
        self.weight_decay = float(weight_decay)
        std = 1.0 / (2.0 * np.sqrt(in_dim))
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)

    def decay_loss(self):
        return self.weight_decay * self.weight.pow(2).sum() / 2.0


class EnsembleDynamics(nn.Module):
    DECAYS = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4)

    def __init__(self, obs_dim, act_dim, *, live_mask, hidden=200, members=7, elites=5, lr=1e-3,
                 max_logvar=0.5, min_logvar=-10.0, logvar_bound_coeff=0.01, device="cpu"):
        super().__init__()
        self.members, self.num_elites, self.obs_dim = members, elites, obs_dim
        self.logvar_bound_coeff = float(logvar_bound_coeff)
        self.register_buffer("live", torch.as_tensor(np.asarray(live_mask), dtype=torch.bool))
        self.out_dim = int(self.live.sum()) + 1
        dims = [obs_dim + act_dim] + [hidden] * 4
        self.layers = nn.ModuleList(EnsembleLinear(members, dims[i], dims[i + 1], self.DECAYS[i]) for i in range(4))
        self.mean_head = EnsembleLinear(members, hidden, self.out_dim, self.DECAYS[4])
        self.var_head = EnsembleLinear(members, hidden, self.out_dim, self.DECAYS[4])
        self.max_logvar = nn.Parameter(torch.ones(1, self.out_dim) * float(max_logvar))
        self.min_logvar = nn.Parameter(torch.ones(1, self.out_dim) * float(min_logvar))
        self.register_buffer("mu", torch.zeros(obs_dim + act_dim))
        self.register_buffer("sigma", torch.ones(obs_dim + act_dim))
        self.register_buffer("elites", torch.arange(elites))
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    @property
    def device(self):
        return self.mu.device

    def _keep(self):
        return torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.nonzero(self.live)[:, 0] + 1])

    def forward(self, x):
        """Scaled inputs [E, B, in] -> (mean, logvar) over the fitted outputs."""
        for layer in self.layers:
            x = layer(x)
            x = x * torch.sigmoid(x)
        logvar = self.var_head(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return self.mean_head(x), logvar

    def full_width(self, mean, logvar):
        """Fitted outputs -> [reward, delta state] over every state dimension."""
        if bool(self.live.all()):
            return mean, logvar
        shape = (*mean.shape[:2], 1 + self.obs_dim)
        full_mean = torch.zeros(shape, device=mean.device).index_copy(2, self._keep(), mean)
        full_logvar = torch.full(shape, -30.0, device=mean.device).index_copy(2, self._keep(), logvar)
        return full_mean, full_logvar

    def scaled(self, inputs):
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        x = (x - self.mu) / self.sigma
        return x.unsqueeze(0).expand(self.members, -1, -1) if x.dim() == 2 else x

    @torch.no_grad()
    def predict(self, inputs):
        return self.full_width(*self.forward(self.scaled(inputs)))

    def nll(self, inputs, targets):
        """Gaussian negative log-likelihood (summed over members) with the regularisers."""
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        if y.dim() == 2:
            y = y.unsqueeze(0).expand(self.members, -1, -1)
        if y.shape[-1] != self.out_dim:
            y = y.index_select(2, self._keep())
        mean, logvar = self.forward(self.scaled(inputs))
        loss = (((mean - y) ** 2 * torch.exp(-logvar)).mean(dim=(1, 2)) + logvar.mean(dim=(1, 2))).sum()
        loss = loss + sum(layer.decay_loss() for layer in (*self.layers, self.mean_head, self.var_head))
        return loss + self.logvar_bound_coeff * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def _holdout(self, x, y):
        mean, _ = self.forward(self.scaled(x))
        return ((mean - y.unsqueeze(0)) ** 2).mean(dim=(1, 2)).cpu().numpy()

    def fit(self, inputs, targets, rng, batch_size=256, holdout_ratio=0.2, patience=5, max_epochs=None) -> dict:
        targets = targets[:, np.concatenate([[True], self.live.cpu().numpy()])]
        perm = rng.permutation(len(inputs))
        n_hold = min(int(len(inputs) * holdout_ratio), 1000)
        hold_x = torch.as_tensor(inputs[perm[:n_hold]], device=self.device)
        hold_y = torch.as_tensor(targets[perm[:n_hold]], device=self.device)
        train_x, train_y = inputs[perm[n_hold:]], targets[perm[n_hold:]]
        sigma = inputs.std(0)
        sigma[sigma < 1e-12] = 1.0
        self.mu.copy_(torch.as_tensor(inputs.mean(0)))
        self.sigma.copy_(torch.as_tensor(sigma))
        idx = rng.integers(len(train_x), size=(self.members, len(train_x)))
        best = np.full(self.members, np.inf)
        snapshots = [None] * self.members
        stale, epoch = 0, 0
        while True:
            epoch += 1
            for start in range(0, len(train_x), batch_size):
                b = idx[:, start:start + batch_size]
                loss = self.nll(torch.as_tensor(train_x[b]), torch.as_tensor(train_y[b]))
                self.optim.zero_grad(); loss.backward(); self.optim.step()
            idx = np.stack([rng.permutation(row) for row in idx])
            holdout = self._holdout(hold_x, hold_y)
            improved = False
            for i in range(self.members):
                if not np.isfinite(best[i]) or (best[i] - holdout[i]) / abs(best[i]) > 0.01:
                    best[i], improved = holdout[i], True
                    snapshots[i] = [(l.weight[i].detach().clone(), l.bias[i].detach().clone())
                                    for l in (*self.layers, self.mean_head, self.var_head)]
            stale = 0 if improved else stale + 1
            print(f"[dynamics] epoch {epoch} holdout mse {np.round(holdout, 4).tolist()}", flush=True)
            if stale > patience or (max_epochs and epoch >= max_epochs):
                break
        with torch.no_grad():
            for i, snapshot in enumerate(snapshots):
                for layer, (w, b) in zip((*self.layers, self.mean_head, self.var_head), snapshot):
                    layer.weight[i].copy_(w)
                    layer.bias[i].copy_(b)
        holdout = self._holdout(hold_x, hold_y)
        self.elites.copy_(torch.as_tensor(np.argsort(holdout)[:self.num_elites]))
        return {"epochs": epoch, "holdout_mse": holdout.tolist(), "elites": self.elites.tolist()}


class ModelEnv:
    """One-step model transitions: a sample from one random elite per row, the reward
    reduced by ``penalty_coeff`` times an uncertainty penalty:
    ``max_aleatoric`` = max over members of ||std_i||, ``ensemble_std`` = ||std over members of mean_i||."""

    def __init__(self, model, termination, penalty_coeff=0.0, penalty_type="max_aleatoric", unnormalize=None):
        self.model, self.termination = model, termination
        self.penalty_coeff, self.penalty_type = float(penalty_coeff), penalty_type
        self.unnormalize = unnormalize or (lambda x: x)

    @torch.no_grad()
    def step(self, obs, act):
        means, logvars = self.model.predict(torch.cat([obs, act], -1))
        means = torch.cat([means[:, :, :1], means[:, :, 1:] + obs.unsqueeze(0)], -1)
        stds = torch.exp(0.5 * logvars)
        samples = means + torch.randn_like(means) * stds
        rows = torch.arange(obs.shape[0], device=obs.device)
        members = self.model.elites[torch.randint(len(self.model.elites), (obs.shape[0],), device=obs.device)]
        sample = samples[members, rows]
        reward, next_obs = sample[:, :1], sample[:, 1:]
        if self.penalty_coeff:
            if self.penalty_type == "max_aleatoric":
                penalty = stds.norm(dim=-1).amax(dim=0)
            else:
                penalty = means.std(dim=0, unbiased=False).norm(dim=-1)
            reward = reward - self.penalty_coeff * penalty.unsqueeze(-1)
        done = self.termination(self.unnormalize(obs).cpu().numpy(), act.cpu().numpy(),
                                self.unnormalize(next_obs).cpu().numpy())
        return next_obs, reward, torch.as_tensor(done, dtype=torch.float32, device=obs.device).reshape(-1, 1)


class ModelPool:
    """First-in-first-out pool of model transitions."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity, self.ptr, self.size = int(capacity), 0, 0
        self.arrays = [np.zeros((self.capacity, d), dtype=np.float32) for d in (obs_dim, act_dim, 1, obs_dim, 1)]

    def add(self, *batch):
        idx = (self.ptr + np.arange(len(batch[0]))) % self.capacity
        for array, values in zip(self.arrays, batch):
            array[idx] = values
        self.ptr = (self.ptr + len(batch[0])) % self.capacity
        self.size = min(self.size + len(batch[0]), self.capacity)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=n)
        return [array[idx] for array in self.arrays]


def rollout(model_env, policy, pool, starts, length):
    """Branch ``length``-step rollouts of ``policy`` from ``starts`` into ``pool``; rows stop at termination."""
    obs = starts
    for _ in range(length):
        with torch.no_grad():
            act, _ = policy.sample(obs)
        next_obs, reward, done = model_env.step(obs, act)
        pool.add(obs.cpu().numpy(), act.cpu().numpy(), reward.cpu().numpy(), next_obs.cpu().numpy(), done.cpu().numpy())
        alive = done.reshape(-1) < 0.5
        if not bool(alive.any()):
            break
        obs = next_obs[alive]


# ------------------------------------------------------------- termination rules
def _hopper(obs, act, next_obs):
    height, angle = next_obs[:, 0], next_obs[:, 1]
    alive = (np.isfinite(next_obs).all(-1) * (np.abs(next_obs[:, 1:]) < 100).all(-1)
             * (height > 0.7) * (np.abs(angle) < 0.2))
    return ~alive[:, None]


def _walker2d(obs, act, next_obs):
    height, angle = next_obs[:, 0], next_obs[:, 1]
    return ~((height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0))[:, None]


def _lunarlander(obs, act, next_obs):
    """Out of the viewport, or both legs down and at rest."""
    awake = (np.abs(next_obs[:, 2]) > 0.01) | (np.abs(next_obs[:, 3]) > 0.01)
    landed = (next_obs[:, 6] > 0.5) & (next_obs[:, 7] > 0.5) & ~awake
    return ((np.abs(next_obs[:, 0]) >= 1.0) | landed | ~np.isfinite(next_obs).all(-1))[:, None]


def _never(obs, act, next_obs):
    return np.zeros((len(obs), 1), dtype=bool)


TERMINATION = {"Hopper": _hopper, "Walker2d": _walker2d, "HalfCheetah": _never, "LunarLander": _lunarlander,
               "AdroitHandDoor": _never, "PointMaze": _never}


def termination_fn(env_id: str):
    for prefix, fn in TERMINATION.items():
        if env_id.startswith(prefix):
            return fn
    raise KeyError(f"no termination rule for {env_id}")
