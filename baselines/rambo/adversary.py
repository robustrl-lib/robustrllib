"""RAMBO's adversarial model update and behaviour-cloning warm start (Rigter et al., 2022).

The dynamics ensemble is trained against the critic: from dataset states the policy is
rolled through the model and each step takes one gradient step on

    adversary_weight * mean(A * log p(s', r | s, a)) + MLE loss on real transitions,

with ``A = r + gamma * Q(s', pi(s')) * (1 - done) - Q(s, a)`` standardised over the batch
and ``log p`` the log-density of the sampled transition under the mixture of the elites
(the sample itself is held fixed: a score-function estimator). Policy, critic and model
all work on normalised observations; the termination rule sees raw ones.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class StateNormalizer:
    """Fixed observation normalisation fitted on the dataset (dimensions with std < 1e-3 keep scale 1)."""

    def __init__(self, states, device):
        self.mean = states.mean(0).astype(np.float32)
        std = states.std(0)
        std[std < 1e-3] = 1.0
        self.std = std.astype(np.float32)
        self._mean_t = torch.as_tensor(self.mean, device=device)
        self._std_t = torch.as_tensor(self.std, device=device)

    def normalize(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def unnormalize(self, x):
        return x * self._std_t + self._mean_t


def _log_prob(policy, obs, act):
    """log pi(a|s) of a given action under the tanh-squashed Gaussian."""
    mean, log_std = policy(obs)
    a = (act / policy.max_action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    eps = (torch.atanh(a) - mean) / log_std.exp()
    logp = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    return logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)


def pretrain_bc(agent, states, actions, rng, *, epochs=50, batch_size=256, lr=1e-4, objective="nll", device="cpu"):
    """Behaviour-clone the policy before RL with the actor's own optimizer (at ``lr``):
    negative log-likelihood of the dataset actions, or ``mse`` of the mean action. One
    bootstrap resample of the dataset (minus a 10% holdout, at most 2000) is reused
    in the same order every epoch."""
    n_hold = min(int(len(states) * 0.1), 2000)
    perm = rng.permutation(len(states))
    hold_s, hold_a = (torch.as_tensor(x[perm[:n_hold]], device=device) for x in (states, actions))
    train_s, train_a = states[perm[n_hold:]], actions[perm[n_hold:]]
    order = rng.integers(0, len(train_s), size=len(train_s))
    opt = agent.policy_opt
    lrs = [group["lr"] for group in opt.param_groups]
    for group in opt.param_groups:
        group["lr"] = float(lr)
    for epoch in range(int(epochs)):
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            s, a = torch.as_tensor(train_s[idx], device=device), torch.as_tensor(train_a[idx], device=device)
            if objective == "mse":
                loss = F.mse_loss(agent.policy.mean_action(s), a)
            else:
                loss = -_log_prob(agent.policy, s, a).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nll = -_log_prob(agent.policy, hold_s, hold_a).mean().item()
            mse = F.mse_loss(agent.policy.mean_action(hold_s), hold_a).item()
        print(f"[bc] epoch {epoch + 1}/{epochs} holdout nll {nll:.4f} mse {mse:.5f}", flush=True)
    for group, value in zip(opt.param_groups, lrs):
        group["lr"] = value


class AdversarialModel:
    def __init__(self, model, agent, normalizer, termination, *, weight, lr=3e-4):
        self.model, self.agent, self.norm, self.termination = model, agent, normalizer, termination
        self.weight = float(weight)
        self.opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    def step(self, obs, mle_inputs, mle_targets, rng):
        """One model update from normalised states ``obs``; returns the sampled next states."""
        model, agent = self.model, self.agent
        with torch.no_grad():
            act, _ = agent.policy.sample(obs)
        mean, logvar = model.full_width(*model.forward(model.scaled(torch.cat([obs, act], -1))))
        means = torch.cat([mean[:, :, :1], mean[:, :, 1:] + obs.unsqueeze(0)], -1)
        stds = torch.exp(0.5 * logvar)
        elites = model.elites.tolist()
        rows = torch.arange(obs.shape[0], device=obs.device)
        pick = torch.as_tensor(rng.choice(elites, size=obs.shape[0]), device=obs.device)
        with torch.no_grad():
            sample = (means + torch.randn_like(means) * stds)[pick, rows]
        reward, next_obs = sample[:, :1], sample[:, 1:]
        m, s = means[elites], stds[elites]
        member_logp = (-0.5 * (sample.unsqueeze(0) - m) ** 2 / s ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = torch.logsumexp(member_logp, dim=0) - math.log(len(elites))
        with torch.no_grad():
            next_q = agent.q(next_obs, agent.policy.mean_action(next_obs)).squeeze(-1)
            done = self.termination(self.norm.unnormalize(obs).cpu().numpy(), act.cpu().numpy(),
                                    self.norm.unnormalize(next_obs).cpu().numpy())
            done = torch.as_tensor(done, dtype=torch.float32, device=obs.device).reshape(-1)
            adv = reward.squeeze(-1) + agent.gamma * next_q * (1.0 - done) - agent.q(obs, act).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        adv_objective = (adv * log_prob).mean()
        loss = self.weight * adv_objective + model.nll(mle_inputs, mle_targets)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return next_obs.detach(), {"model_adv_objective": adv_objective.item()}
