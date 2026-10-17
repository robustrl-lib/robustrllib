"""FWM: IQL whose critic also regresses h-step returns of a flow-matching world model.

Every ``robust_every``-th update samples its own batch and, after IQL's value and actor
steps, regresses the critics onto two h-step targets for every (s, a):

    real      = sum_t gamma^t r[t]     + gamma^h V(s[h])       over the dataset's next h steps
    generated = sum_t gamma^t r_gen[t] + gamma^h V(s_gen[h])   over a window sampled from the
                                                                 flow model given (s, a) and
                                                                 the transition's return-to-go

Real windows are zero-padded past the end of the episode. All other updates are plain IQL;
``mechanism = False`` turns the h-step updates off (the IQL warm start of some cards).
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.fwm.flow import window_returns
from baselines.iql.agent import IQLAgent


class Windows:
    """Return-to-go and the real future window of every transition of a dataset."""

    def __init__(self, episodes, window: int, gamma: float):
        self.episodes, self.window = episodes, int(window)
        self.episode = np.concatenate([np.full(len(ep.rewards), i) for i, ep in enumerate(episodes)])
        self.step = np.concatenate([np.arange(len(ep.rewards)) for ep in episodes])
        self.rtg = np.concatenate([window_returns(ep.rewards, self.window, gamma) for ep in episodes])

    def future(self, idx) -> np.ndarray:
        """[B, window, state_dim + 1]: (state, reward) of the next ``window`` steps."""
        dim = self.episodes[0].observations.shape[1]
        out = np.zeros((len(idx), self.window, dim + 1), dtype=np.float32)
        for row, i in enumerate(idx):
            ep, t = self.episodes[self.episode[i]], self.step[i]
            n = min(self.window, len(ep.rewards) - t)
            out[row, :n, :-1] = ep.observations[t:t + n]
            out[row, :n, -1] = ep.rewards[t:t + n]
        return out


class FWMAgent(IQLAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, *, robust_every=10, h=5, window=12,
                 fm_n_steps=5, fm_ode_solver="midpoint", fm_guidance_weight=2.0, **kwargs):
        super().__init__(state_dim, action_dim, max_action, **kwargs)
        self.robust_every, self.h, self.window = int(robust_every), int(h), int(window)
        self.fm_n_steps, self.fm_ode_solver = int(fm_n_steps), str(fm_ode_solver)
        self.fm_guidance_weight = float(fm_guidance_weight)
        self.batch_size = int(kwargs.get("batch_size", 256))
        self.rng = np.random.default_rng(self.seed)
        self.mechanism = True
        self.flows = self.data = self.windows = None
        self.steps = 0

    def prepare(self, data, flows) -> None:
        self.data, self.flows = data, flows
        self.windows = Windows(data.episodes, self.window, self.gamma)

    def h_step_target(self, states, rewards, horizon, discounts):
        """sum_t gamma^t r[t] + gamma^h V(s[h]) (the bootstrap only when the window reaches h)."""
        out = (rewards[:, :horizon] * discounts).sum(dim=1, keepdim=True)
        if self.h < states.shape[1]:
            out = out + self.gamma ** self.h * self.value(self.normalize(states[:, self.h, :]))
        return out

    def generated_target(self, s_raw, a, rtg, horizon, discounts):
        states, rewards = self.flows[0].sample(torch.cat([s_raw, a], -1), rtg, self.fm_n_steps,
                                               self.fm_ode_solver, self.fm_guidance_weight)
        return self.h_step_target(states, rewards, horizon, discounts)

    def _fm_update(self) -> dict:
        idx = self.rng.integers(0, len(self.data), size=self.batch_size)
        batch = self.data.batch(idx)
        s_raw = self.tensor(batch.states)
        s, a = self.normalize(s_raw), self.tensor(batch.actions)
        value_loss = self.value_step(s, a)
        weights, adv = self.awr_weights(s, a)
        policy_loss = self.actor_step(s, a, weights)
        with torch.no_grad():
            future = self.tensor(self.windows.future(idx))
            rtg = self.tensor(np.clip(self.flows[0].normalizer.rtg(self.windows.rtg[idx]), -1.0, 1.0))
            horizon = min(self.h, self.window)
            discounts = (self.gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32)).view(1, -1)
            generated = self.generated_target(s_raw, a, rtg, horizon, discounts)
            real = self.h_step_target(future[:, :, :-1], future[:, :, -1], horizon, discounts)
        q1_loss, q2_loss = self.critic_step(torch.cat([s, s]), torch.cat([a, a]), torch.cat([real, generated]))
        self.soft_update()
        return {"value_loss": value_loss.item(), "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
                "policy_loss": policy_loss.item(), "adv_mean": adv.mean().item(),
                "fwm_real_target": real.mean().item(), "fwm_generated_target": generated.mean().item()}

    def update(self, batch) -> dict:
        self.steps += 1
        if self.mechanism and self.robust_every > 0 and self.steps % self.robust_every == 0:
            return self._fm_update()
        return super().update(batch)
