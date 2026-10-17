"""PLR-PVL: FWM with an ensemble of flow models and a positive-value-loss pick per sample.

The ``n_flow_models`` flow models (same data, different training seeds) each generate an
h-step window from the same (s, a) and return-to-go. Each window is scored by its positive
value loss, the prioritized-level-replay criterion

    PVL = (1/H) sum_t max(sum_{k>=t} (gamma * pvl_lam)^(k-t) delta_k, 0),
    delta_k = r_k + gamma V(s_{k+1}) - V(s_k)     (s_0 is the real state),

and the critic regresses, per sample, the generated target of the window with the largest
PVL: the model whose future the current critic predicts worst. Everything else is FWM.
"""
from __future__ import annotations

import torch

from baselines.fwm.agent import FWMAgent


class PLRPVLAgent(FWMAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, *, pvl_lam=0.95, **kwargs):
        super().__init__(state_dim, action_dim, max_action, **kwargs)
        self.pvl_lam = float(pvl_lam)

    def _pvl(self, s0, states, rewards):
        horizon = min(self.h, rewards.shape[1])
        traj = torch.cat([s0.unsqueeze(1), states[:, 1:horizon + 1]], dim=1)
        b = s0.shape[0]
        v = self.value(self.normalize(traj.reshape(b * (horizon + 1), -1))).reshape(b, horizon + 1)
        delta = rewards[:, :horizon] + self.gamma * v[:, 1:] - v[:, :-1]
        gae = torch.zeros_like(delta)
        gae[:, -1] = delta[:, -1]
        for t in range(horizon - 2, -1, -1):
            gae[:, t] = delta[:, t] + self.gamma * self.pvl_lam * gae[:, t + 1]
        return gae.clamp(min=0).mean(dim=1)

    def generated_target(self, s_raw, a, rtg, horizon, discounts):
        targets, scores = [], []
        for flow in self.flows:
            states, rewards = flow.sample(torch.cat([s_raw, a], -1), rtg, self.fm_n_steps,
                                          self.fm_ode_solver, self.fm_guidance_weight)
            targets.append(self.h_step_target(states, rewards, horizon, discounts))
            scores.append(self._pvl(s_raw, states, rewards))
        best = torch.stack(scores).argmax(dim=0)                      # [B]
        return torch.stack(targets).gather(0, best.view(1, -1, 1)).squeeze(0)
