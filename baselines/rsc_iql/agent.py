"""RSC-IQL: IQL on batches whose spurious state features are re-sampled (Ding et al., 2024, offline form).

Every ``robust_every``-th update rewrites the batch before the IQL step: a fraction
``rewrite_ratio`` of its rows is selected and, for each observation dimension ``d`` the
causal mask marks as unimportant (normalised importance below ``prob_d_threshold``),
each selected row takes, with probability ``prob_d_scale * prob_d[d]``, the value of
dimension ``d`` from another selected row. Dimensions the learned dynamics depend on
are left alone. All other updates are plain IQL.
"""
from __future__ import annotations

import numpy as np

from baselines.iql.agent import IQLAgent
from baselines.offline.data import Batch


class RSCIQLAgent(IQLAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, *, mask_path=None, rewrite_ratio=0.25,
                 prob_d_threshold=0.7, prob_d_scale=0.3, robust_every=50, **kwargs):
        super().__init__(state_dim, action_dim, max_action, **kwargs)
        mask = np.load(mask_path)
        prob = np.where(mask["importance"] < float(prob_d_threshold), mask["prob_d"], 0.0)
        self.prob_d = (prob * float(prob_d_scale)).astype(np.float32)
        self.rewrite_ratio = float(rewrite_ratio)
        self.robust_every = int(robust_every)
        self.rng = np.random.default_rng(self.seed)
        self.steps = 0

    def rewrite(self, batch: Batch) -> Batch:
        obs = np.array(batch.states, dtype=np.float32, copy=True)
        n = obs.shape[0]
        k = int(round(self.rewrite_ratio * n))
        rows = np.sort(self.rng.choice(n, size=k, replace=False)) if 0 < k < n else np.arange(n if k else 0)
        for d in np.flatnonzero(self.prob_d > 0):
            select = self.rng.random(rows.size) < self.prob_d[d]
            if select.any():
                perm = self.rng.permutation(rows.size)
                obs[rows[select], d] = obs[rows[perm[select]], d]
        return Batch(obs, batch.actions, batch.rewards, batch.next_states, batch.dones)

    def update(self, batch) -> dict:
        self.steps += 1
        if self.robust_every > 0 and self.steps % self.robust_every == 0:
            batch = self.rewrite(batch)
        return super().update(batch)
