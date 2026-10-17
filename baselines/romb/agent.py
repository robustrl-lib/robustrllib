"""ROMB: IQL with an adversarially fine-tuned world model (robust model-based offline RL).

An ensemble Gaussian dynamics model is fitted to the dataset before training. Every
``robust_every``-th update then

1. rolls the current policy ``rollout_length`` steps through the model from dataset
   states (rows that terminate restart from new dataset states) and computes GAE over
   the rollout, giving a policy advantage (through V) and a dynamics advantage (through Q);
2. takes an IQL step whose critic regresses the real batch together with the model
   transitions, the latter weighted by ``model_target_ratio``;
3. applies the actor gradient ``grad(AWR loss) - actor_pg_coef * grad(J)``, ``J`` being
   the discounted, PPO-masked policy-gradient objective over the first
   ``second_order_horizon`` rollout steps (first-order Stackelberg update);
4. updates the model against the policy: a PPO-clipped step on the dynamics advantage,
   divided by ``sl_weight``, plus the supervised likelihood of real transitions.

All other updates are plain IQL.
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.iql.agent import IQLAgent
from baselines.romb.dynamics import GaussianDynamics, termination_fn


class ROMBAgent(IQLAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, *, env_id, robust_every=10,
                 n_ensemble=7, n_elites=5, dynamics_hidden_dims=(200, 200, 200, 200),
                 dynamics_weight_decay=(2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4), dynamics_lr=1e-3,
                 dynamics_max_epochs=30, dynamics_batch_size=256, rollout_length=5,
                 rollout_batch_size=256, gae_lambda=0.95, clip_range=0.2, sl_weight=3000.0,
                 actor_pg_coef=0.1, model_target_ratio=0.5, use_gradient_mask=True,
                 grad_clip_norm=1.0, second_order_horizon=10, **kwargs):
        super().__init__(state_dim, action_dim, max_action, **kwargs)
        self.robust_every = int(robust_every)
        self.dynamics = GaussianDynamics(
            state_dim, action_dim, termination_fn(env_id), hidden_dims=list(dynamics_hidden_dims),
            members=int(n_ensemble), elites=int(n_elites), weight_decays=list(dynamics_weight_decay),
            lr=float(dynamics_lr), device=self.device)
        self.dynamics_max_epochs, self.dynamics_batch_size = int(dynamics_max_epochs), int(dynamics_batch_size)
        self.rollout_length, self.rollout_batch_size = int(rollout_length), int(rollout_batch_size)
        self.gae_lambda, self.clip_range, self.sl_weight = float(gae_lambda), float(clip_range), float(sl_weight)
        self.actor_pg_coef, self.model_target_ratio = float(actor_pg_coef), float(model_target_ratio)
        self.use_gradient_mask, self.grad_clip_norm = bool(use_gradient_mask), float(grad_clip_norm)
        self.second_order_horizon = int(second_order_horizon)
        self.rng = np.random.default_rng(self.seed)
        self.data = None
        self.steps = 0

    def prepare(self, data) -> None:
        """Keep the dataset (rollout start states, supervised batches) and fit the model."""
        self.data = data
        info = self.dynamics.fit(data, self.dynamics_max_epochs, self.dynamics_batch_size, self.rng)
        print(f"[dynamics] {info}", flush=True)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def _rollout(self) -> dict:
        n = len(self.data)
        obs = self.data.states[self.rng.integers(0, n, size=self.rollout_batch_size)]
        keys = ("obs", "act", "rew", "next_obs", "done", "logp", "dyn_logp", "dyn_sample", "member", "v", "q")
        out = {k: [] for k in keys}
        for _ in range(self.rollout_length):
            obs_t = self.tensor(obs)
            s = self.normalize(obs_t)
            act, logp = self.policy.sample(s)
            nxt, rew, done, info = self.dynamics.step(obs, act.cpu().numpy(), self.rng)
            for key, value in (("obs", obs_t), ("act", act), ("rew", self.tensor(rew)),
                               ("next_obs", self.tensor(nxt)), ("done", self.tensor(done)),
                               ("logp", logp.squeeze(-1)), ("dyn_logp", self.tensor(info["log_prob"])),
                               ("dyn_sample", self.tensor(info["sample"])),
                               ("member", torch.as_tensor(info["member"], device=self.device)),
                               ("v", self.value(s).squeeze(-1)), ("q", self.min_q(s, act).squeeze(-1))):
                out[key].append(value)
            ended = done.astype(bool)
            if ended.any():
                nxt = nxt.copy()
                nxt[ended] = self.data.states[self.rng.integers(0, n, size=int(ended.sum()))]
            obs = nxt
        roll = {k: torch.stack(v) for k, v in out.items()}
        s_last = self.normalize(self.tensor(obs))
        a_last, _ = self.policy.sample(s_last)
        next_v, next_q = self.value(s_last).squeeze(-1), self.min_q(s_last, a_last).squeeze(-1)
        adv, dyn_adv = torch.zeros_like(roll["rew"]), torch.zeros_like(roll["rew"])
        gae = q_gae = torch.zeros_like(next_v)
        for t in reversed(range(self.rollout_length)):
            if t < self.rollout_length - 1:
                next_v, next_q = roll["v"][t + 1], roll["q"][t + 1]
            live = 1.0 - roll["done"][t]
            delta = roll["rew"][t] + self.gamma * next_v * live - roll["v"][t]
            q_delta = roll["rew"][t] + self.gamma * next_q * live - roll["q"][t]
            gae = delta + self.gamma * self.gae_lambda * live * gae
            q_gae = q_delta + self.gamma * self.gae_lambda * live * q_gae
            adv[t], dyn_adv[t] = gae, q_gae
        roll["adv"], roll["dyn_adv"] = adv, dyn_adv
        return roll

    # ------------------------------------------------------------- ROMB update
    def _flat_grad(self):
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                          for p in self.policy.parameters()]).clone()

    def _romb_update(self, batch) -> dict:
        roll = self._rollout()
        s, a, r, s2, d = self.unpack(batch)
        value_loss = self.value_step(s, a)

        with torch.no_grad():
            target = r + (1.0 - d) * self.gamma * self.value(s2)
            m_s = self.normalize(roll["obs"].reshape(-1, self.state_dim))
            m_a = roll["act"].reshape(-1, self.action_dim)
            m_t = (roll["rew"].reshape(-1, 1) + (1.0 - roll["done"].reshape(-1, 1)) * self.gamma
                   * self.value(self.normalize(roll["next_obs"].reshape(-1, self.state_dim))))
            c_s, c_a, c_t = torch.cat([s, m_s]), torch.cat([a, m_a]), torch.cat([target, m_t])
            c_w = torch.cat([torch.ones_like(target), torch.full_like(m_t, self.model_target_ratio)])
        q1_loss, q2_loss = self.critic_step(c_s, c_a, c_t, c_w)

        weights, _ = self.awr_weights(s, a)
        bc_loss = (weights * -self.policy.log_prob(s, a)).mean()
        self.policy_opt.zero_grad(); bc_loss.backward()
        bc_grad = self._flat_grad()
        h = min(self.second_order_horizon, self.rollout_length)
        adv = roll["adv"][:h]
        adv = (adv - adv.mean(-1, keepdim=True)) / (adv.std(-1, unbiased=False, keepdim=True) + 1e-8)
        logp = self.policy.log_prob(self.normalize(roll["obs"][:h].reshape(-1, self.state_dim)),
                                    roll["act"][:h].reshape(-1, self.action_dim)).reshape(h, -1)
        ratio = torch.exp(logp - roll["logp"][:h])
        mask = torch.ones_like(adv)
        if self.use_gradient_mask:
            clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            mask = (ratio * adv <= clipped * adv).float().detach()
        discount = (self.gamma ** torch.arange(h, device=self.device, dtype=torch.float32)).view(h, 1)
        objective = (mask * adv * logp * discount).sum(0).mean()
        self.policy.zero_grad(); objective.backward()
        total = bc_grad - self.actor_pg_coef * self._flat_grad()
        self.policy_opt.zero_grad()
        offset = 0
        for p in self.policy.parameters():
            p.grad = total[offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        self.policy_opt.step()
        self.soft_update()
        return {"value_loss": value_loss.item(), "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
                "policy_loss": bc_loss.item(), "romb_mask": mask.mean().item(), **self._model_step(roll)}

    def _model_step(self, roll) -> dict:
        obs = roll["obs"].reshape(-1, self.state_dim)
        act = roll["act"].reshape(-1, self.action_dim)
        sample = roll["dyn_sample"].reshape(-1, roll["dyn_sample"].shape[-1])
        dyn_adv = roll["dyn_adv"].reshape(-1)
        dyn_adv = (dyn_adv - dyn_adv.mean()) / (dyn_adv.std(unbiased=False) + 1e-8)
        ratio = torch.exp(self.dynamics.log_prob(obs, act, sample, roll["member"].reshape(-1))
                          - roll["dyn_logp"].reshape(-1))
        adv_loss = torch.min(ratio * dyn_adv,
                             torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * dyn_adv).mean()
        i = self.rng.integers(0, len(self.data), size=self.dynamics_batch_size)
        sl_loss = self.dynamics.supervised_loss(self.tensor(self.data.states[i]), self.tensor(self.data.actions[i]),
                                                self.tensor(self.data.next_states[i]), self.tensor(self.data.rewards[i]))
        loss = adv_loss / max(self.sl_weight, 1e-6) + sl_loss
        self.dynamics.optim.zero_grad(); loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.dynamics.model.parameters(), self.grad_clip_norm)
        self.dynamics.optim.step()
        return {"model_adv_loss": adv_loss.item(), "model_sl_loss": sl_loss.item()}

    def update(self, batch) -> dict:
        self.steps += 1
        if self.robust_every > 0 and self.steps % self.robust_every == 0:
            return self._romb_update(batch)
        return super().update(batch)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "dynamics": self.dynamics.model.state_dict()}
