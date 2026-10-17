"""RSC-SAC: SAC trained on replay batches that a learned causal model partly rewrites.

RSC (Ding et al., "Seeing is not Believing: Robust Reinforcement Learning against
Spurious Correlation", NeurIPS 2023) trains an agent on counterfactual transitions so
that it does not rely on correlations between state variables that the task does not
cause. A causal model -- a dynamics model whose inputs reach each output only along the
edges of a learned graph -- is fitted on the replay buffer next to the agent. After a
warm-up, a fraction of every replay batch is intervened on:

1. the state variables the graph does not connect to the reward are the candidates;
   one of them (or a given fraction of them) is selected;
2. each intervened transition takes the selected variables from the reference
   transition that differs from it most in those variables and least in all others;
3. the causal model predicts the next state and the reward of the intervened state.

SAC then updates on the mixed batch. For vector observations the variables are the
observation entries, and actor and critic both learn from the rewritten states. For
images the variables are the entries of the critic's encoder output; the critic learns
from the rewritten encodings and the actor from the clean images.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import polyak_update


def _mlp(n_in: int, n_out: int, hidden: int, layers: int) -> nn.Sequential:
    mods, width = [], n_in
    for _ in range(layers + 1):
        mods += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    return nn.Sequential(*mods, nn.Linear(width, n_out))


class CausalModel(nn.Module):
    """Next state and reward from (state, action), through a learned graph.

    Every input variable is encoded on its own (its value and a learned index
    embedding). For each output (a next-state variable or the reward) the graph's mask
    sums the encodings of that output's parents, and a shared decoder reads the output
    from the sum and the output's index embedding. Training draws the mask from the edge
    probabilities (straight-through Gumbel); prediction keeps the edges with probability
    above one half. A sparsity penalty prunes edges. Inputs and targets are standardised
    with running moments.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, layers: int = 3,
                 width: int = 32, embed: int = 10, mask_init: float = 3.0,
                 sparse_weight: float = 0.01, sparse_norm: float = 0.1):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_in = self.state_dim + int(action_dim)
        self.n_out = self.state_dim + 1
        self.in_embed = nn.Embedding(self.n_in, embed)
        self.out_embed = nn.Embedding(self.n_out, embed)
        self.encoder = _mlp(embed + 1, width, hidden, layers)
        self.decoder = _mlp(width + embed, 1, hidden, layers)
        self.logits = nn.Parameter(torch.full((self.n_in, self.n_out), float(mask_init)))
        self.sparse_weight = float(sparse_weight)
        self.sparse_norm = float(sparse_norm)
        n = self.n_in + self.n_out
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.zeros(()))

    def edge_prob(self) -> torch.Tensor:
        """Probability of every edge, inputs (state, action) x outputs (next state, reward)."""
        return torch.sigmoid(self.logits)

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def _update_moments(self, x: torch.Tensor) -> None:
        self.count += 1
        rate = max(1.0 / float(self.count), 1e-3)
        self.mean.lerp_(x.mean(0), rate)
        self.var.lerp_(x.var(0, unbiased=False), rate)

    def _forward(self, s: torch.Tensor, a: torch.Tensor, sample: bool) -> torch.Tensor:
        x = (torch.cat([s, a], 1) - self.mean[: self.n_in]) / self.std[: self.n_in]
        p = self.edge_prob()
        if sample:
            two = torch.stack([1.0 - p, p], -1).clamp_min(1e-6).log()
            mask = F.gumbel_softmax(two, tau=1.0, hard=True)[..., 1]
        else:
            mask = (p > 0.5).float()
        b = x.shape[0]
        h = self.encoder(torch.cat([x.unsqueeze(-1), self.in_embed.weight.expand(b, -1, -1)], -1))
        h = torch.einsum("biw,io->bow", h, mask)
        return self.decoder(torch.cat([h, self.out_embed.weight.expand(b, -1, -1)], -1)).squeeze(-1)

    def loss(self, s, a, s_next, r) -> torch.Tensor:
        """Prediction error on one batch plus the sparsity penalty; updates the moments."""
        self._update_moments(torch.cat([s, a, s_next, r.reshape(-1, 1)], 1))
        target = (torch.cat([s_next, r.reshape(-1, 1)], 1) - self.mean[self.n_in:]) / self.std[self.n_in:]
        fit = F.mse_loss(self._forward(s, a, sample=True), target)
        return fit + self.sparse_weight * self.edge_prob().pow(self.sparse_norm).mean()

    @torch.no_grad()
    def predict(self, s, a) -> tuple[torch.Tensor, torch.Tensor]:
        y = self._forward(s, a, sample=False) * self.std[self.n_in:] + self.mean[self.n_in:]
        return y[:, : self.state_dim], y[:, self.state_dim]


class RSCSAC(SAC):
    """Stable-Baselines3 SAC with RSC's counterfactual replay.

    ``rsc`` holds the RSC settings (see ``robustrllib/configs/algorithm/rsc.yaml``):
    ``warmup``, ``ramp``, ``cooldown``, ``mix``, ``selected_fraction``,
    ``reference_size``, ``protect_last``, ``dynamics_lr``, ``dynamics`` (the causal
    model's sizes), and the learning rates ``actor_lr``, ``critic_lr``, ``lr_final``.
    """

    def __init__(self, *args, rsc: dict, **kwargs):
        self.rsc = dict(rsc)
        kwargs.setdefault("learning_rate", float(self.rsc["actor_lr"]))
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        state_dim = int(self.critic.features_extractor.features_dim)
        self.images = len(self.observation_space.shape) == 3
        self.causal_model = CausalModel(state_dim, get_action_dim(self.action_space),
                                        **(self.rsc.get("dynamics") or {})).to(self.device)
        self.model_optimizer = torch.optim.Adam(self.causal_model.parameters(),
                                                lr=float(self.rsc["dynamics_lr"]))
        self._rng = np.random.default_rng(self.seed)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["causal_model", "model_optimizer", "_rng"]

    # -- schedules -------------------------------------------------------------------
    def _lr(self, start: float) -> float:
        final = self.rsc.get("lr_final")
        if final is None:
            return float(start)
        return float(final) + (float(start) - float(final)) * self._current_progress_remaining

    def _mix(self) -> float:
        """Fraction of the batch intervened on: zero during the warm-up, then ``mix``,
        rising linearly over ``ramp`` steps and falling over the last ``cooldown`` steps."""
        c, t = self.rsc, self.num_timesteps
        if t < int(c["warmup"]):
            return 0.0
        mix = float(c["mix"])
        if c.get("ramp"):
            mix *= min(1.0, (t - int(c["warmup"])) / float(c["ramp"]))
        if c.get("cooldown"):
            mix *= min(1.0, max(0.0, (self._total_timesteps - t) / float(c["cooldown"])))
        return mix

    # -- the intervention ------------------------------------------------------------
    def _features(self, critic, obs) -> torch.Tensor:
        return critic.extract_features(obs, critic.features_extractor)

    def _selected_variables(self, device) -> torch.Tensor:
        eligible = self.causal_model.state_dim - int(self.rsc.get("protect_last") or 0)
        to_reward = self.causal_model.edge_prob()[:eligible, -1]
        candidates = torch.nonzero(to_reward < 0.5).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(eligible, device=device)
        frac = self.rsc.get("selected_fraction")
        k = 1 if not frac else max(1, int(round(float(frac) * candidates.numel())))
        pick = self._rng.choice(candidates.numel(), size=min(k, candidates.numel()), replace=False)
        return candidates[torch.as_tensor(pick, device=device)]

    def _intervene(self, s, a, s_next, r, n: int, reference):
        """Rewrite the first ``n`` transitions of the batch (s, a, s', r)."""
        model = self.causal_model
        sel = self._selected_variables(s.device)
        chosen = torch.zeros(model.n_in, dtype=torch.bool, device=s.device)
        chosen[sel] = True
        ref_s, ref_a = reference
        scale = model.std[: model.n_in]
        x = torch.cat([s[:n].detach(), a[:n]], 1) / scale
        xr = torch.cat([ref_s, ref_a], 1) / scale
        apart = torch.cdist(x[:, chosen], xr[:, chosen])
        close = torch.cdist(x[:, ~chosen], xr[:, ~chosen]) / float((~chosen).sum())
        donor = (apart - close).argmax(1)

        s = s.clone()
        s[:n, sel] = ref_s[donor][:, sel]
        cf_next, cf_reward = model.predict(s[:n].detach(), a[:n])
        s_next, r = s_next.clone(), r.clone()
        s_next[:n] = cf_next
        r[:n, 0] = cf_reward
        return s, s_next, r

    def _reference(self, s, a):
        size = self.rsc.get("reference_size")
        if not size:
            return s.detach(), a
        ref = self.replay_buffer.sample(int(size), env=self._vec_normalize_env)
        with torch.no_grad():
            return self._features(self.critic, ref.observations), ref.actions

    # -- one SAC update per gradient step, on the intervened batch -------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        actor_lr, critic_lr = self._lr(self.rsc["actor_lr"]), self._lr(self.rsc["critic_lr"])
        for opt, lr in ((self.actor.optimizer, actor_lr), (self.critic.optimizer, critic_lr),
                        (self.ent_coef_optimizer, actor_lr)):
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = lr
        mix = self._mix()
        logs = {"actor_loss": [], "critic_loss": [], "ent_coef": [], "model_loss": []}

        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            # the variables of the causal model: observation entries, or the critic's encoding
            s = self._features(self.critic, data.observations)
            with torch.no_grad():
                s_next = self._features(self.critic_target, data.next_observations)
            model_loss = self.causal_model.loss(s.detach(), data.actions, s_next, data.rewards)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()
            logs["model_loss"].append(model_loss.item())

            rewards = data.rewards
            n = int(round(mix * batch_size))
            if n > 0:
                s, s_next, rewards = self._intervene(s, data.actions, s_next, rewards, n,
                                                     self._reference(s, data.actions))
            actor_obs = data.observations if self.images else s.detach()
            next_actor_obs = data.next_observations if self.images else s_next

            actions_pi, log_prob = self.actor.action_log_prob(actor_obs)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            logs["ent_coef"].append(ent_coef.item())

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_actor_obs)
                next_q = torch.cat([q(torch.cat([s_next, next_actions], 1)) for q in self.critic_target.q_networks], 1)
                next_q = next_q.min(1, keepdim=True)[0] - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = rewards + (1 - data.dones) * discounts * next_q
            current_q = [q(torch.cat([s, data.actions], 1)) for q in self.critic.q_networks]
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            logs["critic_loss"].append(critic_loss.item())

            q_pi = torch.cat(self.critic(actor_obs, actions_pi), 1).min(1, keepdim=True)[0]
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            logs["actor_loss"].append(actor_loss.item())

            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate", actor_lr)
        self.logger.record("train/intervened_fraction", mix)
        reward_parents = int((self.causal_model.edge_prob()[: self.causal_model.state_dim, -1] > 0.5).sum())
        self.logger.record("train/reward_parents", reward_parents)
        for key, values in logs.items():
            self.logger.record(f"train/{key}", float(np.mean(values)))
