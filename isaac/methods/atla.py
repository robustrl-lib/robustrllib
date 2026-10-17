"""ATLA-PPO (Zhang et al., 2021): PPO trained against a learned observation adversary.

Each outer iteration has two phases with one rollout each: the protagonist collects a rollout
under the current adversary and takes a PPO update; then the adversary, a Gaussian policy over
bounded observation perturbations ``epsilon * scale * tanh(u)``, collects a rollout against the
fixed protagonist and takes a PPO update on the negative task reward. The adversary uses the
protagonist's discount, GAE lambda, epochs and minibatches.
"""
from __future__ import annotations

import math
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(input_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, output_dim))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, gain=0.01)
    return net


class ATLARunner(OnPolicyRunner):
    """RSL-RL runner that alternates protagonist and adversary updates.

    Args:
        epsilon: perturbation bound, in units of ``attack_scale``.
        attack_scale: per-dimension scale of the perturbation (``None``: 1 for every dimension).
        timeout_value: on a timeout the adversary's reward is bootstrapped with its critic's value
            of the next observation (``"next"``) or of the current one (``"current"``).
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", *,
                 epsilon: float = 0.05, learning_rate: float = 1e-3, clip_param: float = 0.2,
                 entropy_coef: float = 0.0, value_loss_coef: float = 1.0, max_grad_norm: float = 1.0,
                 attack_scale: list[float] | None = None, timeout_value: str = "next"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        obs_dim = self.env.get_observations()["policy"].shape[-1]
        self.epsilon = epsilon
        self.attack_scale = torch.ones(obs_dim, device=self.device) if attack_scale is None \
            else torch.tensor(attack_scale, device=self.device)
        self.adv_actor = _mlp(obs_dim, obs_dim).to(self.device)
        self.adv_critic = _mlp(obs_dim, 1).to(self.device)
        self.adv_log_std = nn.Parameter(torch.zeros(obs_dim, device=self.device))
        self.adv_parameters = [*self.adv_actor.parameters(), *self.adv_critic.parameters(), self.adv_log_std]
        self.adv_optimizer = torch.optim.Adam(self.adv_parameters, lr=learning_rate, eps=1.0e-5)
        self.adv_clip, self.adv_entropy_coef = clip_param, entropy_coef
        self.adv_value_loss_coef, self.adv_max_grad_norm = value_loss_coef, max_grad_norm
        self.timeout_value = timeout_value

    def _adv_dist(self, policy_obs):
        mean = self.adv_actor(policy_obs)
        return Normal(mean, self.adv_log_std.exp().expand_as(mean))

    def _attacked(self, obs, raw=None):
        attacked = obs.clone()
        if raw is None:
            raw = self._adv_dist(obs["policy"]).sample()
        attacked["policy"] = obs["policy"] + self.epsilon * self.attack_scale * torch.tanh(raw)
        return attacked

    def _adversary_rollout(self, obs):
        keys = ("states", "raw_actions", "log_probs", "values", "rewards", "dones")
        rollout = {key: [] for key in keys}
        for _ in range(self.cfg["num_steps_per_env"]):
            clean = obs["policy"]
            dist = self._adv_dist(clean)
            raw = dist.sample()
            actions = self.alg.get_policy()(self._attacked(obs, raw), stochastic_output=True)
            value = self.adv_critic(clean).squeeze(-1)
            next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            if self.cfg.get("check_for_nan", True):
                check_nan(next_obs, rewards, dones)
            next_obs, rewards, dones = next_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            timeouts = extras["time_outs"].to(self.device)
            bootstrap = self.adv_critic(next_obs["policy"]).squeeze(-1) if self.timeout_value == "next" else value
            adv_rewards = -rewards + self.alg.gamma * bootstrap * timeouts
            for key, item in zip(keys, (clean, raw, dist.log_prob(raw).sum(-1), value, adv_rewards, dones)):
                rollout[key].append(item.detach())
            self.logger.process_env_step(rewards, dones, extras, None)
            obs = next_obs
        rollout = {key: torch.stack(value) for key, value in rollout.items()}
        rollout["last_value"] = self.adv_critic(obs["policy"]).squeeze(-1).detach()
        return obs, rollout

    def _update_adversary(self, rollout) -> dict[str, float]:
        gamma, lam = self.alg.gamma, self.alg.lam
        rewards, dones, values = rollout["rewards"], rollout["dones"], rollout["values"]
        returns, advantage = torch.zeros_like(values), torch.zeros_like(values[0])
        for step in reversed(range(rewards.shape[0])):
            next_value = rollout["last_value"] if step == rewards.shape[0] - 1 else values[step + 1]
            not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + gamma * not_terminal * next_value - values[step]
            advantage = delta + gamma * lam * not_terminal * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        states, raw, old_log_probs = (rollout[k].flatten(0, 1) for k in ("states", "raw_actions", "log_probs"))
        old_values, returns, advantages = values.flatten(0, 1), returns.flatten(0, 1), advantages.flatten(0, 1)
        batch = states.shape[0]
        minibatch = max(1, batch // self.alg.num_mini_batches)
        totals, updates = dict(adversary_policy=0.0, adversary_value=0.0, adversary_entropy=0.0), 0
        for _ in range(self.alg.num_learning_epochs):
            indices = torch.randperm(batch, device=self.device)
            for start in range(0, batch, minibatch):
                idx = indices[start:start + minibatch]
                dist = self._adv_dist(states[idx])
                ratio = torch.exp(dist.log_prob(raw[idx]).sum(-1) - old_log_probs[idx])
                entropy = dist.entropy().sum(-1).mean()
                value = self.adv_critic(states[idx]).squeeze(-1)
                surrogate = torch.maximum(-advantages[idx] * ratio,
                                          -advantages[idx] * ratio.clamp(1.0 - self.adv_clip, 1.0 + self.adv_clip)).mean()
                clipped = old_values[idx] + (value - old_values[idx]).clamp(-self.adv_clip, self.adv_clip)
                value_loss = torch.maximum((value - returns[idx]).pow(2), (clipped - returns[idx]).pow(2)).mean()
                loss = surrogate + self.adv_value_loss_coef * value_loss - self.adv_entropy_coef * entropy
                self.adv_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.adv_parameters, self.adv_max_grad_norm)
                self.adv_optimizer.step()
                for key, item in zip(totals, (surrogate, value_loss, entropy)):
                    totals[key] += item.item()
                updates += 1
        return {key: value / updates for key, value in totals.items()}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.adv_actor.train()
        self.adv_critic.train()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Protagonist phase: one rollout under the fixed adversary, one PPO update.
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(self._attacked(obs))
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                self.alg.compute_returns(self._attacked(obs))
            collect_time = time.time() - start
            start = time.time()
            loss_dict = self.alg.update()
            # Adversary phase: one rollout against the fixed protagonist, one PPO update.
            with torch.inference_mode():
                obs, rollout = self._adversary_rollout(obs)
            loss_dict.update(self._update_adversary(rollout))
            self.current_learning_iteration = it
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=collect_time,
                            learn_time=time.time() - start, loss_dict=loss_dict,
                            learning_rate=self.alg.learning_rate, action_std=self.alg.get_policy().output_std,
                            rnd_weight=None)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        saved = self.alg.save()
        saved.update(adversary_actor_state_dict=self.adv_actor.state_dict(),
                     adversary_critic_state_dict=self.adv_critic.state_dict(),
                     adversary_log_std=self.adv_log_std.detach().cpu(),
                     adversary_optimizer_state_dict=self.adv_optimizer.state_dict(),
                     iter=self.current_learning_iteration, infos=infos)
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)
