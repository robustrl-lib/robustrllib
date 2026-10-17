"""ATLA-PPO and ATLA-SA-PPO for image observations (CarRacing), on Stable-Baselines3 PPO.

The observation adversary is a convolutional network that adds a perturbation
``delta(s) = eps * tanh(f(s))`` (in pixel levels) to every observation the
protagonist sees. After each PPO rollout it takes ``adv_adversary_steps`` Adam
steps that lower the protagonist's value of the perturbed observation,
``mean V(s + delta(s))``, over clean observations from the recent rollouts.
ATLA-SA-PPO adds the SA-PPO regulariser to the PPO loss: ``robust_ppo_reg`` times
the KL divergence between the action distributions at an observation and at the
worst observation within ``eps`` found by ``robust_ppo_pgd_steps`` PGD steps.

Hyperparameters (``hparams``): ``mode``, ``robust_ppo_eps`` (eps, pixel levels),
``ppo_lr_adam``, ``n_envs``, ``t`` (rollout steps per environment), ``batch_size``,
``ppo_epochs``, ``gamma``, ``lambda``, ``clip_eps``, ``entropy_coeff``, ``vf_coef``,
``max_grad_norm``, ``adv_ppo_lr_adam``, ``adv_adversary_steps``, ``adv_batch_size``,
and for ATLA-SA-PPO ``robust_ppo_reg`` and ``robust_ppo_pgd_steps``.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import explained_variance, get_device, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from robustrllib.tasks import make_env

CACHE_SIZE = 8192


class ObservationAdversary(nn.Module):
    """Convolutional encoder-decoder from an image stack to a perturbation of the same shape."""

    def __init__(self, channels, eps):
        super().__init__()
        self.eps = float(eps)
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1), nn.Tanh(),
        )
        # clean observations of the recent rollouts, the adversary's training data
        self.clean_cache = deque(maxlen=CACHE_SIZE)

    def forward(self, obs):
        delta = self.head(self.body(obs.float() / 255.0))
        if delta.shape[-2:] != obs.shape[-2:]:
            delta = F.interpolate(delta, size=tuple(obs.shape[-2:]), mode="bilinear", align_corners=False)
        return self.eps * delta


class AdversarialObservation(gym.ObservationWrapper):
    """The protagonist's view: every observation plus the adversary's perturbation."""

    def __init__(self, env, adversary, device):
        super().__init__(env)
        self.adversary = adversary
        self.device = device

    def observation(self, obs):
        frame = np.asarray(obs)
        self.adversary.clean_cache.append(frame.copy())
        with torch.no_grad():
            delta = self.adversary(torch.as_tensor(frame, device=self.device).unsqueeze(0))[0].cpu().numpy()
        return np.clip(frame.astype(np.float32) + delta, 0.0, 255.0).astype(frame.dtype, copy=False)


class AdversaryUpdate(BaseCallback):
    """After each rollout, ``updates`` Adam steps on ``mean V(s + delta(s))`` over cached clean observations."""

    def __init__(self, adversary, optimizer, batch_size, updates):
        super().__init__()
        self.adversary = adversary
        self.optimizer = optimizer
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.value = float("nan")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        cache = list(self.adversary.clean_cache)
        if len(cache) < 32:
            return
        self.adversary.train()
        for _ in range(self.updates):
            idx = np.random.choice(len(cache), size=min(self.batch_size, len(cache)), replace=False)
            obs = torch.as_tensor(np.stack([cache[i] for i in idx]), device=self.model.device).float()
            attacked = (obs + self.adversary(obs)).clamp(0.0, 255.0)
            loss = self.model.policy.predict_values(attacked).mean()
            grads = torch.autograd.grad(loss, self.adversary.parameters(), allow_unused=True)
            self.optimizer.zero_grad()
            for param, grad in zip(self.adversary.parameters(), grads):
                if grad is not None:
                    param.grad = grad
            nn.utils.clip_grad_norm_(self.adversary.parameters(), 1.0)
            self.optimizer.step()
        self.value = float(loss.detach())
        returns = [info["r"] for info in self.model.ep_info_buffer]
        mean_return = float(np.mean(returns)) if returns else float("nan")
        print(f"[atla] steps {self.num_timesteps}  return {mean_return:9.2f}  "
              f"V under the adversary {self.value:8.3f}", flush=True)


class SAPPO(PPO):
    """PPO with the SA-PPO regulariser; ``train`` is Stable-Baselines3's PPO update plus that term."""

    def __init__(self, *args, sa_eps=8.0, sa_steps=2, sa_reg=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sa_eps = float(sa_eps)
        self.sa_steps = int(sa_steps)
        self.sa_reg = float(sa_reg)
        self.sa_step_eps = self.sa_eps / max(self.sa_steps, 1)

    def _categorical_kl(self, obs_clean, obs_adv):
        probs = self.policy.get_distribution(obs_clean).distribution.probs.detach().clamp_min(1e-8)
        log_q = self.policy.get_distribution(obs_adv).distribution.logits.log_softmax(dim=-1)
        return (probs * (probs.log() - log_q)).sum(dim=-1).mean()

    def _pgd_obs(self, obs):
        """The observation within ``sa_eps`` (random start, signed-gradient steps) that maximises the KL."""
        obs = obs.float()
        delta = torch.empty_like(obs).uniform_(-self.sa_eps, self.sa_eps)
        for _ in range(max(self.sa_steps, 1)):
            delta = delta.detach().requires_grad_(True)
            kl = self._categorical_kl(obs.detach(), (obs + delta).clamp(0.0, 255.0))
            grad = torch.autograd.grad(kl, delta)[0]
            delta = (delta + self.sa_step_eps * grad.sign()).clamp(-self.sa_eps, self.sa_eps)
        return (obs + delta.detach()).clamp(0.0, 255.0)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, sa_losses, clip_fractions = [], [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                sa_loss = values.new_zeros(())
                if self.sa_reg > 0.0 and self.sa_steps > 0:
                    obs_adv = self._pgd_obs(rollout_data.observations)
                    sa_loss = self._categorical_kl(rollout_data.observations, obs_adv)
                    sa_losses.append(float(sa_loss.detach()))
                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                        + self.sa_reg * sa_loss)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(loss.detach()))
        self.logger.record("train/explained_variance", explained_var)
        if sa_losses:
            self.logger.record("train/sa_kl", np.mean(sa_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def train_pixel(task, shifts, hp, steps, seed, out: Path) -> Path:
    if int(hp.get("history_length", -1)) > 0:
        raise ValueError("the pixel trainer has a feed-forward policy; set history_length: -1")
    device = get_device("auto")
    set_random_seed(seed)
    probe = make_env(task, seed=seed)
    channels = int(probe.observation_space.shape[0])
    probe.close()
    adversary = ObservationAdversary(channels, eps=hp["robust_ppo_eps"]).to(device)

    def make_one(rank):
        return lambda: Monitor(AdversarialObservation(make_env(task, shifts=shifts, seed=seed + rank),
                                                      adversary, device))

    venv = DummyVecEnv([make_one(i) for i in range(int(hp["n_envs"]))])
    kwargs = dict(learning_rate=hp["ppo_lr_adam"], n_steps=int(hp["t"]), batch_size=int(hp["batch_size"]),
                  n_epochs=int(hp["ppo_epochs"]), gamma=hp["gamma"], gae_lambda=hp["lambda"],
                  clip_range=hp["clip_eps"], ent_coef=hp["entropy_coeff"], vf_coef=hp["vf_coef"],
                  max_grad_norm=hp["max_grad_norm"], seed=seed, device=device, verbose=0)
    if hp["mode"] == "adv_sa_ppo":
        model = SAPPO("CnnPolicy", venv, sa_eps=hp["robust_ppo_eps"], sa_steps=hp["robust_ppo_pgd_steps"],
                      sa_reg=hp["robust_ppo_reg"], **kwargs)
    else:
        model = PPO("CnnPolicy", venv, **kwargs)
    model.set_logger(configure(str(out), ["csv"]))
    update = AdversaryUpdate(adversary, torch.optim.Adam(adversary.parameters(), lr=hp["adv_ppo_lr_adam"]),
                             batch_size=hp["adv_batch_size"], updates=hp["adv_adversary_steps"])
    model.learn(total_timesteps=int(steps), callback=update)

    path = out / "ckpt" / f"step_{model.num_timesteps}.pt"
    torch.save({"backend": "pixel", "hparams": dict(hp), "steps": int(model.num_timesteps),
                "policy": model.policy.state_dict(), "adversary": adversary.state_dict()}, path)
    venv.close()
    return path
