"""RAMU-PPO: the risk-averse model-uncertainty objective of RAMU (Queeney & Benosman, 2023)
applied to PPO's reward and cost value targets.

At every step RAMU draws ``n_samples`` synthetic next observations by scaling the observed change
of the kinematic observation terms by ``1 + U(-2 epsilon, 2 epsilon)``, evaluates the reward and cost
critics on them, and replaces the next-state value by a Wang distortion (``eta``) of the sampled
values: pessimistic for the reward, conservative for the cost. The one-step risk targets are then
traced along the rollout with lambda-returns. The policy loss is the clipped reward surrogate plus
multiplier x clipped cost surrogate, divided by ``1 + multiplier``; the multiplier follows an
exponential moving average of the expected cost once per rollout. Every optimiser step is checked
against the rollout policy (mean KL <= ``max_update_kl``, per-state KL <= ``max_sample_kl``) and
halved up to ``max_backtracks`` times; a step that still fails ends the update of that rollout.
"""
from __future__ import annotations

import copy
import math

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from isaac.methods.spidr import discounted_budget, normalized_advantage, policy_loss

# Observation terms whose change between steps is perturbed; the other reviewed terms are kept.
KINEMATIC_TERMS = {"base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel",
                   "cabinet_joint_pos", "cabinet_joint_vel", "rel_ee_drawer_distance"}
KEPT_TERMS = {"projected_gravity", "velocity_commands", "actions"}


def wang_weights(n: int, eta: float, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Probability weights of the Wang distortion over ``n`` sorted equally likely samples."""
    normal = torch.distributions.Normal(torch.tensor(0.0, device=device, dtype=dtype), 1.0)
    edges = torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    cdf = normal.cdf(normal.icdf(edges) + eta)
    cdf[0], cdf[-1] = 0.0, 1.0
    return cdf.diff()


def wang_value(samples: torch.Tensor, eta: float, *, cost: bool = False) -> torch.Tensor:
    """Wang-distorted expectation over the first dimension; weights the low rewards or high costs."""
    weights = wang_weights(samples.shape[0], eta, device=samples.device, dtype=samples.dtype)
    ordered = samples.sort(dim=0, descending=cost).values
    return (ordered * weights.reshape(-1, *([1] * (samples.ndim - 1)))).sum(0)


def sample_next(current, following, mask, n, epsilon, generator):
    noise = torch.rand((n, *current.shape), device=current.device, dtype=current.dtype,
                       generator=generator) * (4 * epsilon) - 2 * epsilon
    sampled = current + (following - current) * (1 + noise)
    return torch.where(mask, sampled, following.expand_as(sampled))


def td_target(signal, next_value, terminal, gamma):
    return signal + gamma * torch.where(terminal.bool(), 0.0, next_value)


def residual_trace(one_step_targets, values, dones, gamma, lam):
    """Lambda-weighted sum of one-step residuals; traces stop at every episode end."""
    advantage = torch.zeros_like(values[-1])
    result = torch.empty_like(values)
    for t in reversed(range(len(values))):
        advantage = one_step_targets[t] - values[t] + gamma * lam * (~dones[t].bool()) * advantage
        result[t] = advantage
    return result


def policy_kl(actor, observations, old_params):
    """KL from the rollout policy, computed from the parameters only (no sampling)."""
    with torch.no_grad():
        means = actor.mlp(actor.get_latent(observations))
        dist = actor.distribution
        std = dist.std_param if dist.std_type == "scalar" else dist.log_std_param.exp()
        if not torch.isfinite(means).all() or not torch.isfinite(std).all() or not (std > 0).all():
            return torch.full((len(observations),), float("inf"), device=means.device)
        return actor.get_kl_divergence(old_params, (means, std.detach().clone().expand_as(means)))


def _finite(tensors) -> bool:
    flags: dict[str, list[torch.Tensor]] = {}
    for t in tensors:
        if torch.is_tensor(t) and t.is_floating_point():
            flags.setdefault(str(t.device), []).append(torch.isfinite(t).all())
    return all(torch.stack(group).all().item() for group in flags.values())


def _state_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_tensors(item)


def guarded_step(optimizers, positive, check, max_backtracks: int) -> bool:
    """Take an optimiser step that keeps every parameter and optimiser state finite, ``positive``
    strictly positive and ``check()`` true; on failure retry at half the learning rate up to
    ``max_backtracks`` times, then restore the parameters and optimiser state and return False."""
    params = list({id(p): p for opt in optimizers for g in opt.param_groups for p in g["params"]}.values())
    if not _finite(p.grad for p in params if p.grad is not None):
        return False
    values = [p.detach().clone() for p in params]
    states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
    rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    def set_rates(scale):
        for opt, lrs in zip(optimizers, rates):
            for group, rate in zip(opt.param_groups, lrs):
                group["lr"] = rate * scale

    for trial in range(max_backtracks + 1):
        set_rates(0.5 ** trial)
        for opt in optimizers:
            opt.step()
        valid = (_finite(params) and all((p > 0).all().item() for p in positive)
                 and _finite(t for opt in optimizers for t in _state_tensors(opt.state)) and check())
        if valid:
            set_rates(1.0)
            return True
        with torch.no_grad():
            for param, value in zip(params, values):
                param.copy_(value)
        for opt, state in zip(optimizers, states):
            opt.load_state_dict(copy.deepcopy(state))
    set_rates(1.0)
    return False


class RAMUPPO(PPO):
    """RAMU-PPO (see the module docstring). ``feature_masks`` marks, per observation group, the
    entries that are perturbed when sampling next observations."""

    def __init__(self, actor, critic, storage, *, feature_masks: dict, sampling_seed: int, episode_length: int,
                 n_samples: int = 5, epsilon: float = 0.1, eta: float = 0.75, cost_gamma: float = 0.9,
                 cost_lambda: float = 0.95, episode_budget: float = 20.0, dual_lr: float = 0.01,
                 initial_multiplier: float = 0.01, dual_ema_beta: float = 0.9, max_update_kl: float = 0.03,
                 max_sample_kl: float = 0.3, max_backtracks: int = 8, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        if actor.is_recurrent or critic.is_recurrent or self.rnd or self.symmetry:
            raise ValueError("RAMU-PPO supports feed-forward models without RND or symmetry")
        self.n, self.epsilon, self.eta = n_samples, epsilon, eta
        self.masks = {key: torch.tensor(value, device=self.device, dtype=torch.bool) for key, value in feature_masks.items()}
        self.generator = torch.Generator(device=self.device).manual_seed(int(sampling_seed))
        # The cost critic has the reward critic's architecture and its own initialisation.
        self.cost_critic = copy.deepcopy(critic).to(self.device)
        for module in self.cost_critic.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.learning_rate)
        self.cost_gamma, self.cost_lambda = cost_gamma, cost_lambda
        self.budget = discounted_budget(episode_budget, episode_length, cost_gamma)
        self.dual_lr, self.dual_ema_beta = dual_lr, dual_ema_beta
        self.multiplier = torch.tensor(float(initial_multiplier), device=self.device)
        self.cost_ema = None
        self.max_update_kl, self.max_sample_kl, self.max_backtracks = max_update_kl, max_sample_kl, max_backtracks
        shape = (storage.num_transitions_per_env, storage.num_envs, 1)
        self.costs, self.cost_values, self.cost_returns, self.cost_advantages = (
            torch.zeros(shape, device=self.device) for _ in range(4))
        self._current = self._cost_value = None

    def act(self, obs):
        self._current = obs.clone()
        self._cost_value = self.cost_critic(obs).detach()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        index = self.storage.step
        following = extras["next_observation"].to(self.device)
        terminal = extras["terminal"].to(self.device).view(-1, 1)
        sampled = TensorDict({key: sample_next(self._current[key], following[key], mask, self.n, self.epsilon,
                                               self.generator) for key, mask in self.masks.items()},
                             batch_size=[self.n, self.storage.num_envs]).flatten(0, 1)
        reward_samples = self.critic(sampled).reshape(self.n, self.storage.num_envs, 1)
        # The cost lies in [0, 1], so its discounted value lies in [0, 1 / (1 - cost_gamma)].
        cost_samples = self.cost_critic(sampled).reshape(self.n, self.storage.num_envs, 1)
        cost_samples = cost_samples.clamp(0.0, 1.0 / (1.0 - self.cost_gamma))
        cost = extras["raw_cost"].to(self.device).view(-1, 1)
        self.storage.returns[index].copy_(td_target(rewards.view(-1, 1), wang_value(reward_samples, self.eta),
                                                    terminal, self.gamma))
        self.cost_returns[index].copy_(td_target(cost, wang_value(cost_samples, self.eta, cost=True),
                                                 terminal, self.cost_gamma))
        self.costs[index].copy_(cost)
        self.cost_values[index].copy_(self._cost_value)
        # The targets above bootstrap terminal and timeout steps explicitly, so the rewards are
        # stored as observed (PPO.process_env_step would add its own timeout bootstrap).
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        for model in (self.actor, self.critic, self.cost_critic):
            model.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        reward_advantage = residual_trace(st.returns, st.values, st.dones, self.gamma, self.lam)
        cost_advantage = residual_trace(self.cost_returns, self.cost_values, st.dones, self.cost_gamma, self.cost_lambda)
        st.returns.copy_(st.values + reward_advantage)
        self.cost_returns.copy_((self.cost_values + cost_advantage).clamp(0.0, 1.0 / (1.0 - self.cost_gamma)))
        st.advantages.copy_(normalized_advantage(st.returns - st.values))
        self.cost_advantages.copy_(normalized_advantage(self.cost_returns - self.cost_values))
        expected = self.cost_returns.mean().detach()
        self.cost_ema = expected.clone() if self.cost_ema is None else (
            self.dual_ema_beta * self.cost_ema + (1 - self.dual_ema_beta) * expected)

    def _within_kl(self, obs, old_params) -> bool:
        kl = policy_kl(self.actor, obs, old_params)
        return bool(torch.isfinite(kl).all()) and kl.mean().item() <= self.max_update_kl \
            and kl.max().item() <= self.max_sample_kl

    def update(self):
        st = self.storage
        obs = st.observations.flatten(0, 1)
        actions, old_log = st.actions.flatten(0, 1), st.actions_log_prob.flatten().detach()
        old_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
        returns, values = st.returns.flatten(0, 1), st.values.flatten(0, 1)
        cost_returns, cost_values = self.cost_returns.flatten(0, 1), self.cost_values.flatten(0, 1)
        adv, cadv = st.advantages.flatten(), self.cost_advantages.flatten()
        totals = dict.fromkeys(("value", "cost_value", "surrogate", "cost_surrogate", "entropy"), 0.0)
        accepted, stop = 0, False

        def value_loss(prediction, previous, target):
            loss = (prediction - target).square()
            if self.use_clipped_value_loss:
                clipped = previous + (prediction - previous).clamp(-self.clip_param, self.clip_param)
                loss = torch.maximum(loss, (clipped - target).square())
            return loss.mean()

        for _ in range(self.num_learning_epochs):
            for idx in torch.randperm(len(obs), device=self.device).chunk(self.num_mini_batches):
                batch = obs[idx]
                self.actor(batch, stochastic_output=True)
                log_ratio = self.actor.get_output_log_prob(actions[idx]) - old_log[idx]
                if not torch.isfinite(log_ratio).all() or (log_ratio.abs() > 60).any():
                    stop = True
                    break
                surrogate, reward_loss, cost_loss = policy_loss(log_ratio.exp(), adv[idx], cadv[idx],
                                                                self.multiplier, self.clip_param)
                surrogate = surrogate / (1 + self.multiplier.detach())
                entropy = self.actor.output_entropy.mean()
                if self.schedule == "adaptive" and self.desired_kl is not None:
                    with torch.no_grad():
                        kl = self.actor.get_kl_divergence(tuple(p[idx] for p in old_params),
                                                          self.actor.output_distribution_params).mean()
                        if kl > 2 * self.desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl < self.desired_kl / 2:
                            self.learning_rate = min(1e-3, self.learning_rate * 1.5)
                        for optimizer in (self.optimizer, self.cost_optimizer):
                            for group in optimizer.param_groups:
                                group["lr"] = self.learning_rate
                rloss = value_loss(self.critic(batch), values[idx], returns[idx])
                closs = value_loss(self.cost_critic(batch), cost_values[idx], cost_returns[idx])
                loss = surrogate + self.value_loss_coef * rloss + closs - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                self.cost_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for model in (self.actor, self.critic, self.cost_critic):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                positive = [self.actor.distribution.std_param] if self.actor.distribution.std_type == "scalar" else []
                if not guarded_step([self.optimizer, self.cost_optimizer], positive,
                                    lambda: self._within_kl(obs, old_params), self.max_backtracks):
                    stop = True
                    break
                accepted += 1
                for key, value in zip(totals, (rloss, closs, reward_loss, cost_loss, entropy)):
                    totals[key] += value.detach().item()
            if stop:
                break
        if accepted:
            with torch.no_grad():
                self.multiplier.add_(self.dual_lr * (self.cost_ema - self.budget)).clamp_(min=0.0)
        st.clear()
        result = {key: value / max(accepted, 1) for key, value in totals.items()}
        result["multiplier"] = self.multiplier.item()
        return result

    def train_mode(self):
        super().train_mode()
        self.cost_critic.train()

    def eval_mode(self):
        super().eval_mode()
        self.cost_critic.eval()

    def save(self):
        output = super().save()
        output.update(cost_critic_state_dict=self.cost_critic.state_dict(),
                      cost_optimizer_state_dict=self.cost_optimizer.state_dict(),
                      multiplier=self.multiplier.detach().cpu())
        return output


def feature_mask(names, dimensions) -> list[bool]:
    mask = []
    for name, dimension in zip(names, dimensions):
        if name not in KINEMATIC_TERMS | KEPT_TERMS:
            raise ValueError(f"observation term {name!r} has no RAMU perturbation rule")
        size = math.prod(dimension) if isinstance(dimension, (tuple, list)) else int(dimension)
        mask += [name in KINEMATIC_TERMS] * size
    return mask


class RAMUVecEnv(RslRlVecEnvWrapper):
    """Isaac Lab wrapper that also reports, for every step, the next observation before any
    automatic reset, whether the step ended an episode, and the safety cost."""

    def __init__(self, env, cost_fn, clip_actions=None):
        super().__init__(env, clip_actions)
        raw = self.unwrapped
        manager = raw.observation_manager
        self.feature_masks = {group: feature_mask(names, manager.group_obs_term_dim[group])
                              for group, names in manager.active_terms.items()}
        compute = raw.reward_manager.compute

        def compute_and_capture(*args, **kwargs):
            reward = compute(*args, **kwargs)
            self._next = TensorDict(raw.observation_manager.compute(update_history=False),
                                    batch_size=[self.num_envs]).clone()
            self._cost = cost_fn(raw).clone()
            self._terminal = raw.reset_terminated.clone()
            if raw.cfg.is_finite_horizon:
                self._terminal |= raw.reset_time_outs
            return reward

        raw.reward_manager.compute = compute_and_capture

    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        following = obs.clone()
        ended = done.bool()
        for key in following.keys():
            following[key][ended] = self._next[key][ended]
        extras.update(next_observation=following, terminal=self._terminal, raw_cost=self._cost)
        return obs, reward, done, extras
