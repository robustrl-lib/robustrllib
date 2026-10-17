"""SPiDR-PPO (As et al., 2026): constrained PPO under domain randomisation whose cost is the
task's safety cost plus a penalty on the disagreement of one-step next states across a
randomised dynamics ensemble.

PrimalDualPPO
    Torch port of the PPO solver of the SPiDR reference implementation (safe-learning,
    ss2r/algorithms/ppo): generalised advantage estimation over fixed-length sequences with
    the targets recomputed by the current critics on every minibatch, a separate cost critic,
    the reward surrogate plus multiplier x cost surrogate, and a projected multiplier step after
    every minibatch. With ``constrained=False`` the cost terms are inactive and it is the same PPO;
    every Drawer method uses it that way as its base learner.

EnsembleVecEnv
    RSL-RL view of the behaviour environments of a scene laid out as
    [behaviour | member 1 | ... | member K]. Before every step each member is reset to its
    behaviour environment's state, so after the step the variance of the members' next states
    measures how much the randomised dynamics disagree about that transition. It also applies
    per-environment action delays and periodic velocity pushes of the training envelope.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import nn


# --------------------------------------------------------------------------- solver
def discounted_budget(episode_budget: float, episode_length: int, gamma: float) -> float:
    """Per-step discounted value of an undiscounted per-episode cost budget."""
    return (episode_budget / episode_length) / (1.0 - gamma)


def gae(truncation, termination, rewards, values, bootstrap, gamma, lam):
    """Time-major GAE of the reference implementation; truncated steps carry no advantage."""
    with torch.no_grad():
        mask = 1.0 - truncation
        next_values = torch.cat((values[1:], bootstrap.unsqueeze(0)), dim=0)
        deltas = (rewards + gamma * (1.0 - termination) * next_values - values) * mask
        acc = torch.zeros_like(bootstrap)
        corrections = torch.empty_like(values)
        for step in reversed(range(values.shape[0])):
            acc = deltas[step] + gamma * (1.0 - termination[step]) * mask[step] * lam * acc
            corrections[step] = acc
        targets = values + corrections
        next_targets = torch.cat((targets[1:], bootstrap.unsqueeze(0)), dim=0)
        advantages = (rewards + gamma * (1.0 - termination) * next_targets - values) * mask
    return targets, advantages


def normalized_advantage(advantage):
    return (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)


def policy_loss(ratio, reward_adv, cost_adv, multiplier, epsilon):
    """Clipped reward surrogate plus multiplier x clipped cost surrogate."""
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    reward_loss = -torch.minimum(ratio * reward_adv, clipped * reward_adv).mean()
    cost_loss = -torch.minimum(ratio * -cost_adv, clipped * -cost_adv).mean()
    return reward_loss + multiplier.detach() * cost_loss, reward_loss, cost_loss


def multiplier_step(multiplier, expected_cost, budget, learning_rate):
    """Projected gradient step of the Lagrange multiplier."""
    with torch.no_grad():
        multiplier.add_(learning_rate * (expected_cost.detach() - budget)).clamp_(min=0.0)


class PrimalDualPPO(PPO):
    """PPO with a cost critic and a Lagrange multiplier (see the module docstring)."""

    def __init__(self, actor, critic, storage, *, constrained: bool, episode_length: int,
                 unroll_length: int = 20, cost_gamma: float = 0.9, cost_lambda: float = 0.95,
                 episode_budget: float = 20.0, dual_lr: float = 0.01, initial_multiplier: float = 0.01,
                 **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent or self.rnd or self.symmetry:
            raise ValueError("PrimalDualPPO supports feed-forward models without RND or symmetry")
        if storage.num_transitions_per_env % unroll_length:
            raise ValueError("steps per environment must be a multiple of unroll_length")
        # The cost critic has the reward critic's architecture and its own initialisation.
        self.cost_critic = copy.deepcopy(self.critic).to(self.device)
        for module in self.cost_critic.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.learning_rate)
        self.constrained = constrained
        self.unroll_length = unroll_length
        self.cost_gamma, self.cost_lambda = cost_gamma, cost_lambda
        self.budget = discounted_budget(episode_budget, episode_length, cost_gamma)
        self.dual_lr = dual_lr
        self.multiplier = torch.tensor(initial_multiplier if constrained else 0.0, device=self.device)
        shape = (storage.num_transitions_per_env, storage.num_envs, 1)
        self.rewards = torch.zeros(shape, device=self.device)
        self.truncations = torch.zeros(shape, device=self.device)
        self.costs = torch.zeros(shape, device=self.device)
        self._last_obs = None

    def process_env_step(self, obs, rewards, dones, extras):
        # Truncations are masked in the GAE, so rewards are stored without timeout bootstrapping;
        # the observation normalisers are updated once per rollout in update().
        index = self.storage.step
        self.rewards[index].copy_(rewards.view(-1, 1))
        self.truncations[index].copy_(extras["time_outs"].view(-1, 1))
        self.costs[index].copy_(extras["cost"].view(-1, 1))
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        for model in (self.actor, self.critic, self.cost_critic):
            model.reset(dones)

    def compute_returns(self, obs):
        self._last_obs = obs.clone()

    def update(self):
        st, length = self.storage, self.unroll_length
        blocks = st.num_transitions_per_env // length
        count = blocks * st.num_envs

        def sequences(x):
            return x.reshape(blocks, length, st.num_envs, *x.shape[2:]).transpose(0, 1).flatten(1, 2)

        obs = sequences(st.observations)
        bootstrap_obs = torch.cat((st.observations[length::length], self._last_obs.unsqueeze(0)), dim=0).flatten(0, 1)
        for model in (self.actor, self.critic, self.cost_critic):
            model.update_normalization(st.observations.flatten(0, 1))
        actions = sequences(st.actions)
        old_logprob = sequences(st.actions_log_prob).squeeze(-1)
        rewards = sequences(self.rewards).squeeze(-1)
        costs = sequences(self.costs).squeeze(-1)
        truncation = sequences(self.truncations).squeeze(-1)
        termination = sequences(st.dones).squeeze(-1).float() * (1.0 - truncation)
        sums = dict.fromkeys(("value", "cost_value", "surrogate", "cost_surrogate", "entropy", "expected_cost"), 0.0)
        updates = 0
        for _ in range(self.num_learning_epochs):
            for idx in torch.randperm(count, device=self.device).chunk(self.num_mini_batches):
                batch_obs = obs[:, idx].flatten(0, 1)
                values = self.critic(batch_obs).reshape(length, -1)
                with torch.no_grad():
                    bootstrap = self.critic(bootstrap_obs[idx]).squeeze(-1)
                targets, advantage = gae(truncation[:, idx], termination[:, idx], rewards[:, idx],
                                         values.detach(), bootstrap, self.gamma, self.lam)
                value_loss = 0.25 * (targets - values).square().mean()
                cost_loss = torch.zeros((), device=self.device)
                cost_advantage = torch.zeros_like(advantage)
                expected_cost = torch.zeros((), device=self.device)
                if self.constrained:
                    cost_values = self.cost_critic(batch_obs).reshape(length, -1)
                    with torch.no_grad():
                        cost_bootstrap = self.cost_critic(bootstrap_obs[idx]).squeeze(-1)
                    cost_targets, cost_advantage = gae(truncation[:, idx], termination[:, idx], costs[:, idx],
                                                       cost_values.detach(), cost_bootstrap,
                                                       self.cost_gamma, self.cost_lambda)
                    cost_loss = 0.25 * (cost_targets - cost_values).square().mean()
                    expected_cost = cost_targets.mean()
                    cost_advantage = normalized_advantage(cost_advantage)
                self.actor(batch_obs, stochastic_output=True)
                logprob = self.actor.get_output_log_prob(actions[:, idx].flatten(0, 1)).reshape(length, -1)
                ratio = torch.exp(logprob - old_logprob[:, idx])
                surrogate, reward_surr, cost_surr = policy_loss(
                    ratio, normalized_advantage(advantage), cost_advantage, self.multiplier, self.clip_param)
                entropy = self.actor.output_entropy.mean()
                loss = surrogate + value_loss + cost_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                self.cost_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for model in (self.actor, self.critic, self.cost_critic):
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.constrained:
                    self.cost_optimizer.step()
                    multiplier_step(self.multiplier, expected_cost, self.budget, self.dual_lr)
                for key, value in zip(sums, (value_loss, cost_loss, reward_surr, cost_surr, entropy, expected_cost)):
                    sums[key] += value.detach().item()
                updates += 1
        st.clear()
        result = {key: value / updates for key, value in sums.items()}
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


# --------------------------------------------------------------------------- dynamics ensemble
def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def _repeat_behaviour(value: torch.Tensor, behavior_envs: int, ensemble_size: int) -> torch.Tensor:
    return value[:behavior_envs].repeat((ensemble_size,) + (1,) * (value.ndim - 1))


def _repeat_scene_state(state: dict[str, Any], behavior_envs: int, ensemble_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out[key] = _repeat_scene_state(value, behavior_envs, ensemble_size)
        elif torch.is_tensor(value):
            out[key] = _repeat_behaviour(value, behavior_envs, ensemble_size)
        else:
            out[key] = value
    return out


def _copy_to_members(tensor: torch.Tensor, behavior_envs: int, ensemble_size: int, batch_dim: int = 0) -> None:
    total = behavior_envs * (ensemble_size + 1)
    if tensor.ndim <= batch_dim or tensor.shape[batch_dim] != total:
        return
    source, target, repeats = [slice(None)] * tensor.ndim, [slice(None)] * tensor.ndim, [1] * tensor.ndim
    source[batch_dim], target[batch_dim], repeats[batch_dim] = slice(0, behavior_envs), slice(behavior_envs, total), ensemble_size
    tensor[tuple(target)].copy_(tensor[tuple(source)].repeat(tuple(repeats)))


def _copy_attributes_to_members(obj: Any, behavior_envs: int, ensemble_size: int) -> None:
    if obj is None or not hasattr(obj, "__dict__"):
        return
    for value in vars(obj).values():
        values = value.values() if isinstance(value, dict) else (value,)
        for item in values:
            if torch.is_tensor(item):
                _copy_to_members(item, behavior_envs, ensemble_size)


def restore_controller_targets(scene, behavior_envs: int, ensemble_size: int, member_ids) -> None:
    """Give every member its behaviour environment's joint position and velocity targets
    (resetting the physical state alone would leave the member's own targets in place)."""
    for asset in scene.articulations.values():
        for field, setter in (("joint_pos_target", "set_joint_position_target_index"),
                              ("joint_vel_target", "set_joint_velocity_target_index")):
            source = _repeat_behaviour(_torch(getattr(asset.data, field)), behavior_envs, ensemble_size)
            getattr(asset, setter)(target=source, env_ids=member_ids)


class EnsembleVecEnv(VecEnv):
    """Behaviour environments plus a one-step dynamics ensemble (see the module docstring).

    Args:
        env: the Isaac Lab environment with ``behavior_envs * (ensemble_size + 1)`` environments.
        measure: called on the raw environment just before resets; returns ``cost`` (safety cost
            per environment), ``state`` (next state per environment) and optional extra tensors
            that are passed on in ``extras`` for the behaviour environments.
        pessimism: weight of the ensemble disagreement added to the safety cost.
        delay_range: per-environment action delay, drawn once from ``randint(low, high)`` control
            steps; ``None`` for no delay.
        fixed_delay: delay every environment by this many steps instead.
        push_interval_s / push_speed: set the base velocity of every environment to ``push_speed``
            in a random horizontal direction each ``push_interval_s`` seconds of an episode.
    """

    def __init__(self, env, behavior_envs: int, ensemble_size: int, measure: Callable[[Any], dict],
                 pessimism: float = 0.0, delay_range: tuple[int, int] | None = None, fixed_delay: int | None = None,
                 push_interval_s: float | None = None, push_speed: float = 1.0, clip_actions: float | None = None,
                 seed: int = 0, disagreement_clip: float = 1000.0):
        self.env = env
        self.num_envs = self.behavior_envs = int(behavior_envs)
        self.ensemble_size = int(ensemble_size)
        self.measure, self.pessimism, self.disagreement_clip = measure, float(pessimism), disagreement_clip
        self.clip_actions = clip_actions
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        total = self.behavior_envs * (self.ensemble_size + 1)
        if self.unwrapped.num_envs != total:
            raise ValueError(f"the scene has {self.unwrapped.num_envs} environments, expected {total}")
        self._member_ids = torch.arange(self.behavior_envs, total, device=self.device)
        self._delay = torch.zeros(total, dtype=torch.long, device=self.device)
        if delay_range is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed) + 17041)
            self._delay = torch.randint(*delay_range, (total,), generator=generator, device=self.device)
        if fixed_delay is not None:
            self._delay.fill_(fixed_delay)
        self._history = torch.zeros(max(4, int(self._delay.max()) + 1), self.behavior_envs, self.num_actions,
                                    device=self.device)
        self._push_steps = None if push_interval_s is None else round(push_interval_s / self.unwrapped.step_dt)
        self.push_speed = push_speed
        self._measured: dict[str, torch.Tensor] = {}
        compute = self.unwrapped.reward_manager.compute

        def measure_then_compute(dt):
            # Runs inside env.step after physics and before the automatic reset.
            self._measured = {key: value.clone() for key, value in measure(self.unwrapped).items()}
            return compute(dt=dt)

        self.unwrapped.reward_manager.compute = measure_then_compute
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf[:self.behavior_envs]

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf[:self.behavior_envs].copy_(value)
        self.unwrapped.episode_length_buf[self.behavior_envs:].copy_(value.repeat(self.ensemble_size))

    def _behaviour(self, obs):
        return TensorDict({key: value[:self.behavior_envs] for key, value in obs.items()}, batch_size=[self.behavior_envs])

    def reset(self):
        obs, extras = self.env.reset()
        self._history.zero_()
        return self._behaviour(obs), extras

    def get_observations(self):
        return self._behaviour(self.unwrapped.observation_manager.compute(update_history=False))

    def _start_members_from_behaviour(self):
        if not self.ensemble_size:
            return
        n, k, raw = self.behavior_envs, self.ensemble_size, self.unwrapped
        state = _repeat_scene_state(raw.scene.get_state(is_relative=True), n, k)
        raw.scene.reset_to(state, env_ids=self._member_ids, is_relative=True)
        restore_controller_targets(raw.scene, n, k, self._member_ids)
        _copy_to_members(raw.episode_length_buf, n, k)
        _copy_to_members(raw.action_manager._action, n, k)
        _copy_to_members(raw.action_manager._prev_action, n, k)
        for term in (*raw.action_manager._terms.values(), *raw.command_manager._terms.values()):
            _copy_attributes_to_members(term, n, k)
        for timer in getattr(raw.event_manager, "_interval_term_time_left", []):
            _copy_to_members(timer, n, k)
        for group in raw.observation_manager._group_obs_term_history_buffer.values():
            for buffer in group.values():
                if buffer._buffer is not None:
                    _copy_to_members(buffer._buffer, n, k, batch_dim=1)
                _copy_to_members(buffer._num_pushes, n, k)
        for sensor in raw.scene.sensors.values():
            _copy_attributes_to_members(getattr(sensor, "data", None), n, k)

    def _push(self):
        if self._push_steps is None:
            return
        lengths = self.unwrapped.episode_length_buf[:self.behavior_envs]
        due = torch.nonzero((lengths > 0) & (lengths % self._push_steps == 0), as_tuple=False).squeeze(-1)
        if not due.numel():
            return
        ids = torch.cat([due + member * self.behavior_envs for member in range(self.ensemble_size + 1)])
        robot = self.unwrapped.scene["robot"]
        velocity = _torch(robot.data.root_vel_w)[ids].clone()
        angle = 2.0 * math.pi * torch.rand(ids.numel(), device=self.device)
        velocity[:, 0] = self.push_speed * torch.cos(angle)
        velocity[:, 1] = self.push_speed * torch.sin(angle)
        robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)

    def _delayed(self, actions):
        self._history = torch.roll(self._history, shifts=-1, dims=0)
        self._history[-1].copy_(actions)
        behaviour = torch.arange(self._delay.numel(), device=self.device) % self.behavior_envs
        return self._history[-1 - self._delay, behaviour]

    def step(self, actions):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        self._start_members_from_behaviour()
        self._push()
        obs, rewards, terminated, truncated, extras = self.env.step(self._delayed(actions))
        n, measured = self.behavior_envs, self._measured
        disagreement = torch.zeros(n, device=self.device)
        if self.ensemble_size:
            members = measured["state"][n:].reshape(self.ensemble_size, n, -1)
            disagreement = torch.nan_to_num(members.var(0, unbiased=False).mean(-1), nan=0.0,
                                            posinf=self.disagreement_clip).clamp(max=self.disagreement_clip)
        raw_cost = measured["cost"][:n]
        dones = (terminated[:n] | truncated[:n]).long()
        extras = dict(extras, time_outs=truncated[:n], raw_cost=raw_cost,
                      cost=raw_cost + self.pessimism * disagreement, disagreement=disagreement,
                      **{key: value[:n] for key, value in measured.items() if key not in ("cost", "state")})
        extras["log"] = {"Safety/raw_cost": raw_cost.mean(), "Safety/disagreement": disagreement.mean()}
        self._history[:, dones.bool()] = 0.0
        return self._behaviour(obs), rewards[:n], dones, extras

    def seed(self, seed=-1):
        return self.unwrapped.seed(seed)

    def close(self):
        return self.env.close()
