"""The ATLA trainer for vector observations: a PPO protagonist and a PPO observation adversary,
trained in alternation (ATLA-PPO), with the LSTM policy and the SA-PPO regulariser (ATLA-SA-PPO).

Adapted from ATLA (https://github.com/huanzhang12/ATLA_robust_RL),
``src/policy_gradients/agent.py`` and ``custom_env.py``, keeping the ``adv_ppo`` and
``adv_sa_ppo`` modes. The adversary is a Gaussian policy from the (normalised)
observation to a perturbation ``ROBUST_PPO_EPS * hardtanh(a)``; it is rewarded with
the protagonist's negative reward. Each iteration collects one rollout of ``T``
steps for the protagonist (acting on perturbed observations) and one for the
adversary, and takes a PPO step on each.
"""
import random

import numpy as np
import torch
import torch as ch
import torch.optim as optim

from .models import CtsLSTMPolicy, CtsPolicy, ValueDenseNet, ValueLSTMNet
from .steps import LinearEpsSchedule, ppo_step, robust_ppo_step, value_step
from .torch_utils import (Identity, Parameters, RewardFilter, Trajectories, ZFilter, cpu_tensorize,
                          discount_path, get_path_indices, shape_equal_cmp)

MODES = ("adv_ppo", "adv_sa_ppo")


class Env:
    """A gymnasium environment with the trainer's observation and reward normalisers.

    ``reset() -> state`` and ``step(action) -> (state, reward, done, info)``, where
    ``done`` covers termination and truncation and ``info['done']`` holds
    ``(length, return)`` at the end of an episode. Every reset draws the
    environment seed from Python's ``random``.
    """

    def __init__(self, env, norm_states, norm_rewards, gamma, clip_obs, clip_rew):
        self.env = env
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew
        assert len(env.observation_space.shape) == 1, "vector observations only"
        assert len(env.action_space.shape) == 1, "continuous actions only"
        self.num_features = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], clip=clip_obs)
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=gamma, clip=clip_rew)
        self.total_true_reward = 0.0
        self.counter = 0
        self._read_only = False

    @property
    def normalizer_read_only(self):
        return self._read_only

    @normalizer_read_only.setter
    def normalizer_read_only(self, value):
        self._read_only = bool(value)
        for f in (self.state_filter, self.reward_filter):
            if hasattr(f, "read_only"):
                f.read_only = self._read_only

    def reset(self):
        obs, _ = self.env.reset(seed=random.getrandbits(31))
        self.total_true_reward = 0.0
        self.counter = 0
        self.state_filter.reset()
        self.reward_filter.reset()
        return self.state_filter(np.asarray(obs, dtype=np.float64), reset=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = bool(terminated or truncated)
        state = self.state_filter(np.asarray(obs, dtype=np.float64))
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info = dict(info)
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info

    def close(self):
        self.env.close()


class Trainer:
    """Alternating PPO training of the protagonist and the observation adversary.

    ``params`` holds the hyperparameters (keys as in the upstream configuration
    files); attributes are read case-insensitively from it (``self.T``).
    ``make_env()`` returns the gymnasium environment to train on.
    """

    def __init__(self, params, make_env):
        self.params = Parameters(params)
        assert self.MODE in MODES, f"mode must be one of {MODES}"
        self.params.NUM_ACTORS = 1
        self.envs = [Env(make_env(), norm_states=self.NORM_STATES, norm_rewards=self.NORM_REWARDS,
                         gamma=self.GAMMA, clip_obs=self.CLIP_OBSERVATIONS, clip_rew=self.CLIP_REWARDS)]
        self.params.NUM_ACTIONS = self.envs[0].num_actions
        self.params.NUM_FEATURES = self.envs[0].num_features
        recurrent = self.HISTORY_LENGTH > 0
        policy_net_class = CtsLSTMPolicy if recurrent else CtsPolicy
        value_net_class = ValueLSTMNet if recurrent else ValueDenseNet
        self.n_steps = 0

        self.policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS)
        if self.MODE == 'adv_sa_ppo':
            self.robust_eps_scheduler = LinearEpsSchedule(self.ROBUST_PPO_EPS,
                                                          self.ROBUST_PPO_EPS_SCHEDULER_OPTS)

        # the adversary maps a state to a perturbation of the same size
        self.adversary_policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES)
        self.params.ADV_POLICY_ADAM = optim.Adam(self.adversary_policy_model.parameters(),
                                                 lr=self.ADV_PPO_LR_ADAM, eps=1e-5)
        self.adversary_val_model = value_net_class(self.NUM_FEATURES)
        self.adversary_val_opt = optim.Adam(self.adversary_val_model.parameters(), lr=self.ADV_VAL_LR, eps=1e-5)
        if self.ANNEAL_LR:
            adv_lam = lambda f: 1 - f / self.TRAIN_STEPS
            self.params.ADV_POLICY_SCHEDULER = optim.lr_scheduler.LambdaLR(self.ADV_POLICY_ADAM, lr_lambda=adv_lam)
            self.params.ADV_VALUE_SCHEDULER = optim.lr_scheduler.LambdaLR(self.adversary_val_opt, lr_lambda=adv_lam)

        kwargs = {'lr': self.PPO_LR_ADAM}
        if self.params.ADAM_EPS > 0:
            kwargs['eps'] = self.ADAM_EPS
        self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(), **kwargs)
        self.val_model = value_net_class(self.NUM_FEATURES)
        self.val_opt = optim.Adam(self.val_model.parameters(), lr=self.VAL_LR, eps=1e-5)
        if self.ANNEAL_LR:
            lam = lambda f: 1 - f / self.TRAIN_STEPS
            self.params.POLICY_SCHEDULER = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, lr_lambda=lam)
            self.params.VALUE_SCHEDULER = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)

    def __getattr__(self, x):
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)

    def advantage_and_return(self, rewards, values, not_dones):
        """GAE advantages and discounted returns; ``V(s_{t+1}) = 0`` after a terminal step."""
        assert shape_equal_cmp(rewards, values, not_dones)
        V_s_tp1 = ch.cat([values[:, 1:], values[:, -1:]], 1) * not_dones
        deltas = rewards + self.GAMMA * V_s_tp1 - values
        advantages = ch.zeros_like(rewards)
        returns = ch.zeros_like(rewards)
        for agent, start, end in get_path_indices(not_dones):
            advantages[agent, start:end] = discount_path(deltas[agent, start:end], self.LAMBDA * self.GAMMA)
            returns[agent, start:end] = discount_path(rewards[agent, start:end], self.GAMMA)
        return advantages.clone().detach(), returns.clone().detach()

    def reset_envs(self, envs):
        return cpu_tensorize([env.reset() for env in envs]).unsqueeze(1)

    def multi_actor_step(self, actions, envs):
        normed_rewards, states, not_dones = [], [], []
        completed_episode_info = []
        for action, env in zip(actions, envs):
            new_state, normed_reward, is_done, info = env.step(action[0].cpu().numpy())
            if is_done:
                completed_episode_info.append(info['done'])
                new_state = env.reset()
            normed_rewards.append([normed_reward])
            not_dones.append([int(not is_done)])
            states.append([new_state])
        data = list(map(cpu_tensorize, [normed_rewards, states, not_dones]))
        return [completed_episode_info, *data]

    def run_trajectories(self, num_saps, collect_adversary_trajectory=False):
        """One rollout of ``num_saps`` steps with the adversary perturbing every observation.

        For the protagonist the trajectory holds the perturbed states it acted on;
        for the adversary it holds the true states, the perturbations as actions and
        the negated rewards, and the normalisers are frozen.
        """
        if collect_adversary_trajectory:
            old_env_read_only_flags = []
            for e in self.envs:
                old_env_read_only_flags.append(e.normalizer_read_only)
                e.normalizer_read_only = True

        envs = self.envs
        initial_states = self.reset_envs(envs)
        self.policy_model.reset()
        completed_episode_info = []
        traj_length = int(num_saps // self.NUM_ACTORS)
        shape = (self.NUM_ACTORS, traj_length)
        rewards, not_dones, action_log_probs = [ch.zeros(shape) for _ in range(3)]
        actions_shape = shape + ((self.NUM_FEATURES,) if collect_adversary_trajectory else (self.NUM_ACTIONS,))
        actions = ch.zeros(actions_shape)
        action_means = ch.zeros(actions_shape)
        states = ch.zeros((self.NUM_ACTORS, traj_length + 1) + initial_states.shape[2:])
        collect_perturbed_state = not collect_adversary_trajectory

        if collect_perturbed_state:
            # states are stored after the perturbation, which is not drawn yet
            last_states = initial_states.squeeze(1)
        else:
            states[:, 0, :] = initial_states
            last_states = states[:, 0, :]
        for t in range(traj_length):
            adv_perturbation_pds = self.adversary_policy_model(last_states)
            next_adv_perturbation_means, _ = adv_perturbation_pds
            next_adv_perturbations = self.adversary_policy_model.sample(adv_perturbation_pds)
            next_adv_perturbation_log_probs = self.adversary_policy_model.get_loglikelihood(
                adv_perturbation_pds, next_adv_perturbations)
            last_states = last_states + ch.nn.functional.hardtanh(next_adv_perturbations) * self.ROBUST_PPO_EPS
            next_adv_perturbations = next_adv_perturbations.unsqueeze(1)

            action_pds = self.policy_model(last_states)
            next_action_means, _ = action_pds
            next_actions = self.policy_model.sample(action_pds)
            next_action_log_probs = self.policy_model.get_loglikelihood(action_pds, next_actions).unsqueeze(1)
            next_actions = next_actions.unsqueeze(1)

            done_info, next_rewards, next_states, next_not_dones = self.multi_actor_step(next_actions, envs)
            if next_not_dones.item() == 0:
                self.policy_model.reset()
            if len(done_info) > 0 and (t != self.T - 1 or len(completed_episode_info) == 0):
                completed_episode_info.extend(done_info)

            if collect_adversary_trajectory:
                pairs = [
                    (rewards, -next_rewards),
                    (not_dones, next_not_dones),
                    (actions, next_adv_perturbations),
                    (action_means, next_adv_perturbation_means),
                    (action_log_probs, next_adv_perturbation_log_probs),
                    (states, next_states),
                ]
            else:
                pairs = [
                    (rewards, next_rewards),
                    (not_dones, next_not_dones),
                    (actions, next_actions),
                    (action_means, next_action_means),
                    (action_log_probs, next_action_log_probs),
                    (states, last_states.unsqueeze(1)),
                ]
            for total, v in pairs:
                if total is states and not collect_perturbed_state:
                    total[:, t + 1] = v
                else:
                    total[:, t] = v
            last_states = next_states[:, 0, :]

        if collect_perturbed_state:
            # perturb the state after the last step as well
            adv_perturbation_pds = self.adversary_policy_model(last_states)
            next_adv_perturbations = self.adversary_policy_model.sample(adv_perturbation_pds)
            last_states = last_states + ch.nn.functional.hardtanh(next_adv_perturbations) * self.ROBUST_PPO_EPS
            states[:, -1] = last_states.unsqueeze(1)
        else:
            for e, flag in zip(self.envs, old_env_read_only_flags):
                e.normalizer_read_only = flag

        infos = np.array(list(zip(*completed_episode_info)))
        if infos.size > 0:
            avg_episode_length, avg_episode_reward = np.mean(infos, axis=1)
        else:
            avg_episode_length = avg_episode_reward = -1
        # the last state is never acted on
        states = states[:, :-1, :]
        trajs = Trajectories(rewards=rewards, action_log_probs=action_log_probs, not_dones=not_dones,
                             actions=actions, states=states, action_means=action_means)
        return avg_episode_length, avg_episode_reward, trajs

    def collect_saps(self, num_saps, collect_adversary_trajectory=False):
        """A rollout with values, GAE advantages and returns, unrolled to ``(T, ...)``."""
        with torch.no_grad():
            avg_ep_length, avg_ep_reward, trajs = self.run_trajectories(
                num_saps, collect_adversary_trajectory=collect_adversary_trajectory)
            val_model = self.adversary_val_model if collect_adversary_trajectory else self.val_model
            if self.HISTORY_LENGTH > 0:
                values = val_model(trajs.states, trajs.not_dones).squeeze(-1)
            else:
                values = val_model(trajs.states).squeeze(-1)
            advantages, returns = self.advantage_and_return(trajs.rewards, values, trajs.not_dones)
            trajs.advantages = advantages
            trajs.returns = returns
            trajs.values = values
            saps = trajs.unroll()
        return saps, avg_ep_reward, avg_ep_length

    def take_steps(self, saps, adversary_step=False, increment_scheduler=True):
        if adversary_step:
            policy_model = self.adversary_policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.ADV_POLICY_SCHEDULER
                val_scheduler = self.ADV_VALUE_SCHEDULER
            policy_params = Parameters(self.params.copy())
            policy_params.PPO_LR_ADAM = self.ADV_PPO_LR_ADAM
            policy_params.POLICY_ADAM = self.ADV_POLICY_ADAM
            policy_params.CLIP_EPS = policy_params.ADV_CLIP_EPS
            policy_params.ENTROPY_COEFF = policy_params.ADV_ENTROPY_COEFF
            val_model = self.adversary_val_model
            val_opt = self.adversary_val_opt
        else:
            policy_model = self.policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.POLICY_SCHEDULER
                val_scheduler = self.VALUE_SCHEDULER
            policy_params = self.params
            val_model = self.val_model
            val_opt = self.val_opt

        val_loss = value_step(saps.states, saps.returns, saps.advantages, saps.not_dones, val_model,
                              val_opt, self.params, old_vs=saps.values.detach()).mean()
        if self.ANNEAL_LR and increment_scheduler:
            val_scheduler.step()

        args = [saps.states, saps.actions, saps.action_log_probs, saps.rewards, saps.returns,
                saps.not_dones, saps.advantages, policy_model, policy_params]
        if self.MODE == 'adv_sa_ppo' and not adversary_step:
            policy_loss, surr_loss, entropy_bonus = robust_ppo_step(*args, self.robust_eps_scheduler)
        else:
            policy_loss, surr_loss, entropy_bonus = ppo_step(*args)
        if self.ANNEAL_LR and increment_scheduler:
            policy_scheduler.step()
        return policy_loss, surr_loss, entropy_bonus, val_loss.item()

    def train_step(self):
        """One iteration: ``ADV_POLICY_STEPS`` protagonist updates, then ``ADV_ADVERSARY_STEPS``
        adversary updates. Returns the mean episode returns of the last rollout of each."""
        for i in range(int(self.ADV_POLICY_STEPS)):
            avg_ep_reward = self.train_step_impl(adversary_step=False,
                                                 increment_scheduler=(i == self.ADV_POLICY_STEPS - 1))
        for i in range(int(self.ADV_ADVERSARY_STEPS)):
            adv_ep_reward = self.train_step_impl(adversary_step=True,
                                                 increment_scheduler=(i == self.ADV_ADVERSARY_STEPS - 1))
        self.n_steps += 1
        return avg_ep_reward, adv_ep_reward

    def train_step_impl(self, adversary_step=False, increment_scheduler=True):
        saps, avg_ep_reward, _ = self.collect_saps(self.T * self.NUM_ACTORS, collect_adversary_trajectory=adversary_step)
        self.take_steps(saps, adversary_step=adversary_step, increment_scheduler=increment_scheduler)
        return avg_ep_reward
