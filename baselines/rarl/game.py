"""The two-player zero-sum game of RARL.

A protagonist acts in the task; an adversary perturbs the environment and is rewarded
with the negative of the protagonist's reward. The adversary is defined per task by the
``adversary`` hyperparameters, and acts through a unit box ``[-1, 1]^k`` scaled by ``force``:

``body_force``    forces on named MuJoCo bodies along chosen world axes
                  ``{kind: body_force, bodies: [foot], axes: [x, z], force: 5.0}``
``action_nudge``  an offset added to the first ``dims`` entries of the protagonist's action
                  ``{kind: action_nudge, dims: 3, force: 0.1}``
``hull_force``    a planar force at the centre of the CarRacing car body
                  ``{kind: hull_force, force: 5.0}``

Each player trains on its own view of the game (``PlayerView``): it chooses its own
action, and the other player's current policy, held in a ``Player``, fills in the other.
Both views wrap the environment that ``robustrllib.tasks.make_env`` returns, so the two
players see the same observations and the same (possibly shifted) reward.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def _unit_box(dim: int) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (int(dim),), dtype=np.float32)


class BodyForce:
    """External forces on MuJoCo bodies, written to ``xfrc_applied`` before every step."""

    def __init__(self, env, bodies, axes, force):
        base = env.unwrapped
        self._sim = base if hasattr(base, "data") else base.point_env   # PointMaze keeps MuJoCo on point_env
        self.ids = [int(self._sim.model.body(name).id) for name in bodies]
        self.axes = [AXES[a] for a in axes]
        self.force = float(force)
        self.space = _unit_box(len(self.ids) * len(self.axes))

    def reset(self):
        self._sim.data.xfrc_applied[:] = 0.0

    def apply(self, action, adv):
        u = self.force * np.clip(np.asarray(adv, dtype=np.float64).reshape(-1), -1.0, 1.0)
        xfrc = self._sim.data.xfrc_applied
        xfrc[:] = 0.0
        k = 0
        for body in self.ids:
            for axis in self.axes:
                xfrc[body, axis] = u[k]
                k += 1
        return action


class ActionNudge:
    """A bounded offset on the first ``dims`` action entries, clipped to the action space."""

    def __init__(self, env, dims, force):
        self.dims = int(dims)
        self.force = float(force)
        self.low, self.high = env.action_space.low, env.action_space.high
        self.space = _unit_box(self.dims)

    def reset(self):
        pass

    def apply(self, action, adv):
        a = np.array(action, dtype=np.float64).reshape(-1)
        a[: self.dims] += self.force * np.clip(np.asarray(adv, dtype=np.float64).reshape(-1), -1.0, 1.0)
        return np.clip(a, self.low, self.high).astype(np.float32)


class HullForce:
    """A planar (x, y) force at the centre of the CarRacing car body, applied before every step."""

    def __init__(self, env, force):
        self._base = env.unwrapped
        self.force = float(force)
        self.space = _unit_box(2)

    def reset(self):
        pass

    def apply(self, action, adv):
        u = self.force * np.clip(np.asarray(adv, dtype=np.float64).reshape(-1), -1.0, 1.0)
        self._base.car.hull.ApplyForceToCenter((float(u[0]), float(u[1])), True)
        return action


ADVERSARIES = {"body_force": BodyForce, "action_nudge": ActionNudge, "hull_force": HullForce}


def make_adversary(env, spec):
    spec = dict(spec)
    return ADVERSARIES[spec.pop("kind")](env, **spec)


def _as_action(space, action):
    if isinstance(space, gym.spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    return np.asarray(action, dtype=np.float32).reshape(space.shape)


class Player:
    """The current policy of one player, queried from inside the other player's view."""

    def __init__(self):
        self.model = None      # the SB3 agent, set once it exists
        self.vecnorm = None    # its observation normalisation, if any
        self.active = True     # an inactive adversary applies no perturbation

    def act(self, obs, space):
        if not self.active:
            return np.zeros(space.shape, dtype=np.float32)
        if self.model is None:
            return _as_action(space, space.sample())
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(obs, deterministic=False)
        return _as_action(space, action)


class PlayerView(gym.Wrapper):
    """One player's view of the game.

    ``role="protagonist"``: the task's action space and reward; with probability
    ``zero_force_prob`` a step runs without the adversary.
    ``role="adversary"``: the perturbation's action space and the negated reward.
    """

    def __init__(self, env, adversary, role, other: Player, zero_force_prob=0.0, seed=None):
        super().__init__(env)
        self.adversary = make_adversary(env, adversary)
        self.role = role
        self.other = other
        self.zero_force_prob = float(zero_force_prob)
        self._rng = np.random.default_rng(seed)
        self._obs = None
        if role == "adversary":
            self.action_space = self.adversary.space

    def reset(self, **kwargs):
        self.adversary.reset()
        self._obs, info = self.env.reset(**kwargs)
        return self._obs, info

    def step(self, action):
        space = self.adversary.space
        if self.role == "protagonist":
            pro = action
            if self.zero_force_prob > 0.0 and self._rng.random() < self.zero_force_prob:
                adv = np.zeros(space.shape, dtype=np.float32)
            else:
                adv = self.other.act(self._obs, space)
        else:
            pro, adv = self.other.act(self._obs, self.env.action_space), action
        self._obs, reward, terminated, truncated, info = self.env.step(self.adversary.apply(pro, adv))
        if self.role == "adversary":
            reward = -float(reward)
        return self._obs, reward, terminated, truncated, info
