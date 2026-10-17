"""Build the environment a task card describes.

A task card names the environment and how a learner sees it::

    name: fetchreach
    env_id: FetchReachDense-v3
    max_episode_steps: 50
    observation: {keys: [observation, desired_goal], relative_goal: true}

``make_env`` stacks, in order: the environment with its shifts (``make_robust``), the
observation adapter of the card, and the native ``gymnasium`` interface that
third-party learners check for. Training and evaluation build their environments
through this one function, so a policy always meets the observation it was trained on.

Card keys
---------
``env_id``             registered id
``env_kwargs``         constructor arguments
``max_episode_steps``  time limit (optional)
``observation``        optional adapter:
                       ``keys`` -- concatenate these entries of a dict observation;
                       ``relative_goal: true`` -- the goal entry becomes
                       ``desired_goal - observation[:3]`` (gripper-relative);
                       ``frames: k`` -- stack the last ``k`` images, channel first;
                       ``gray: true`` -- convert each image to grayscale first.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from robustrllib.compat import as_gymnasium
from robustrllib.pipeline import make_robust
from robustrllib.spec import ShiftSpec


def _specs(shifts) -> list[ShiftSpec]:
    return [s if isinstance(s, ShiftSpec) else ShiftSpec(**s) for s in (shifts or [])]


class FlattenKeys(gym.ObservationWrapper):
    """Concatenate entries of a dict observation; optionally make the goal relative."""

    def __init__(self, env, keys: Sequence[str], relative_goal: bool = False):
        super().__init__(env)
        self.keys = list(keys)
        self.relative_goal = bool(relative_goal)
        dim = sum(int(np.prod(env.observation_space[k].shape)) for k in self.keys)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)

    def observation(self, obs):
        parts = []
        for key in self.keys:
            value = np.asarray(obs[key], dtype=np.float32).ravel()
            if self.relative_goal and key == "desired_goal":
                value = value - np.asarray(obs["observation"], dtype=np.float32)[:3]
            parts.append(value)
        return np.concatenate(parts)


class FrameStack(gym.Wrapper):
    """Channel-first stack of the last ``frames`` images: (H, W, C) -> (frames * C, H, W),
    or (frames, H, W) with ``gray``."""

    def __init__(self, env, frames: int, gray: bool = False):
        super().__init__(env)
        self.frames, self.gray = int(frames), bool(gray)
        h, w, c = env.observation_space.shape
        channels = self.frames * (1 if self.gray else c)
        self.observation_space = gym.spaces.Box(0, 255, (channels, h, w), dtype=np.uint8)
        self._stack: list[np.ndarray] = []

    def _chw(self, obs):
        img = np.asarray(obs, dtype=np.uint8)
        if self.gray:
            img = (img @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)[..., None]
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._stack = [self._chw(obs)] * self.frames
        return np.concatenate(self._stack, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._stack = self._stack[1:] + [self._chw(obs)]
        return np.concatenate(self._stack, axis=0), reward, terminated, truncated, info


class ResampleAttributes(gym.Wrapper):
    """Draw environment attributes uniformly before every reset.

    For parameters an environment reads when it resets (CarRacing's
    ``friction_scale``, ``morphology_scale``, ``engine_power_scale``), domain
    randomisation sets the attribute and lets the reset build the episode from it.
    """

    def __init__(self, env, ranges: Mapping[str, Sequence[float]], seed: int | None = None):
        super().__init__(env)
        self.ranges = {k: (float(v[0]), float(v[1])) for k, v in ranges.items()}
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        base = self.env.unwrapped
        for name, (low, high) in self.ranges.items():
            setattr(base, name, float(self._rng.uniform(low, high)))
        return self.env.reset(**kwargs)


def make_env(
    task: Mapping[str, Any],
    shifts: Sequence[ShiftSpec | Mapping] | None = None,
    seed: int | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    resample: Mapping[str, Sequence[float]] | None = None,
) -> gym.Env:
    """The environment of ``task`` with ``shifts`` stacked on it, as a ``gymnasium.Env``.

    ``env_kwargs`` are merged over the card's constructor arguments (an evaluation
    condition that is a constructor argument, such as CarRacing's friction).
    ``resample`` draws the named environment attributes before every reset.
    """
    kwargs = dict(task.get("env_kwargs") or {})
    kwargs.update(env_kwargs or {})
    if task.get("max_episode_steps"):
        kwargs["max_episode_steps"] = int(task["max_episode_steps"])
    env = as_gymnasium(make_robust(task["env_id"], shifts=_specs(shifts), seed=seed, **kwargs))
    if resample:
        env = ResampleAttributes(env, resample, seed=seed)
    obs = task.get("observation") or {}
    if obs.get("keys"):
        env = FlattenKeys(env, obs["keys"], relative_goal=obs.get("relative_goal", False))
    if obs.get("frames"):
        env = FrameStack(env, obs["frames"], gray=obs.get("gray", False))
    return env


__all__ = ["make_env", "FlattenKeys", "FrameStack", "ResampleAttributes"]
