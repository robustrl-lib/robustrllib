"""Expose a robustrllib env as a native ``gymnasium.Env``.

robustrllib is a hard fork with its own ``Env`` / ``spaces`` classes, so
third-party trainers (Stable-Baselines3, etc.) reject it on ``isinstance``
checks. This shim re-declares the spaces as gymnasium spaces and forwards the
(identical) reset/step API.
"""
from __future__ import annotations

import gymnasium as g
import gymnasium.spaces as gsp


def _to_gym_space(sp):
    name = type(sp).__name__
    if name == "Box":
        return gsp.Box(low=sp.low, high=sp.high, shape=sp.shape, dtype=sp.dtype)
    if name == "Discrete":
        return gsp.Discrete(int(sp.n), start=int(getattr(sp, "start", 0)))
    if name == "MultiDiscrete":
        return gsp.MultiDiscrete(sp.nvec)
    if name == "MultiBinary":
        return gsp.MultiBinary(sp.n)
    if name == "Dict":  # goal-conditioned / manipulation envs
        return gsp.Dict({k: _to_gym_space(v) for k, v in sp.spaces.items()})
    if name == "Tuple":
        return gsp.Tuple(tuple(_to_gym_space(s) for s in sp.spaces))
    raise NotImplementedError(f"no gymnasium mapping for space {name}")


class GymnasiumCompat(g.Env):
    """Wrap a robustrllib env so it *is a* ``gymnasium.Env``.

    Forwards the attributes third-party trainers (SB3, Tianshou, CleanRL) probe
    -- ``spec``, ``reward_range``, ``np_random``, ``render_mode``, ``metadata``
    -- and delegates anything else to the wrapped env via ``__getattr__``.
    """

    def __init__(self, env):
        self._env = env
        self.observation_space = _to_gym_space(env.observation_space)
        self.action_space = _to_gym_space(env.action_space)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.spec = getattr(env, "spec", None)
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    @property
    def np_random(self):
        return self._env.unwrapped.np_random

    @np_random.setter
    def np_random(self, value):
        self._env.unwrapped.np_random = value

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def env(self):
        """The wrapped robustrllib env, so that code walking a wrapper chain passes through."""
        return self._env

    def __getattr__(self, name):
        # Only reached for attributes not set on the compat object itself.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)


def as_gymnasium(env):
    """Public helper: present any robust env as a native ``gymnasium.Env``
    for third-party trainers. (The SB3 adapter applies this automatically.)"""
    return GymnasiumCompat(env)
