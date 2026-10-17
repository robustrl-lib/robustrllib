"""Box2D dynamics adapter (LunarLander / BipedalWalker / CarRacing).

Unlike MuJoCo there is no single ``model`` array: parameters live as instance
attributes read live each step (``wind_power``), on the live ``b2World``
(``gravity``), or on already-created fixtures (``friction``). So each logical
param is a (getter, setter) pair. The adapter is applied *after* ``env.reset()``
(LunarLander rebuilds its world on reset), which our ``DynamicsShift.reset``
already guarantees.
"""
from __future__ import annotations

import numpy as np

from robustrllib.adapters.base import DynamicsAdapter


def _set_gravity(env, v):
    if hasattr(env, "gravity"):
        env.gravity = v
    if getattr(env, "world", None) is not None:
        env.world.gravity = (0.0, v)


def _set_enable_wind(env, value):
    enabled = bool(value)
    env.enable_wind = enabled
    # DynamicsShift applies after reset, while LunarLander normally creates
    # these indices inside reset only when wind was already enabled.
    if enabled and (
        not hasattr(env, "wind_idx") or not hasattr(env, "torque_idx")
    ):
        env.wind_idx = env.np_random.integers(-9999, 9999)
        env.torque_idx = env.np_random.integers(-9999, 9999)


def _all_fixtures(env):
    world = getattr(env, "world", None)
    if world is None:
        return
    for body in world.bodies:
        for fx in body.fixtures:
            yield fx


# logical name -> (getter, setter); both take the unwrapped env.
_FIELDS = {
    "gravity": (lambda e: getattr(e, "gravity", e.world.gravity[1]), _set_gravity),
    "enable_wind": (
        lambda e: float(e.enable_wind),
        _set_enable_wind,
    ),
    "wind_power": (lambda e: e.wind_power, lambda e, v: setattr(e, "wind_power", v)),
    "turbulence_power": (lambda e: e.turbulence_power,
                         lambda e, v: setattr(e, "turbulence_power", v)),
    "main_engine_power": (
        lambda e: e.main_engine_power,
        lambda e, v: setattr(e, "main_engine_power", v),
    ),
    "side_engine_power": (
        lambda e: e.side_engine_power,
        lambda e, v: setattr(e, "side_engine_power", v),
    ),
    # One logical axis for the coupled thrust scale used by the benchmark.
    # Sampling main and side engines independently would add an unintended
    # engine-imbalance shift that is absent from the evaluation grid.
    "engine_power": (
        lambda e: (e.main_engine_power, e.side_engine_power),
        lambda e, v: (
            setattr(e, "main_engine_power", float(v[0])),
            setattr(e, "side_engine_power", float(v[1])),
        ),
    ),
    "friction": (
        lambda e: next(iter(_all_fixtures(e))).friction,
        lambda e, v: [setattr(fx, "friction", v) for fx in _all_fixtures(e)],
    ),
}


class Box2DDynamicsAdapter(DynamicsAdapter):
    def __init__(self, env):
        self._env = env
        self._nominal: dict = {}

    def list_params(self):
        return tuple(_FIELDS)

    def get_nominal(self, param, index=None):
        if param not in self._nominal:
            value = _FIELDS[param][0](self._env)
            self._nominal[param] = (
                np.asarray(value, dtype=np.float64)
                if param == "engine_power"
                else float(value)
            )
        return self._nominal[param]

    def set_param(self, param, value, index=None):
        self.get_nominal(param)          # cache nominal before first write
        _FIELDS[param][1](
            self._env,
            np.asarray(value, dtype=np.float64)
            if param == "engine_power"
            else float(value),
        )

    def reset_all(self):
        for param, nominal in self._nominal.items():
            _FIELDS[param][1](self._env, nominal)
