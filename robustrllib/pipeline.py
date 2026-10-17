"""Compose a base env with an ordered list of shift wrappers."""
from __future__ import annotations

from typing import Sequence

import robustrllib.gym as gym
from robustrllib.core_wrapper import RobustCoreWrapper
from robustrllib.shifts import SHIFT_REGISTRY
from robustrllib.spec import ShiftSpec


def build_pipeline(env: gym.Env, shifts: Sequence[ShiftSpec], seed: int | None = None) -> gym.Env:
    """Wrap ``env`` with the compatibility bridge, then with each shift, in list order.

    Earlier specs sit closer to the env; e.g. put a ``dynamics`` shift before an
    ``observation`` shift so the dynamics change feeds the observation.
    """
    _warn_duplicate_dynamics(shifts)
    env = RobustCoreWrapper(env)
    for spec in shifts:
        wrapper_cls = SHIFT_REGISTRY[spec.target]
        env = wrapper_cls(env, spec, seed=seed)
    return env


def _warn_duplicate_dynamics(shifts: Sequence[ShiftSpec]) -> None:
    """Two dynamics shifts on the same (param, index) both set the value
    absolutely from nominal, so the outer one wins (not a product). Warn rather
    than silently surprise the user."""
    import warnings

    seen = set()
    for s in shifts:
        if s.target == "dynamics":
            key = (s.params.get("param"), s.params.get("index"))
            if key in seen:
                warnings.warn(
                    f"multiple dynamics shifts target {key}; the last one wins "
                    "(they do not compose multiplicatively)",
                    stacklevel=3,
                )
            seen.add(key)


def make_robust(
    env_id: str,
    shifts: Sequence[ShiftSpec] | None = None,
    seed: int | None = None,
    **make_kwargs,
) -> gym.Env:
    """``gym.make`` + :func:`build_pipeline`. Returns a standard ``step(action)`` env.

    Everything this function perturbs is declared in ``shifts``; nothing arrives
    through the command line.
    """
    env = gym.make(env_id, **make_kwargs)
    return build_pipeline(env, shifts or [], seed=seed)
