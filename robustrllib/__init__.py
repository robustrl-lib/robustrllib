"""RobustRLlib: a unified library and benchmark for robust reinforcement learning.

A shift is declared as data (``ShiftSpec``), stacked on an environment as a wrapper
by ``make_robust`` and drawn from a seeded generator of its own, so that any list of
shifts composes and reproduces. The environment API is ``robustrllib.gym``.

Quick start::

    from robustrllib import make_robust, ShiftSpec

    env = make_robust(
        "Hopper-v5",
        shifts=[
            ShiftSpec("observation", "gauss", {"sigma": 0.05}),
            ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
        ],
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
"""
from robustrllib.spec import ShiftSpec
from robustrllib.pipeline import build_pipeline, make_robust
from robustrllib.config import load_config, RobustConfig
from robustrllib.shifts import bind_actor
from robustrllib.tasks import make_env

__all__ = [
    "ShiftSpec",
    "build_pipeline",
    "make_robust",
    "load_config",
    "RobustConfig",
    "bind_actor",
    "make_env",
]
