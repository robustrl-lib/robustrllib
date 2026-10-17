"""Dynamics adapters: one interface to edit a simulator's physics in memory.

Three backends: MuJoCo (locomotion, Fetch, PointMaze, Adroit), Box2D (LunarLander,
CarRacing) and robosuite (DoorCausal). ``get_adapter`` picks one from the environment.
"""
from robustrllib.adapters.base import DynamicsAdapter
from robustrllib.adapters.mujoco import MujocoDynamicsAdapter
from robustrllib.adapters.box2d import Box2DDynamicsAdapter


def get_adapter(env):
    """Pick a dynamics adapter for ``env`` by sniffing its unwrapped attributes.

    For PointMaze the physics live in an inner ``point_env``, which is the model the
    dynamics shifts must reach.
    """
    inner = env.unwrapped
    # Robosuite keeps its physics under env.sim (env.sim.model).
    if hasattr(inner, "sim") and hasattr(inner.sim, "model"):
        from robustrllib.adapters.robosuite import RobosuiteDynamicsAdapter

        return RobosuiteDynamicsAdapter(inner)
    inner = getattr(inner, "point_env", inner)
    if hasattr(inner, "model") and hasattr(inner, "data"):
        return MujocoDynamicsAdapter(inner)
    if hasattr(inner, "world"):  # Box2D b2World
        return Box2DDynamicsAdapter(inner)
    raise NotImplementedError(
        f"no dynamics adapter for {type(inner).__name__}; "
        "implement one in robustrllib/adapters/"
    )


__all__ = [
    "DynamicsAdapter",
    "MujocoDynamicsAdapter",
    "Box2DDynamicsAdapter",
    "get_adapter",
]
