"""The robosuite DoorCausal task as a registered environment.

DoorCausal draws two factors at every reset: the door's rotation (left or right) and the
latch height (high or low). ``spurious_type`` binds them: ``"hhl"`` puts the latch high
when the door is rotated left, ``"hhr"`` inverts the binding. Training on one binding and
evaluating on the other is the semantic shift of "Seeing is not Believing" (Ding et al.,
NeurIPS 2023). The robot is a Kinova3 arm under joint-velocity control, observed through
proprioception and the object state, with a 300-step horizon.
"""
from __future__ import annotations

from robustrllib.gym.core import Env
from robustrllib.gym import spaces

HORIZON = 300
SPURIOUS_TYPES = ("hhl", "hhr")


def make_door_causal(spurious_type="hhl", **kwargs):
    """robosuite's ``GymWrapper`` around ``DoorCausal`` (flat observation, 5-tuple step)."""
    from robustrllib.gym.envs import robosuite
    from robustrllib.gym.envs.robosuite.wrappers import GymWrapper

    if spurious_type not in SPURIOUS_TYPES:
        raise ValueError(f"spurious_type must be one of {SPURIOUS_TYPES}, got {spurious_type!r}")
    config = {
        "robots": "Kinova3",
        "use_camera_obs": False,
        "use_object_obs": True,
        "has_offscreen_renderer": False,
        "has_renderer": False,
        "reward_shaping": True,
        "ignore_done": False,
        "control_freq": 20,
        "horizon": HORIZON,
        "render_camera": "agentview",
        "camera_names": "agentview",
        "camera_heights": 128,
        "camera_widths": 128,
        "controller_configs": robosuite.load_controller_config(default_controller="JOINT_VELOCITY"),
        "spurious_type": spurious_type,
    }
    config.update(kwargs)
    return GymWrapper(robosuite.make("DoorCausal", **config))


class DoorCausalEnv(Env):
    """``robustrllib.gym.Env`` view of :func:`make_door_causal`.

    The spaces are rebuilt as ``robustrllib.gym.spaces.Box`` and ``sim`` is passed through,
    so the dynamics adapter can edit ``sim.model``. ``step`` takes the raw action.
    """

    def __init__(self, spurious_type="hhl", **kwargs):
        self._env = make_door_causal(spurious_type, **kwargs)
        obs_space, act_space = self._env.observation_space, self._env.action_space
        self.observation_space = spaces.Box(low=obs_space.low, high=obs_space.high, dtype=obs_space.dtype)
        self.action_space = spaces.Box(low=act_space.low, high=act_space.high, dtype=act_space.dtype)
        self.metadata = getattr(self._env, "metadata", {"render_modes": []})
        self.render_mode = None

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    @property
    def sim(self):
        return self._env.sim


__all__ = ["DoorCausalEnv", "make_door_causal", "HORIZON", "SPURIOUS_TYPES"]
