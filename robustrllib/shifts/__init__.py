"""Shift wrappers: one composable Gymnasium wrapper per ShiftSpec."""
from robustrllib.shifts.base import ShiftWrapper, sample_noise
from robustrllib.shifts.observation import ObservationShift, bind_actor
from robustrllib.shifts.action import ActionShift
from robustrllib.shifts.reward import RewardShift
from robustrllib.shifts.dynamics import DynamicsShift
from robustrllib.shifts.appearance import AppearanceShift
from robustrllib.shifts.transition import TransitionShift
from robustrllib.shifts.latency import LatencyShift
from robustrllib.shifts.push import PushShift


def _dynamics(env, spec, seed=None):
    """`dynamics` dispatch: mode `push` edits mjData (an external wrench),
    every other mode edits mjModel through DynamicsShift."""
    if spec.mode == "push":
        return PushShift(env, spec, seed=seed)
    return DynamicsShift(env, spec, seed=seed)

def _latency(env, spec, seed=None):
    """`latency` dispatch: mode `delay` postpones the REWARD (credit latency)
    and is carried by RewardShift, which owns the accumulator and the flush on
    termination; every other mode delays the action, the observation or the
    control period through LatencyShift."""
    if spec.mode == "delay":
        return RewardShift(env, spec, seed=seed)
    return LatencyShift(env, spec, seed=seed)

# Dispatch table used by the pipeline builder.
SHIFT_REGISTRY = {
    "observation": ObservationShift,
    "action": ActionShift,
    "reward": RewardShift,
    "cost": RewardShift,
    "dynamics": _dynamics,
    "appearance": AppearanceShift,
    "transition": TransitionShift,
    "latency": _latency,
}

__all__ = [
    "ShiftWrapper",
    "sample_noise",
    "ObservationShift",
    "ActionShift",
    "RewardShift",
    "DynamicsShift",
    "AppearanceShift",
    "TransitionShift",
    "LatencyShift",
    "PushShift",
    "SHIFT_REGISTRY",
    "bind_actor",
]
