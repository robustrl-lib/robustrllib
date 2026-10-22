"""Declarative specs for tasks and shifts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Where a shift acts in the MDP loop.
#
# `latency` is the sim-to-real axis: the delay between a command being issued
# and its effect being observed. It is NOT expressible as observation or action
# noise -- it changes the effective transition operator, and the sim-to-real
# literature repeatedly finds it dominant. Peng et al. 2018 (arXiv:1710.06537)
# Table III: freezing the action timestep drops real-robot success 0.89 -> 0.29,
# the single largest drop of any ablated axis. Tan et al. 2018 (arXiv:1804.10332)
# call missing latency modelling one of the "two major causes of the reality gap"
# and randomize it over 0-40 ms.
TARGETS = ("observation", "action", "reward", "cost", "dynamics", "transition",
           "latency", "appearance")
# How the perturbation value is drawn / applied.
#   loguniform  dynamics: exp(U(log lo, log hi)). Required for the wide
#               multiplicative ranges the sim-to-real papers use -- over
#               [0.2, 20] a plain uniform puts ~90% of its mass above 1, so the
#               "symmetric" range is not symmetric in effect.
#   relative    observation: sigma proportional to each feature's own scale
#               (Peng et al. use 5% of the running std per feature).
#   bias        observation: constant offset drawn once per episode, i.e. a
#               sensor zero-offset rather than per-step noise (Tan et al.
#               randomize IMU bias over +/-0.05 rad separately from IMU noise).
#   fixed/buffer/substep/interp  latency modes; see shifts/latency.py.
#   delay       latency: the REWARD arrives late while observation and action
#               stay on time -- credit latency, the fifth timing mode. It lives
#               under `latency` because it is a timing mismatch, not a change in
#               what the reward is. `reward` + `delay` builds the same wrapper.
#   adversarial a bounded perturbation computed from the current state or
#               action to counter the policy, instead of being drawn or fixed.
#               observation: K-step PGD in an L-inf ball of radius `eps` around
#               the true observation, maximizing the change in the policy's
#               action; needs the policy's differentiable actor.
#               action: `kind` selects oppose / rotate / oppose_goal, which are
#               also accepted as mode names of their own.
#   hue         appearance: rotate an object's RGB triple by `degrees` in HSV,
#               preserving saturation and value -- "the same object in a
#               different colour", which scaling RGBA cannot express.
#   translate   dynamics: add a metric offset to a position vector (body_pos).
#               Scaling moves an entity radially by an amount that depends on
#               its distance from the parent origin, so it cannot express
#               "5 cm to the left".
#   oppose      action: subtract a FIXED magnitude along the action's own
#               direction, a' = a - eps*a/|a|. Distinct from actuator_gear
#               (a' = k*a) because the reduction does not scale with |a|: a
#               small action is reversed outright while a large one is barely
#               touched, and gear can never flip a sign.
#   rotate      action: turn the action by `degrees`, magnitude exactly
#               preserved. The one member of the adversarial axis that changes
#               direction rather than size, so it is orthogonal to both gear and
#               oppose. 2-D action spaces only.
#   oppose_goal action: remove eps of the action's goal-directed component and
#               leave the perpendicular component intact. The training-free
#               stand-in for ATLA's learned worst case -- "away from the goal"
#               replaces a critic as the value proxy, so the condition needs no
#               weights and stays reproducible. Goal-conditioned envs only.
#   tint        appearance: SET the geom colour to a saturated hue instead of
#               rotating it. Required wherever the visible colour comes from a
#               texture and geom_rgba is an achromatic multiplier (a grey rgba
#               under a coloured texture), where rotating the hue changes nothing.
#   swap        appearance: exchange the colour of two named geoms. A two-entity
#               operation, so it is not expressible as a dynamics parameter. Use
#               it when the instruction names a colour: swapping target and
#               distractor moves the REFERENT, separating a policy that grounds
#               the language from one that memorised a position or shape.
MODES = ("gauss", "shift", "uniform", "scale", "set", "adversarial",
         "loguniform", "relative", "bias",
         "fixed", "buffer", "substep", "interp",
         "hue", "swap", "translate", "tint",
         "oppose", "rotate", "oppose_goal",
         "push", "delay")


@dataclass
class ShiftSpec:
    """One perturbation configuration. A benchmark experiment is an ordered list of these.

    A perturbation configuration is the tuple (source, factor, mode, magnitude,
    resampling rule, optional schedule, seed). The fields write it: ``target``
    is the source; ``mode`` is the mode and also fixes the resampling rule (per
    step, per episode or fixed; the latency modes ``buffer`` and ``interp`` take
    ``params["resample"] = "step"``); ``params`` holds the factor (``param`` and
    ``index``, ``body``, ``geoms``, ``keys`` or ``info_key``; otherwise the whole
    signal) and the magnitude (``factor``, ``value``, ``sigma``, ``eps``,
    ``steps`` ...); ``schedule`` is the optional schedule; ``seed`` the seed.

    The list order is the wrapper-stacking order, so it is explicit and
    reproducible (e.g. dynamics before observation noise).

    Args:
        target: one of :data:`TARGETS`.
        mode: one of :data:`MODES`.
        params: mode-specific knobs, e.g. ``{"mu":0,"sigma":0.05}`` for gauss,
            ``{"low":-.1,"high":.1}`` for uniform, ``{"param":"gravity",
            "factor":1.3}`` for dynamics scale.
        schedule: optional non-stationary schedule modulating intensity over
            time, e.g. ``{"type":"linear","start":0.0,"end":1.0,"t0":0,"t1":1e6}``.
            ``None`` means stationary.
        seed: optional per-shift seed; falls back to the pipeline seed.
    """

    target: str
    mode: str
    params: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ValueError(f"unknown shift target {self.target!r}, expected one of {TARGETS}")
        if self.mode not in MODES:
            raise ValueError(f"unknown shift mode {self.mode!r}, expected one of {MODES}")
