"""Template dynamics adapter -- copy this file to add a new simulator.

Worked example for a hypothetical robot stack whose physics live on the env as
plain attributes and on a list of joint objects. Replace the bodies of the
getters/setters with however your simulator stores these quantities; the class
below needs no changes at all.

Three things an adapter must guarantee, and why:

1. `get_nominal` returns the value the *pristine* model had. Shifts are usually
   expressed as `factor` (x0.8, x1.2) relative to nominal, so if you read the
   already-perturbed value you would compound factors across resets and the
   severity levels in an eval grid would stop meaning what they say.

2. `set_param` writes into the *live* model, i.e. the object the simulator will
   actually step. Writing to a config object that is only read at construction
   time silently does nothing -- the classic way to produce a grid where every
   condition scores the same.

3. `reset_all` restores nominal exactly, so a single process can evaluate many
   conditions in sequence without drift.

The framework calls these at well-defined times: nominal is cached eagerly when
the wrapper is built (while the model is still pristine), and `set_param` is
re-applied after every `env.reset()`, because most simulators rebuild their
model there.
"""
from __future__ import annotations

from robustrllib.adapters.base import DynamicsAdapter


# --- helpers for the per-entity parameters -------------------------------

def _joints(env):
    """Every joint object whose properties we are allowed to perturb."""
    return list(env.robot.joints)


def _resolve(env, index):
    """Turn `index` into concrete joints.

    `index` is what an eval grid writes, and it may be:
      * "all"          -> every joint (global scaling)
      * "shoulder_pan" -> one joint, addressed by NAME
      * 3              -> one joint, addressed by position
    Supporting names is worth the few lines: a grid that says
    `index: shoulder_pan` stays readable and survives reordering of the model.
    """
    joints = _joints(env)
    if index == "all" or index is None:
        return joints
    if isinstance(index, str):
        for j in joints:
            if j.name == index:
                return [j]
        raise ValueError(f"no joint named {index!r}")
    return [joints[int(index)]]


# --- the parameter table --------------------------------------------------
# logical name -> (getter, setter). Both take (env, index).
#
# The logical names are the benchmark's vocabulary: an eval grid says
# `param: actuator_gear` and does not care whether the backend is MuJoCo, Box2D
# or your stack. Reuse an existing name whenever the physical meaning matches,
# so the same grid can run on several simulators.
_FIELDS = {
    # --- scalar, whole-scene parameters ---
    "gravity": (
        lambda e, i: e.physics.gravity_z,
        lambda e, i, v: setattr(e.physics, "gravity_z", v),
    ),
    "timestep": (
        lambda e, i: e.physics.dt,
        lambda e, i, v: setattr(e.physics, "dt", v),
    ),

    # --- per-entity parameters (respect `index`) ---
    # Control authority: usually the single most informative axis for a real
    # robot, since torque limits are what differ most between rigs.
    "actuator_gear": (
        lambda e, i: _resolve(e, i)[0].torque_limit,
        lambda e, i, v: [setattr(j, "torque_limit", v) for j in _resolve(e, i)],
    ),
    "dof_damping": (
        lambda e, i: _resolve(e, i)[0].damping,
        lambda e, i, v: [setattr(j, "damping", v) for j in _resolve(e, i)],
    ),
    # Static (stiction) resistance. On manipulation tasks this, not damping, is
    # what actually resists motion -- Adroit's door latch has damping 0.05 but
    # frictionloss 5, so perturbing damping barely moves the task.
    "dof_frictionloss": (
        lambda e, i: _resolve(e, i)[0].static_friction,
        lambda e, i, v: [setattr(j, "static_friction", v) for j in _resolve(e, i)],
    ),
    "body_mass": (
        lambda e, i: _resolve(e, i)[0].link_mass,
        lambda e, i, v: [setattr(j, "link_mass", v) for j in _resolve(e, i)],
    ),
}


class TemplateDynamicsAdapter(DynamicsAdapter):
    """Reference implementation -- the class body is backend-independent.

    Everything backend-specific lives in `_FIELDS` above, so adapting a new
    simulator means writing getters and setters, not new control flow.
    """

    def __init__(self, env):
        self._env = env
        self._nominal: dict = {}

    def list_params(self):
        return tuple(_FIELDS)

    def get_nominal(self, param, index=None):
        # Key on (param, index): scaling one joint must not overwrite the
        # nominal cached for another.
        key = (param, index)
        if key not in self._nominal:
            self._nominal[key] = float(_FIELDS[param][0](self._env, index))
        return self._nominal[key]

    def set_param(self, param, value, index=None):
        self.get_nominal(param, index)   # cache nominal BEFORE the first write
        _FIELDS[param][1](self._env, index, float(value))

    def reset_all(self):
        for (param, index), nominal in self._nominal.items():
            _FIELDS[param][1](self._env, index, nominal)


# --- wiring -----------------------------------------------------------------
# Add a branch to robustrllib/adapters/__init__.py:get_adapter that
# recognises your env, e.g.
#
#     if hasattr(inner, "robot") and hasattr(inner, "physics"):
#         from robustrllib.adapters.template import TemplateDynamicsAdapter
#         return TemplateDynamicsAdapter(inner)
#
# Sniff an attribute that only your stack has, and put the branch before more
# generic ones (a wrapper exposing `.model`/`.data` would otherwise be captured
# by the MuJoCo adapter).
#
# Then write an eval grid over the axes you exposed, and SCREEN THEM with a
# trained policy before trusting the numbers -- see docs/ROBUST_GYM_V2_ONBOARDING.md
# section 5. Axes that look obvious can be complete no-ops: global gravity and
# body-mass scaling move a fixed-base manipulator's return by less than the
# evaluation noise, because the controller compensates them.
