"""Action perturbation (applied before the action reaches the dynamics).

Two families live here.

ADDITIVE NOISE (gauss / shift / uniform) is the original behaviour: a draw from
``sample_noise`` added to the action, unchanged.

ADVERSARIAL MODES (oppose / oppose_goal) answer a different question. The
existing grids perturb what the world does to the agent -- gear scales the
agent's authority, frictionloss adds resistance -- but never oppose the agent's
*intent*. These do: the perturbation is a function of the action the policy just
chose, so the harm is directed rather than ambient.

WHY NEITHER IS A RENAMED ``actuator_gear``. That axis is a' = k*a for constant k.
Any adversarial mode that merely shrinks the action along its own direction
would reduce to exactly that and add nothing, so both modes here are built to be
outside its reach:

  oppose       subtracts a FIXED magnitude, not a fixed fraction. At eps 0.2 an
               action of norm 0.1 is reversed outright while an action of norm
               1.0 loses 20%. Gear can never flip a sign; it scales every action
               by the same factor regardless of size. This is the "small
               careful corrections are the easiest to undo" regime, which is
               where a maze policy threading a gap actually lives.

  oppose_goal  removes only the component of the action pointing at the goal and
               leaves the perpendicular component untouched. Gear cannot express
               this at all: it has no access to the task. This is the
               training-free stand-in for the SA-MDP worst case that ATLA's
               learned adversary approximates -- ATLA searches an L-inf ball for
               the perturbation that most reduces value; here "moves away from
               the goal" is used as the value proxy, so no critic and no
               training are needed and the condition stays reproducible.

``adversarial`` is the mode CLASS these three belong to, and is accepted as a
mode name: ``ShiftSpec("action", "adversarial", {"kind": "oppose", "eps":
0.2})`` builds exactly what ``ShiftSpec("action", "oppose", {"eps": 0.2})``
builds. `kind` defaults to ``oppose``, the only one defined on every action
space.
"""
from __future__ import annotations

import numpy as np

from robustrllib.shifts.base import ShiftWrapper, sample_noise


def _unit(v, eps=1e-8):
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


class ActionShift(ShiftWrapper):
    #: Modes that depend on the action or the environment state rather than on a
    #: fresh noise draw. Kept out of ``sample_noise`` deliberately: that helper
    #: is shared with the reward shift and the multi-agent wrappers, which have
    #: no action or state to condition on, so routing these through it would let
    #: a nonsensical config resolve silently instead of failing.
    _ADVERSARIAL = ("oppose", "rotate", "oppose_goal")

    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        # `adversarial` names the class; `kind` names the member.
        self._mode = spec.mode
        if spec.mode == "adversarial":
            self._mode = spec.params.get("kind", "oppose")
            if self._mode not in self._ADVERSARIAL:
                raise ValueError(
                    f"unknown adversarial action kind {self._mode!r}; "
                    f"expected one of {', '.join(self._ADVERSARIAL)}")
        self._adversarial = self._mode in self._ADVERSARIAL
        # ``step`` receives only the action, so a state-conditioned mode needs the
        # observation cached from the previous transition. Same pattern as
        # ObservationShift's running statistics.
        self._last_obs = None

    # -- goal geometry -----------------------------------------------------
    def _goal_direction(self):
        """Unit vector from the agent toward the goal, or None if unavailable.

        Read from the unwrapped env rather than the observation: the task
        configs flatten Dict observations (pointmaze_online sets
        flatten_keys: [observation, desired_goal]), so by the time an obs
        reaches this wrapper the goal may be an unlabelled slice. ``u.goal`` and
        ``point_env.data.qpos`` are stable regardless of the obs layout.
        """
        u = self.env.unwrapped
        goal = getattr(u, "goal", None)
        if goal is None:
            return None
        inner = getattr(u, "point_env", getattr(u, "ant_env", None))
        if inner is None or not hasattr(inner, "data"):
            return None
        pos = np.asarray(inner.data.qpos, dtype=np.float64)[:2]
        return _unit(np.asarray(goal, dtype=np.float64)[:2] - pos)

    # -- perturbations -----------------------------------------------------
    def _oppose(self, action):
        eps = float(self._spec.params.get("eps", 0.2)) * self._scale
        return action - eps * _unit(action)

    def _rotate(self, action):
        """Turn the action by a fixed angle, keeping its magnitude exactly.

        The third member of the axis, and the only one that leaves |a| alone:
        oppose changes magnitude, oppose_goal changes the goal-directed
        component, rotate changes neither -- it changes direction. actuator_gear
        cannot express it for the same reason it cannot express a sign flip.
        Defined for 2-D action spaces; higher-dimensional rotation needs a plane
        to be named, which no config currently does.
        """
        if action.shape[0] != 2:
            raise ValueError(
                f"rotate is defined for a 2-D action space, got {action.shape[0]}-D; "
                "a higher-dimensional rotation would need an explicit plane")
        th = np.deg2rad(float(self._spec.params.get("degrees", 60.0)) * self._scale)
        c, s_ = np.cos(th), np.sin(th)
        return np.array([c * action[0] - s_ * action[1],
                         s_ * action[0] + c * action[1]])

    def _oppose_goal(self, action):
        g = self._goal_direction()
        if g is None:
            raise ValueError(
                "oppose_goal needs a goal-conditioned env exposing `goal` and an "
                "inner point_env/ant_env; this env exposes neither, and silently "
                "falling back to an undirected perturbation would report a "
                "task-aware condition that is not one")
        eps = float(self._spec.params.get("eps", 1.0)) * self._scale
        if g.shape[0] != action.shape[0]:
            g = np.pad(g, (0, action.shape[0] - g.shape[0]))[: action.shape[0]]
        # Strip eps of the goal-directed component; leave the rest of the action
        # alone. eps=1 cancels it exactly, eps=2 reverses it.
        return action - eps * float(np.dot(action, g)) * g

    def _apply(self, action):
        # `clip_first` saturates the command to the action space BEFORE the
        # perturbation. It is a no-op for any policy that respects the action
        # space, and it exists because one that does not can hide behind the
        # actuator: a policy that emits the mean of an unbounded Gaussian can sit
        # far outside the action range, where a perturbation added upstream of
        # the actuator is clipped away before it reaches the plant and the
        # condition scores the parameterisation rather than the policy.
        # Perturbing the saturated command is what the
        # actuation kernel A(.|s,a) is meant to model, and it makes the channel
        # comparable across policies whose outputs live on different scales.
        if self._spec.params.get("clip_first"):
            lo, hi = self.action_space.low, self.action_space.high
            action = np.clip(action, lo, hi)
        if self._mode == "oppose":
            return self._oppose(action)
        if self._mode == "rotate":
            return self._rotate(action)
        if self._mode == "oppose_goal":
            return self._oppose_goal(action)
        return action + sample_noise(
            self._rng, self._spec.mode, self._spec.params, action.shape, self._scale)

    # -- gym API -----------------------------------------------------------
    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._last_obs = out[0] if isinstance(out, tuple) else out
        return out

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = self._apply(action)
        if self.action_space is not None and hasattr(self.action_space, "low"):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, term, trunc, info = self.env.step(action)
        self._last_obs = obs
        self._t += 1
        return obs, reward, term, trunc, info
