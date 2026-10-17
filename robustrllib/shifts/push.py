"""External push disturbance (target: dynamics, mode: push).

Every dynamics mode before this one edits ``mjModel`` -- a parameter of the
world. A push edits ``mjData.xfrc_applied``: an external wrench on one body for
a few control steps, then nothing. It is the standard locomotion sim-to-real
primitive (HumanoidGym pushes the base every few seconds with a random
horizontal velocity; OmniH2O uses an interval + magnitude pair) and it cannot be
imitated by any model-parameter edit, because it is impulsive and directional
rather than a change of the operator the policy closes its loop around.

Params
------
body       body name receiving the force (Hopper: "torso").
interval   control steps between push onsets (default 200).
duration   control steps a push lasts (default 10).
force      magnitude in Newtons; multiplied by the schedule scale.
direction  "random_xy" (default: uniform angle in the horizontal plane, drawn
           per push), "random_x" (sign drawn per push, sagittal plane -- the
           right choice for planar Hopper/Walker, whose y motion is
           unactuated), or a fixed 3-vector.

Per-episode phase: the first onset is drawn uniformly in [0, interval), so a
policy cannot phase-lock to a fixed push clock, and paired conditions get the
same push sequence because ``reset(seed=...)`` reseeds the wrapper RNG.
``xfrc_applied`` is cleared on reset and whenever a push window ends, so no
force leaks across episodes or into an unshifted evaluation of the same env.
"""
from __future__ import annotations

import numpy as np

from robustrllib.shifts.base import ShiftWrapper


class PushShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        import mujoco

        base = env.unwrapped
        if not hasattr(base, "model") or not hasattr(base, "data"):
            raise NotImplementedError(
                "push shift needs a MuJoCo env exposing model/data; "
                f"{type(base).__name__} does not")
        p = spec.params
        self._mj = mujoco
        self._base = base
        name = p.get("body", "torso")
        self._body = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self._body < 0:
            raise ValueError(f"push: no body named {name!r}")
        self._interval = int(p.get("interval", 200))
        self._duration = int(p.get("duration", 10))
        if self._interval <= 0 or self._duration <= 0:
            raise ValueError("push: interval and duration must be positive")
        self._force = float(p.get("force", 50.0))
        self._direction = p.get("direction", "random_xy")
        self._phase = 0
        self._active_until = -1
        self._vec = np.zeros(3)
        self.push_log: list[tuple[int, np.ndarray]] = []

    # -- helpers -----------------------------------------------------------
    def _clear(self):
        self._base.data.xfrc_applied[self._body, :] = 0.0

    def _draw_direction(self):
        d = self._direction
        if isinstance(d, str):
            if d == "random_xy":
                ang = self._rng.uniform(0.0, 2.0 * np.pi)
                return np.array([np.cos(ang), np.sin(ang), 0.0])
            if d == "random_x":
                return np.array([self._rng.choice([-1.0, 1.0]), 0.0, 0.0])
            raise ValueError(f"push: unknown direction {d!r}")
        v = np.asarray(d, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # -- gym API -----------------------------------------------------------
    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._clear()
        self._phase = int(self._rng.integers(0, self._interval))
        self._active_until = -1
        self.push_log = []
        return out

    def step(self, action):
        t = self._t
        if t >= self._phase and (t - self._phase) % self._interval == 0:
            self._vec = self._draw_direction()
            self._active_until = t + self._duration
            self.push_log.append((t, self._vec.copy()))
        if t < self._active_until:
            self._base.data.xfrc_applied[self._body, :3] = self._vec * self._force * self._scale
        else:
            self._clear()
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        if term or trunc:
            self._clear()
        return obs, reward, term, trunc, info
