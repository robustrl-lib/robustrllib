"""Transition / process-noise shift.

Perturbs the *latent simulator state* after each step (``set_state``), so the
next transition starts from a noisy state -- a stochastic transition kernel.
This is the canonical way to perturb transitions (e.g. for maze tasks) and is
fundamentally different from observation noise, which leaves the true state
untouched.

Requires a MuJoCo-backed env (plain MuJoCo, maze point/ant env, ...).

The noise is Gaussian and that is the only form the target has, so the mode
must be ``gauss``; any other mode name is rejected.
"""
from __future__ import annotations

from robustrllib.adapters import get_adapter
from robustrllib.shifts.base import ShiftWrapper


class TransitionShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        if spec.mode != "gauss":
            raise ValueError(
                f"transition shift does not support mode {spec.mode!r}; the state "
                "noise is Gaussian, write mode 'gauss'")
        self._adapter = get_adapter(env)
        if not hasattr(self._adapter, "inject_state_noise"):
            raise NotImplementedError(
                "transition shift needs a MuJoCo-backed env (set_state); "
                f"{type(self._adapter).__name__} does not support it"
            )

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        if not (term or trunc):
            p = self._spec.params
            self._adapter.inject_state_noise(
                self._rng,
                q_sigma=p.get("q_sigma", p.get("sigma", 0.0)),
                v_sigma=p.get("v_sigma", p.get("sigma", 0.0)),
                scale=self._scale,
            )
        return obs, reward, term, trunc, info
