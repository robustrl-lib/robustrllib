"""Dynamics adapter for robosuite (parameters live under ``env.sim.model``)."""
from __future__ import annotations

import numpy as np

from robustrllib.adapters.mujoco import MujocoDynamicsAdapter


class RobosuiteDynamicsAdapter(MujocoDynamicsAdapter):
    """The MuJoCo adapter pointed at ``env.sim.model`` / ``env.sim.data``.

    ``env`` is the robosuite environment, or any object exposing ``.sim``. robosuite
    rebuilds its simulator on reset, so the model and data are looked up on every access.
    """

    def __init__(self, env):
        self._wrap = env
        self._nominal: dict = {}
        self._needs_setconst = False

    @property
    def _model(self):
        return self._wrap.sim.model

    @property
    def _env(self):
        return self._wrap.sim

    def _name2id(self, kind, name):
        # robosuite's model binding uses mujoco-py-style name lookups; kinds the binding
        # has no lookup for go through mujoco.mj_name2id on the raw model.
        m = self._model
        if kind == "dof":
            return int(m.jnt_dofadr[m.joint_name2id(name)])
        lookup = {"body": "body_name2id", "geom": "geom_name2id",
                  "actuator": "actuator_name2id", "camera": "camera_name2id",
                  "light": "light_name2id", "material": "material_name2id"}
        fn = getattr(m, lookup.get(kind, ""), None)
        if fn is not None:
            return int(fn(name))
        import mujoco

        from robustrllib.adapters.mujoco import _OBJ

        raw = getattr(m, "_model", m)
        idx = mujoco.mj_name2id(raw, getattr(mujoco.mjtObj, _OBJ[kind]), name)
        if idx < 0:
            raise ValueError(f"no {kind} named {name!r} in this model")
        return int(idx)

    def apply(self, *a, **k):
        out = super().apply(*a, **k)
        # A forward pass recomputes the world-frame quantities (camera and body poses)
        # that depend on the edited model fields.
        try:
            self._env.forward()
        except Exception:
            pass
        return out

    def _recompute_derived(self):
        """``mj_setConst`` on the raw ``mjModel`` / ``mjData`` behind robosuite's bindings.

        robosuite exposes them as ``sim.model._model`` and ``sim.data._data``; when a
        binding does not, ``sim.forward()`` refreshes the derived quantities instead.
        """
        if not self._needs_setconst:
            return
        import mujoco

        raw_m = getattr(self._model, "_model", None)
        raw_d = getattr(getattr(self._env, "data", None), "_data", None)
        if raw_m is None or raw_d is None:
            self._env.forward()
            self._needs_setconst = False
            return
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(raw_m, spec), dtype=np.float64)
        mujoco.mj_getState(raw_m, raw_d, state, spec)
        mujoco.mj_setConst(raw_m, raw_d)
        mujoco.mj_setState(raw_m, raw_d, state, spec)
        mujoco.mj_forward(raw_m, raw_d)
        self._needs_setconst = False

    def inject_state_noise(self, rng, q_sigma, v_sigma, scale=1.0):
        sim = self._env
        state = sim.get_state()
        if q_sigma:
            state.qpos[:] = state.qpos + rng.normal(0.0, q_sigma * scale, size=state.qpos.shape)
        if v_sigma:
            state.qvel[:] = state.qvel + rng.normal(0.0, v_sigma * scale, size=state.qvel.shape)
        sim.set_state(state)
        sim.forward()
