"""MuJoCo dynamics adapter (in-memory; no XML round-trip).

Supports global vector fields (gravity, wind), per-entity 1D fields (body_mass,
dof_damping) and per-entity 2D fields (geom_friction, actuator_gear). Entities
may be addressed by integer index or by **name** (e.g. ``index="object0"``),
resolved via ``mujoco.mj_name2id`` -- needed for manipulation tasks.
"""
from __future__ import annotations

import numpy as np

from robustrllib.adapters.base import DynamicsAdapter

# logical name -> (model attribute, kind, default component column)
#   vec      : 1D vector, addressed by component (default col)
#   body/dof : 1D per-entity, addressed by entity index/name
#   geom/act : 2D per-entity, addressed by entity index/name, component col
_FIELDS = {
    "gravity": ("opt.gravity", "vec", 2),
    "wind": ("opt.wind", "vec", 0),
    "body_mass": ("body_mass", "body", None),
    "dof_damping": ("dof_damping", "dof", None),
    # Static joint resistance. On manipulation tasks the resistance of an object's
    # joint (a door latch, a cabinet slide) lives in frictionloss rather than
    # damping -- e.g. Adroit's `latch` has damping 0.05 but frictionloss 5.
    "dof_frictionloss": ("dof_frictionloss", "dof", None),
    "geom_friction": ("geom_friction", "geom", 0),
    "actuator_gear": ("actuator_gear", "actuator", 0),
    # morphology (e.g. torso/foot length): geom size & body offset components
    "geom_size": ("geom_size", "geom", 0),
    # Scale the complete compiled MuJoCo geom-size vector.  For capsules this
    # changes both radius and half-length, unlike ``geom_size`` above, which is
    # intentionally the historical radius-only axis.  This is useful for a
    # visibly interpretable morphology intervention and remains a distinct
    # logical parameter so existing evaluation grids do not silently change.
    "geom_size_xyz": ("geom_size", "geom", None),
    "body_pos": ("body_pos", "body", 2),
    "body_inertia": ("body_inertia", "body", 0),
    # Limb-length shift (morphology). Unlike ``body_pos`` above --
    # which touches only the z offset and is therefore a no-op on Ant, whose legs
    # extend in x/y -- this scales the full 3-vector, so it lengthens/shortens
    # every limb regardless of the body's orientation.
    "body_pos_xyz": ("body_pos", "body", None),
    # --- appearance / sensing fields -----------------------------------------
    # These are mjModel fields like any other, so they need no new adapter
    # machinery -- only an entry here and, for the named kinds, an _OBJ row.
    # They are perturbed through target: appearance (AppearanceShift), NOT
    # through target: dynamics, because they change what the policy SEES rather
    # than how the world moves; keeping the two targets separate is what lets a
    # results table say which axis a method is fragile to.
    "geom_rgba": ("geom_rgba", "geom", None),      # per-geom RGBA (full 4-vec)
    "mat_rgba": ("mat_rgba", "material", None),    # per-material RGBA
    "light_diffuse": ("light_diffuse", "light", None),
    "light_ambient": ("light_ambient", "light", None),
    "light_pos": ("light_pos", "light", None),
    "cam_pos": ("cam_pos", "camera", None),
    "cam_quat": ("cam_quat", "camera", None),
    "cam_fovy": ("cam_fovy", "camera", None),
    # A free-floating object's pose is not in model.body_pos (that stays at the
    # origin); it lives in the free joint's qpos, which a scene's placement
    # sampler writes at reset. This addresses data.qpos through the body's free
    # joint.
    "freejoint_pos": ("__freejoint__", "body", None),
}

# fields that are 2D per-entity even though their kind is body/geom
_2D = {"geom_size", "body_pos", "body_inertia"}
# appearance fields whose per-entity value is a full vector scaled as a whole
_APPEARANCE_VEC = {"geom_rgba", "mat_rgba", "light_diffuse", "light_ambient",
                   "light_pos", "cam_pos", "cam_quat"}
# Edits to these invalidate mjModel's cached derived constants (subtree masses,
# inverse inertias, reference lengths), so mj_setConst must run afterwards.
_DERIVED_DEPENDENT = {"body_mass", "body_inertia", "body_pos", "body_pos_xyz",
                      "geom_size", "geom_size_xyz"}
# fields whose per-entity value is a full vector to be scaled as a whole
_VEC_PER_ENTITY = {"body_pos_xyz", "geom_size_xyz"} | _APPEARANCE_VEC

# entity kind -> mujoco object type for name resolution
_OBJ = {"body": "mjOBJ_BODY", "geom": "mjOBJ_GEOM", "actuator": "mjOBJ_ACTUATOR", "dof": "mjOBJ_JOINT",
        "light": "mjOBJ_LIGHT", "camera": "mjOBJ_CAMERA", "material": "mjOBJ_MATERIAL"}


def _resolve_attr(model, attr):
    obj = model
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class MujocoDynamicsAdapter(DynamicsAdapter):
    def __init__(self, env):
        self._env = env
        self._model = env.model
        self._nominal: dict = {}
        self._needs_setconst = False

    def _uses_mocap_control_gain(self, param):
        """Map logical actuator gear to Fetch's real control mechanism.

        FetchReach has no torque/position actuators: its Cartesian action moves
        a welded mocap target, and ``model.actuator_gear`` is empty. The benchmark
        vocabulary remains ``actuator_gear`` (control authority), while the task
        backend scales its explicit Cartesian command gain.
        """
        return (
            param == "actuator_gear"
            and self._model.nu == 0
            and hasattr(self._env, "position_action_scale")
        )

    @staticmethod
    def _validate_mocap_control_index(index):
        if index not in (None, "all"):
            raise ValueError(
                "mocap control gain is a global Cartesian scale; "
                f"expected index=None/'all', got {index!r}"
            )

    def list_params(self):
        return tuple(_FIELDS)

    def _name2id(self, kind, name):
        import mujoco

        idx = mujoco.mj_name2id(self._model, getattr(mujoco.mjtObj, _OBJ[kind]), name)
        if idx < 0:
            raise ValueError(f"no {kind} named {name!r} in this model")
        if kind == "dof":  # joint name -> first dof address
            return int(self._model.jnt_dofadr[idx])
        return idx

    def _freejoint_slot(self, index):
        """(qpos array, slice) for the free joint of body ``index``."""
        import mujoco

        model = getattr(self._model, "_model", self._model)
        bid = self._name2id("body", index) if isinstance(index, str) else int(index)
        jadr, jnum = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
        for j in range(jadr, jadr + jnum):
            if int(model.jnt_type[j]) == 0:          # mjJNT_FREE
                a = int(model.jnt_qposadr[j])
                data = self._env.data
                return data.qpos, slice(a, a + 3)
        raise ValueError(
            f"body {index!r} has no free joint, so it has no free pose to shift; "
            "use body_pos for a body welded to its parent")

    def _locate(self, param, index):
        if param == "freejoint_pos":
            return self._freejoint_slot(index)
        """Return (array, key) where ``array[key]`` is the perturbed value.

        ``index="all"`` selects every entity (global scaling, e.g. all geoms'
        friction), matching the morphology shifts of the benchmark.
        """
        attr, kind, col = _FIELDS[param]
        arr = _resolve_attr(self._model, attr)
        if param in _VEC_PER_ENTITY:  # scale the entity's whole 3-vector
            if index == "all":
                return arr, slice(None)
            if isinstance(index, str):
                index = self._name2id(kind, index)
            return arr, int(index)
        two_d = kind in ("geom", "actuator") or param in _2D
        if index == "all":
            return arr, ((slice(None), col) if two_d else slice(None))
        if kind == "vec":
            return arr, (col if index is None else index)
        if isinstance(index, str):
            index = self._name2id(kind, index)
        index = 0 if index is None else int(index)
        return arr, ((index, col) if two_d else index)

    def read(self, param, index=None):
        """The parameter's CURRENT value in the live model -- not the cached
        nominal. Differs from get_nominal whenever the env itself rewrites the
        field on reset (Adroit Door randomises body_pos[frame] every episode);
        a translate shift must add its offset to this, not to the XML default.
        """
        if self._uses_mocap_control_gain(param):
            self._validate_mocap_control_index(index)
            return float(self._env.position_action_scale)
        arr, k = self._locate(param, index)
        val = arr[k]
        return np.array(val, copy=True) if np.ndim(val) else float(val)

    def get_nominal(self, param, index=None):
        key = (param, index if not isinstance(index, np.ndarray) else tuple(index))
        if key not in self._nominal:
            if self._uses_mocap_control_gain(param):
                self._validate_mocap_control_index(index)
                val = float(self._env.position_action_scale)
            else:
                if param == "actuator_gear" and self._model.nu == 0:
                    raise ValueError(
                        "model has no actuators and the environment exposes no "
                        "mocap control gain; refusing a silent actuator_gear no-op"
                    )
                arr, k = self._locate(param, index)
                val = arr[k]
            self._nominal[key] = np.array(val, copy=True) if np.ndim(val) else float(val)
        return self._nominal[key]

    def _recompute_derived(self):
        """Propagate a model edit into MuJoCo's derived constants.

        `mjModel` caches quantities computed from the primary fields --
        `body_subtreemass`, `body_invweight0`, `dof_invweight0`, `dof_M0`, and
        the reference lengths. Writing `body_mass` or `body_pos` alone leaves
        those stale, so the solver would keep using the unperturbed inertia
        scaling. MuJoCo's documentation requires `mj_setConst` after runtime edits
        to mass, inertia or body frames.
        """
        if not self._needs_setconst:
            return
        import mujoco

        # mj_setConst uses MjData as workspace and overwrites integration state
        # (most visibly qpos). Shifts are applied right after env.reset has
        # sampled the initial state, so calling it directly would make the reset
        # observation describe one state while the simulator advances another.
        # Preserve the complete integration state rather than a hand-picked
        # qpos/qvel subset, then forward the restored state under the edited model.
        data = self._env.data
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self._model, state_spec), dtype=np.float64)
        mujoco.mj_getState(self._model, data, state, state_spec)

        mujoco.mj_setConst(self._model, data)
        mujoco.mj_setState(self._model, data, state, state_spec)
        mujoco.mj_forward(self._model, data)
        self._needs_setconst = False

    def set_param(self, param, value, index=None):
        self.get_nominal(param, index)  # cache nominal before first write
        if self._uses_mocap_control_gain(param):
            self._validate_mocap_control_index(index)
            self._env.position_action_scale = float(value)
            return
        arr, k = self._locate(param, index)
        arr[k] = value
        if param in _DERIVED_DEPENDENT:
            self._needs_setconst = True
        self._recompute_derived()

    def reset_all(self):
        for (param, index), nominal in self._nominal.items():
            if self._uses_mocap_control_gain(param):
                self._validate_mocap_control_index(index)
                self._env.position_action_scale = float(nominal)
                continue
            arr, k = self._locate(param, index)
            arr[k] = nominal
            if param in _DERIVED_DEPENDENT:
                self._needs_setconst = True
        self._recompute_derived()

    def inject_state_noise(self, rng, q_sigma, v_sigma, scale=1.0):
        """Add process noise to the live simulator state (perturbs the transition).

        Writing into ``qpos``/``qvel`` via ``set_state`` makes the *next*
        transition start from a perturbed state, i.e. a stochastic transition
        kernel -- distinct from observation noise, which only corrupts the
        returned obs.
        """
        m = self._env
        qpos = m.data.qpos.copy()
        qvel = m.data.qvel.copy()
        if q_sigma:
            qpos += rng.normal(0.0, q_sigma * scale, size=m.model.nq)
        if v_sigma:
            qvel += rng.normal(0.0, v_sigma * scale, size=m.model.nv)
        m.set_state(qpos, qvel)
