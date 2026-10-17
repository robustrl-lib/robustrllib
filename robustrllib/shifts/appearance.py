"""Appearance shift: perturb what the policy SEES, not how the world moves.

Motivation. Every shift target that existed before this one moves the physics
(gravity, mass, gear, friction) or corrupts a state vector. A vision-language
policy can be perfectly robust to all of those and still fail the moment a
target object changes colour, because its grounding of the instruction was never
tested. This wrapper adds that axis.

It reuses the MuJoCo adapter unchanged -- `geom_rgba`, `mat_rgba`, `light_*` and
`cam_*` are ordinary mjModel fields, resolved by the same getattr/name2id path
as `body_mass`. What is new is the set of OPERATIONS, because "scale the colour
by 1.2" is not a meaningful perturbation:

  hue   : rotate the RGB triple by `degrees` in HSV space, preserving
          saturation and value. This is the perturbation a human would call
          "the same object in a different colour".
  swap  : exchange the colour of two named geoms. Use it when the instruction
          names a colour ("pick up the red block"): swapping target and
          distractor colours moves the REFERENT, so a policy that grounds the
          language must follow the colour while one that memorised a position
          or a shape shortcut will not. Nominal performance does not predict
          this axis at all, which is the whole point of reporting it.
  scale / set / gauss / uniform :
          delegated to the ordinary dynamics path, for lights and cameras where
          multiplicative or absolute perturbation IS meaningful.

`swap` is deliberately NOT expressible as a dynamics shift: it is a two-entity
operation, and it changes the meaning of the instruction rather than the value
of a parameter.
"""
from __future__ import annotations

import colorsys

import numpy as np

from robustrllib.adapters import get_adapter
from robustrllib.shifts.base import ShiftWrapper
from robustrllib.shifts.dynamics import apply_dynamics

def _rgba_slot(model, geom_id):
    """Where a geom's colour actually lives.

    Always ``geom_rgba``. MuJoCo multiplies the geom colour by its material's
    ``mat_rgba``, which is usually a white multiplier (on Hopper-v5 every geom has
    a material with ``mat_rgba`` = [1, 1, 1, 1] and the visible colour is
    ``geom_rgba``), so editing ``geom_rgba`` tints the geom even under a material.
    ``_assert_effective`` raises when a colour edit would change nothing.
    """
    return model.geom_rgba, geom_id


def _assert_effective(tag, before, after):
    """Refuse to pretend a perturbation happened when the pixels cannot move.

    A hue rotation of an achromatic colour (white/grey/black) is mathematically
    the identity, and an alpha-only or zero-degree edit leaves the render
    untouched. A silent no-op would hide a wrong name, so it is an error here.
    """
    if np.allclose(before[:3], after[:3], atol=1e-8):
        r, g, b = (float(c) for c in before[:3])
        s = colorsys.rgb_to_hsv(r, g, b)[1]
        raise ValueError(
            f"appearance shift on {tag!r} changed nothing: rgb {np.round(before[:3], 4).tolist()} "
            f"-> {np.round(after[:3], 4).tolist()} (saturation {s:.4f}). "
            "A hue rotation cannot move an achromatic colour; give the geom a "
            "saturated nominal colour, target a different geom, or use mode "
            "'set'/'scale' instead of 'hue'."
        )


def _tint(rgba, degrees, saturation, value):
    """Absolute colour for a geom whose nominal rgba is an achromatic multiplier.

    When an object's visible colour comes from its material texture, its
    geom_rgba is grey (saturation 0) and multiplies the texture; rotating the hue
    of grey changes nothing. Setting geom_rgba to a saturated colour instead tints
    the rendered object while keeping its shading. Use `hue` when the geom's own
    rgba carries the colour (MuJoCo locomotion) and `tint` when a texture does
    (most manipulation assets).
    """
    r, g, b = colorsys.hsv_to_rgb((degrees / 360.0) % 1.0, saturation, value)
    out = np.array(rgba, dtype=np.float64, copy=True)
    out[:3] = (r, g, b)
    return out


def _rotate_hue(rgba, degrees):
    r, g, b = (float(c) for c in rgba[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    out = np.array(rgba, dtype=np.float64, copy=True)
    out[:3] = (r2, g2, b2)
    return out


class AppearanceShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        self._adapter = get_adapter(env)
        self._mode = spec.mode
        self._nonstationary = spec.schedule is not None
        self._nominal_rgba: dict = {}

    # -- helpers ----------------------------------------------------------
    def _model(self):
        return self._adapter._model

    def _geom_id(self, name):
        return self._adapter._name2id("geom", name) if isinstance(name, str) else int(name)

    def _resolve_geoms(self, p):
        """Geom ids named by this spec, by explicit name or by prefix.

        `geom_prefix` names an object rather than its current geoms, so a grid
        stays valid when a scene draws a different object asset (with a
        different number of geoms) at every reset.
        """
        model = self._model()
        if p.get("geom_prefix"):
            pref = p["geom_prefix"]
            prefs = [pref] if isinstance(pref, str) else list(pref)
            ids = []
            for i in range(model.ngeom):
                try:
                    n = model.geom_id2name(i)
                except Exception:
                    n = None
                if n and any(n.startswith(q) for q in prefs) and model.geom_rgba[i][3] > 0.9:
                    ids.append((n, i))
            if not ids:
                raise ValueError(
                    f"appearance shift matched no visible geom for prefix {prefs!r}; "
                    "a silent empty match would perturb nothing")
            return ids
        names = p["geoms"] if isinstance(p.get("geoms"), (list, tuple)) else [p["geom"]]
        return [(n, self._geom_id(n)) for n in names]

    def _cache_nominal(self, arr, key, tag):
        if tag not in self._nominal_rgba:
            self._nominal_rgba[tag] = np.array(arr[key], copy=True)
        return self._nominal_rgba[tag]

    # -- operations -------------------------------------------------------
    def _apply_hue(self):
        model, p = self._model(), self._spec.params
        degrees = float(p.get("degrees", 0.0)) * self._scale
        for name, gid in self._resolve_geoms(p):
            arr, key = _rgba_slot(model, gid)
            nominal = self._cache_nominal(arr, key, ("hue", str(name)))
            arr[key] = _rotate_hue(nominal, degrees)
            if degrees % 360 != 0:
                _assert_effective(name, nominal, arr[key])

    def _apply_swap(self):
        model, p = self._model(), self._spec.params
        a, b = self._geom_id(p["geom_a"]), self._geom_id(p["geom_b"])
        arr_a, key_a = _rgba_slot(model, a)
        arr_b, key_b = _rgba_slot(model, b)
        nom_a = self._cache_nominal(arr_a, key_a, ("swap", "a"))
        nom_b = self._cache_nominal(arr_b, key_b, ("swap", "b"))
        # `_scale` lets a non-stationary schedule interpolate the exchange
        # instead of toggling it, so the axis is graded like every other one.
        w = float(self._scale)
        arr_a[key_a] = (1.0 - w) * nom_a + w * nom_b
        arr_b[key_b] = (1.0 - w) * nom_b + w * nom_a
        if w != 0.0:
            _assert_effective(f"{p['geom_a']}<->{p['geom_b']}", nom_a, arr_a[key_a])

    def _apply_tint(self):
        model, p = self._model(), self._spec.params
        degrees = float(p.get("degrees", 0.0))
        sat = float(p.get("saturation", 0.85))
        val = float(p.get("value", 1.0))
        w = float(self._scale)
        for name, gid in self._resolve_geoms(p):
            arr, key = _rgba_slot(model, gid)
            nominal = self._cache_nominal(arr, key, ("tint", str(name)))
            target = _tint(nominal, degrees, sat, val)
            # `_scale` interpolates nominal -> target so a schedule grades the
            # axis instead of toggling it, as for every other shift here.
            arr[key] = (1.0 - w) * nominal + w * target
            if w != 0.0:
                _assert_effective(name, nominal, arr[key])

    def _apply(self):
        if self._mode == "tint":
            self._apply_tint()
        elif self._mode == "hue":
            self._apply_hue()
        elif self._mode == "swap":
            self._apply_swap()
        else:
            apply_dynamics(self._adapter, self._spec, self._scale, self._rng)

    # -- gym API ----------------------------------------------------------
    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._apply()
        return out

    def step(self, action):
        if self._nonstationary:
            self._apply()
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        return obs, reward, term, trunc, info
