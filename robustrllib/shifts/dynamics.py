"""Dynamics / task shift via a DynamicsAdapter.

By default the shift is *episodic*: a fresh parameter value is drawn at each
``reset`` and held for the episode. With a non-stationary ``schedule`` the value
is re-applied every step using the schedule multiplier.
"""
from __future__ import annotations

import inspect

import numpy as np

from robustrllib.adapters import get_adapter
from robustrllib.shifts.base import ShiftWrapper


def _takes_no_arguments(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


def draw_dynamics_factor(spec, rng) -> float:
    """The per-reset multiplier for a dynamics shift (shared by single-agent and
    multi-agent dynamics wrappers). ``set`` mode returns None (absolute path)."""
    p, mode = spec.params, spec.mode
    if mode == "scale":
        return p.get("factor", 1.0)
    if mode == "set":
        return None
    if mode == "gauss":
        return float(rng.normal(p.get("mu", 1.0), p.get("sigma", 0.1)))
    if mode == "uniform":
        return float(rng.uniform(p.get("low", 0.8), p.get("high", 1.2)))
    if mode == "loguniform":
        # exp(U(log lo, log hi)). The sim-to-real papers log-sample every
        # multiplicative physical parameter (Peng et al. 2018 Table I: link mass
        # over [0.25, 4]x, joint damping over [0.2, 20]x) and the reason is not
        # cosmetic: a plain uniform over [0.2, 20] puts 99% of its mass above
        # 1.0, so "randomize damping by 0.2-20x" would in practice only ever
        # make the joints stiffer. Log-sampling makes halving and doubling
        # equally likely, which is what a multiplicative range is meant to say.
        lo, hi = float(p.get("low", 0.8)), float(p.get("high", 1.2))
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(
                f"loguniform needs a strictly positive range, got [{lo}, {hi}]. "
                "For a parameter whose nominal is zero (PointMaze frictionloss, "
                "LunarLander wind) use uniform with operation: set instead.")
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    raise ValueError(f"dynamics shift does not support mode {mode!r}")


def apply_dynamics(adapter, spec, factor_scale, rng, translate_base=None):
    """Apply a dynamics ShiftSpec to a model via its adapter.

    ``translate_base`` is the position a translate offset is added to.
    DynamicsShift snapshots it once per episode, AFTER the inner env's reset
    has run, so a translate means "wherever the env put the body this episode,
    plus offset". None falls back to the cached XML nominal, which is only
    right for bodies the env never moves on reset.
    """
    param, index = spec.params["param"], spec.params.get("index")
    if spec.mode == "translate":
        # Additive offset on a position vector. Scaling cannot express "move the
        # mug 5 cm to the left": body_pos is measured from the parent frame's
        # origin, so a multiplicative factor moves an entity RADIALLY by an
        # amount that depends on how far it already sits from that origin --
        # the same factor is 1 cm for one object and 20 cm for another. The
        # visual-shift grid needs a metric offset, so this mode adds one.
        #
        # The base has to be the env's OWN pose for this episode, not the XML
        # default. Adroit Door randomises body_pos[frame] in reset_model;
        # adding the offset to the XML nominal replaced that sample with a
        # fixed pose ~30 cm off the training distribution, so a "+2 cm"
        # condition scored as an unsolvable task .
        # Re-adding to the same per-episode base also means a scheduled
        # per-step _apply cannot accumulate.
        offset = np.asarray(spec.params["offset"], dtype=np.float64) * factor_scale
        base = translate_base
        if base is None:
            base = adapter.get_nominal(param, index=index)
        adapter.apply(param, value=np.asarray(base, dtype=np.float64) + offset, index=index)
        return
    if spec.mode == "set":
        adapter.apply(param, value=spec.params["value"], index=index)
    elif spec.params.get("operation", "scale") == "set":
        # Absolute randomization is required when nominal is zero (PointMaze
        # frictionloss and LunarLander wind). A multiplicative factor could
        # never move either parameter away from zero. Preserve schedule
        # semantics by interpolating the sampled value back toward nominal.
        sampled = draw_dynamics_factor(spec, rng)
        nominal = float(spec.params["nominal"])
        value = nominal + (sampled - nominal) * factor_scale
        adapter.apply(param, value=value, index=index)
    else:
        adapter.apply(param, factor=draw_dynamics_factor(spec, rng) * factor_scale, index=index)


class DynamicsShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        self._adapter = get_adapter(env)
        self._param = spec.params["param"]
        self._index = spec.params.get("index")
        self._nonstationary = spec.schedule is not None
        # Cache the *true* nominal eagerly, while the model is still pristine.
        # If left lazy, an inner DynamicsShift on the same param would modify the
        # model before this one's first read, so this wrapper would cache an
        # already-perturbed value as "nominal" and the two factors would cascade.
        # Box2D builds its world lazily at reset, so some params aren't yet
        # readable here -- fall back to lazy caching at first apply for those.
        if spec.mode != "set":
            try:
                self._adapter.get_nominal(self._param, index=self._index)
            except Exception:
                pass

    def _apply(self):
        apply_dynamics(self._adapter, self._spec, self._scale, self._rng,
                       translate_base=self._translate_base)

    _translate_base = None

    def reset(self, **kwargs):
        if self._spec.mode == "translate":
            # Put the body back at its XML nominal BEFORE the env resets. An env
            # that randomises this body (Adroit Door's frame) overwrites it in
            # reset_model; one that never touches it (Hopper's foot) is now at
            # nominal again, so the offset never compounds across episodes.
            self._adapter.apply(
                self._param, index=self._index,
                value=np.asarray(self._adapter.get_nominal(self._param, index=self._index),
                                 dtype=np.float64))
        out = super().reset(**kwargs)  # resets clock; model already rebuilt by env
        if self._spec.mode == "translate":
            # Snapshot the pose the env itself set for this episode (after its
            # own reset_model randomisation) so the offset is relative to it.
            # adapter.read is the live value; get_nominal is the cached XML
            # default. There is no fallback: an adapter without read cannot
            # translate relative to the episode's own pose.
            self._translate_base = np.array(
                self._adapter.read(self._param, index=self._index),
                dtype=np.float64, copy=True)
        self._apply()
        # The inner reset computed its observation before the shift was applied.
        # Recompute it when the env can (MuJoCo envs expose an argument-free
        # _get_obs, and the adapter has already run mj_forward), so that the first
        # observation describes the shifted world, e.g. a translated door handle.
        get_obs = getattr(self.env.unwrapped, "_get_obs", None)
        if callable(get_obs) and _takes_no_arguments(get_obs) and isinstance(out, tuple) and len(out) == 2:
            out = (get_obs(), out[1])
        return out

    def step(self, action):
        if self._nonstationary:
            self._apply()
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        return obs, reward, term, trunc, info
