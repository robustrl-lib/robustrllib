"""Base class shared by all shift wrappers: seeded RNG + non-stationary clock."""
from __future__ import annotations

import hashlib

import numpy as np

import robustrllib.gym as gym
from robustrllib.schedule import make_schedule
from robustrllib.spec import ShiftSpec


def _stable_salt(spec: ShiftSpec) -> int:
    """A reproducible per-shift RNG salt covering the shift's full identity.

    The salt covers the target, the mode and the perturbed parameter and entity,
    so that stacked shifts (several `dynamics`/`uniform` axes of a domain
    randomisation, say) draw from independent streams; blake2b keeps it the same
    across processes and Python versions, unlike the built-in `hash()`.
    """
    p = spec.params
    identity = "|".join(str(x) for x in (
        spec.target,
        spec.mode,
        p.get("param", ""),
        p.get("index", ""),
    ))
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def sample_noise(rng: np.random.Generator, mode: str, params: dict, shape, scale: float):
    """Draw an additive perturbation of ``shape`` for the additive modes.

    ``scale`` is the schedule multiplier (1.0 when stationary).
    """
    if mode == "gauss":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 0.05) * scale
        return rng.normal(mu, sigma, size=shape)
    if mode == "shift":
        return np.full(shape, params.get("shift", params.get("value", 0.0)) * scale)
    if mode == "uniform":
        low = params.get("low", -params.get("range", 0.1))
        high = params.get("high", params.get("range", 0.1))
        return rng.uniform(low * scale, high * scale, size=shape)
    raise ValueError(f"mode {mode!r} is not an additive noise mode")


class ShiftWrapper(gym.Wrapper):
    """Common machinery: deterministic RNG and a per-episode timestep counter."""

    def __init__(self, env, spec: ShiftSpec, seed: int | None = None):
        super().__init__(env)
        self._spec = spec
        # Per-wrapper salt so stacked shifts get decorrelated (but deterministic)
        # streams from the same base seed. Must depend on the perturbed
        # parameter, not just (target, mode) -- see _stable_salt.
        self._salt = _stable_salt(spec)
        self._base_seed = spec.seed if spec.seed is not None else seed
        self._rng = np.random.default_rng(self._seed_seq(self._base_seed))
        self._schedule = make_schedule(spec.schedule)
        self._t = 0

    def _seed_seq(self, seed):
        return None if seed is None else np.random.SeedSequence([seed, self._salt])

    def reset(self, *, seed=None, **kwargs):
        self._t = 0
        # Mirror Gymnasium semantics: an explicit reset seed reseeds the
        # perturbation stream too, so whole-stack rollouts are reproducible.
        if seed is not None:
            self._base_seed = seed
            self._rng = np.random.default_rng(self._seed_seq(seed))
        return self.env.reset(seed=seed, **kwargs)

    @property
    def _scale(self) -> float:
        return self._schedule(self._t)
