"""Settings of the perturbation branches inside the environment code, all switched off.

Several environments carry perturbation branches of their own, which read a settings
object (``args.noise_factor``, ``args.noise_type``, ...). The library applies every
shift as a wrapper (``robustrllib.make_robust``), so these settings select no branch
and the environment steps as the simulator does.
"""
from __future__ import annotations

from types import SimpleNamespace

_INERT = dict(
    noise_factor="none",
    noise_type="none",
    noise_mu=0.0,
    noise_sigma=0.0,
    noise_shift=0.0,
    env_robosuite_robust="none",
)


def inert_args(**overrides) -> SimpleNamespace:
    """A new settings object under which no in-environment perturbation branch runs."""
    values = dict(_INERT)
    values.update(overrides)
    return SimpleNamespace(**values)


__all__ = ["inert_args"]
