"""Non-stationary intensity schedules.

A schedule maps an integer timestep ``t`` to a scalar multiplier applied to a
shift's nominal intensity, turning any stationary shift into a non-stationary
one without special-casing it inside the environment.
"""
from __future__ import annotations

import math
from typing import Any


def make_schedule(spec: dict[str, Any] | None):
    """Return ``f(t) -> float`` from a schedule dict. ``None`` -> constant 1.0."""
    if spec is None:
        return lambda t: 1.0

    kind = spec.get("type", "constant")
    start = float(spec.get("start", 0.0))
    end = float(spec.get("end", 1.0))
    t0 = float(spec.get("t0", 0))
    t1 = float(spec.get("t1", 1))
    period = float(spec.get("period", 1000))

    def frac(t):
        return min(max((t - t0) / max(t1 - t0, 1e-8), 0.0), 1.0)

    if kind == "constant":
        return lambda t: start
    if kind == "linear":
        return lambda t: start + (end - start) * frac(t)
    if kind == "step":
        return lambda t: start if t < t1 else end
    if kind == "sine":
        return lambda t: start + (end - start) * 0.5 * (1 + math.sin(2 * math.pi * t / period))
    raise ValueError(f"unknown schedule type {kind!r}")
