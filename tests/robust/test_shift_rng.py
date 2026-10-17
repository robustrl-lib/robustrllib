"""Per-shift RNG identity.

Each shift draws from its own stream, salted by a stable hash of the shift: stacked
shifts do not share a stream, and the same seed gives the same stream in every process.
"""
import subprocess
import sys

import numpy as np

from robustrllib import ShiftSpec
from robustrllib.shifts.base import _stable_salt


def _dr_specs():
    return [
        ShiftSpec("dynamics", "uniform", {"param": "body_mass", "index": "all",
                                          "low": 0.9, "high": 1.1}),
        ShiftSpec("dynamics", "uniform", {"param": "actuator_gear", "index": "all",
                                          "low": 0.9, "high": 1.1}),
        ShiftSpec("dynamics", "uniform", {"param": "geom_friction", "index": "all",
                                          "low": 0.85, "high": 1.15}),
        ShiftSpec("dynamics", "uniform", {"param": "dof_damping", "index": "all",
                                          "low": 0.85, "high": 1.15}),
        ShiftSpec("dynamics", "uniform", {"param": "gravity", "low": 0.95, "high": 1.05}),
    ]


def _quantiles(specs, seed=0):
    """Where each axis's draw falls inside its own range, in [0, 1]."""
    out = []
    for spec in specs:
        rng = np.random.default_rng(np.random.SeedSequence([seed, _stable_salt(spec)]))
        low, high = spec.params["low"], spec.params["high"]
        out.append((rng.uniform(low, high) - low) / (high - low))
    return out


def test_axes_get_distinct_streams():
    specs = _dr_specs()
    assert len({_stable_salt(s) for s in specs}) == len(specs)
    # The real symptom: identical salts made every axis land on the same
    # quantile, so the joint distribution collapsed to one variable.
    q = _quantiles(specs)
    assert len({round(x, 9) for x in q}) == len(specs)


def test_same_param_different_entity_is_independent():
    a = ShiftSpec("dynamics", "uniform", {"param": "body_mass", "index": "torso",
                                          "low": 0.9, "high": 1.1})
    b = ShiftSpec("dynamics", "uniform", {"param": "body_mass", "index": "foot",
                                          "low": 0.9, "high": 1.1})
    assert _stable_salt(a) != _stable_salt(b)


def test_salt_is_stable_across_processes():
    """Same seed must give the same stream in a fresh interpreter.

    Run with an explicit, different PYTHONHASHSEED: the salt must not depend on
    Python's per-process hash randomization.
    """
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from robustrllib import ShiftSpec;"
        "from robustrllib.shifts.base import _stable_salt;"
        "print(_stable_salt(ShiftSpec('dynamics','uniform',"
        "{'param':'body_mass','index':'all','low':0.9,'high':1.1})))"
    )
    salts = set()
    for hashseed in ("1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"})
        salts.add(out.stdout.strip())
    assert len(salts) == 1, f"salt varies across processes: {salts}"
