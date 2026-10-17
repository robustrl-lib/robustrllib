"""Schedule semantics of dynamics shifts.

`DynamicsShift.step` re-applies the shift every step when a schedule is set,
and `apply_dynamics` multiplies the drawn factor by the schedule value. Two
consequences:

1. `scale` + schedule: the live value is nominal * factor * schedule(t). With
   factor 1.0 the schedule's start/end ARE the factor range, so a sine with
   start 0.8 / end 1.2 sweeps gear between 0.8x and 1.2x.
2. `uniform` / `loguniform` + schedule REDRAWS a fresh random factor every
   step (draw_dynamics_factor is called inside _apply). That is per-step
   process noise, not a slowly drifting parameter; the evaluation grids do not
   combine a stochastic dynamics mode with a schedule.
"""
import math

import numpy as np

from robustrllib import ShiftSpec, make_robust


def _gravity(env):
    return float(env.unwrapped.model.opt.gravity[2])


def test_scale_with_step_schedule_switches_at_t1():
    env = make_robust("Hopper-v5", shifts=[ShiftSpec(
        "dynamics", "scale", {"param": "gravity", "factor": 1.0},
        schedule={"type": "step", "start": 1.0, "end": 0.8, "t1": 3})], seed=0)
    env.reset(seed=0)
    g0 = _gravity(env)
    seen = []
    for _ in range(5):
        env.step(env.action_space.sample())
        seen.append(_gravity(env) / g0)
    env.close()
    # step k applies schedule(t=k) before the physics step: t=0,1,2 -> start; t>=3 -> end
    assert np.allclose(seen[:3], 1.0), seen
    assert np.allclose(seen[3:], 0.8), seen


def test_scale_with_sine_schedule_tracks_formula():
    period = 8
    env = make_robust("Hopper-v5", shifts=[ShiftSpec(
        "dynamics", "scale", {"param": "actuator_gear", "index": "all", "factor": 1.0},
        schedule={"type": "sine", "start": 0.8, "end": 1.2, "period": period})], seed=0)
    env.reset(seed=0)
    nominal = env.unwrapped.model.actuator_gear[:, 0].copy()
    for t in range(period):
        env.step(env.action_space.sample())
        expect = 0.8 + 0.4 * 0.5 * (1 + math.sin(2 * math.pi * t / period))
        got = env.unwrapped.model.actuator_gear[:, 0] / nominal
        assert np.allclose(got, expect, atol=1e-9), (t, got, expect)
    env.close()


def test_stochastic_mode_plus_schedule_redraws_every_step():
    """Documented limitation: loguniform under a schedule is per-step noise."""
    env = make_robust("Hopper-v5", shifts=[ShiftSpec(
        "dynamics", "loguniform", {"param": "gravity", "low": 0.5, "high": 2.0},
        schedule={"type": "constant", "start": 1.0})], seed=0)
    env.reset(seed=0)
    vals = []
    for _ in range(6):
        env.step(env.action_space.sample())
        vals.append(_gravity(env))
    env.close()
    assert len(set(np.round(vals, 9))) > 1, vals


def test_stochastic_mode_without_schedule_is_fixed_within_episode():
    env = make_robust("Hopper-v5", shifts=[ShiftSpec(
        "dynamics", "loguniform", {"param": "gravity", "low": 0.5, "high": 2.0})], seed=0)
    env.reset(seed=0)
    g = _gravity(env)
    for _ in range(6):
        env.step(env.action_space.sample())
        assert _gravity(env) == g
    env.reset(seed=1)
    assert _gravity(env) != g  # redrawn per episode
    env.close()


def test_paired_seeds_give_identical_draws_across_wrapper_stacks():
    """reset(seed=k) must reseed the shift RNG so two conditions that share a
    stochastic spec see the same parameter sample for the same episode."""
    spec = ShiftSpec("dynamics", "loguniform", {"param": "body_mass", "index": "all",
                                               "low": 0.6, "high": 1.5})
    a = make_robust("Hopper-v5", shifts=[spec], seed=0)
    b = make_robust("Hopper-v5", shifts=[spec, ShiftSpec("observation", "gauss", {"sigma": 0.1})],
                    seed=123)
    a.reset(seed=7); b.reset(seed=7)
    assert np.allclose(a.unwrapped.model.body_mass, b.unwrapped.model.body_mass)
    a.close(); b.close()
