"""robosuite DoorCausal through the shift pipeline.

DoorCausal builds without a renderer, so it runs with ``MUJOCO_GL=disable``.
Tests skip if robosuite's dependencies are unavailable.
"""
import math

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust


def _make(env_id="DoorCausalHHL-v0", shifts=(), seed=0):
    try:
        return make_robust(env_id, shifts=list(shifts), seed=seed)
    except Exception as e:  # pragma: no cover - robosuite dependencies missing
        pytest.skip(f"{env_id} unavailable: {e}")


def _door_state(env):
    """(door rotated left, latch high) of the current episode."""
    rs = env.unwrapped._env.env
    model = rs.sim.model
    qw, _, _, qz = model.body_quat[model.body_name2id(rs.door.root_body)]
    yaw = 2.0 * math.atan2(qz, qw)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    latch_z = model.body_pos[model.body_name2id("Door_latch")][2]
    return yaw < -math.pi / 2, latch_z > 0


def test_stacked_shifts_and_dynamics():
    env = _make(shifts=[
        ShiftSpec("observation", "gauss", {"sigma": 0.02}),
        ShiftSpec("action", "gauss", {"sigma": 0.02}),
        ShiftSpec("reward", "gauss", {"sigma": 0.01}),
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
    ])
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1
    assert np.isclose(env.unwrapped.sim.model.opt.gravity[2], -9.81 * 1.3, atol=1e-2)
    for _ in range(3):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and isinstance(term, bool)
    env.close()


def test_reproducible():
    def roll():
        env = _make(shifts=[ShiftSpec("observation", "gauss", {"sigma": 0.02}),
                            ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3})])
        env.action_space.seed(0)
        obs, _ = env.reset(seed=0)
        trace = [obs]
        for _ in range(6):
            obs, *_ = env.step(env.action_space.sample())
            trace.append(obs)
        env.close()
        return np.stack(trace)

    assert np.allclose(roll(), roll())


@pytest.mark.parametrize("env_id,high_when_left", [("DoorCausalHHL-v0", True), ("DoorCausalHHR-v0", False)])
def test_binding_of_rotation_and_latch(env_id, high_when_left):
    env = _make(env_id)
    rotations = set()
    for seed in range(6):
        env.reset(seed=seed)
        left, high = _door_state(env)
        rotations.add(left)
        assert high == (left if high_when_left else not left), (env_id, seed)
    assert rotations == {True, False}
    env.close()
