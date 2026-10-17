"""Tests for dynamics adapters and the transition shift.

Box2D logic is tested with a mock (the box2d extra is often absent); MuJoCo /
maze tests skip if those env extras are unavailable.
"""

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust
from robustrllib.adapters.box2d import Box2DDynamicsAdapter


# --- Box2D adapter (mock, no box2d dependency) -------------------------------
class _FX:
    def __init__(self):
        self.friction = 0.5


class _Body:
    def __init__(self):
        self.fixtures = [_FX()]


class _World:
    def __init__(self):
        self.gravity = [0.0, -10.0]
        self.bodies = [_Body(), _Body()]


class _FakeLunar:
    def __init__(self):
        self.gravity = -10.0
        self.world = _World()
        self.wind_power = 15.0
        self.turbulence_power = 1.5


def test_box2d_scale_set_and_reset():
    e = _FakeLunar()
    a = Box2DDynamicsAdapter(e)
    a.apply("gravity", factor=1.5)
    assert e.world.gravity[1] == -15.0 and e.gravity == -15.0
    a.apply("wind_power", value=18.0)
    assert e.wind_power == 18.0
    a.apply("friction", factor=2.0)
    assert all(fx.friction == 1.0 for b in e.world.bodies for fx in b.fixtures)
    a.apply("gravity", factor=1.5)  # no drift: nominal cached
    assert e.world.gravity[1] == -15.0
    a.reset_all()
    assert e.world.gravity[1] == -10.0 and e.wind_power == 15.0


# --- MuJoCo named + 2D field resolution --------------------------------------
def _mk(shifts, env_id="Hopper-v5"):
    try:
        env = make_robust(env_id, shifts=shifts, seed=0)
        env.reset(seed=0)
        return env
    except Exception as e:  # pragma: no cover
        pytest.skip(f"{env_id} unavailable: {e}")


def test_mujoco_named_body_mass():
    import mujoco

    base = _mk([])
    bid = mujoco.mj_name2id(base.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    nominal = float(base.unwrapped.model.body_mass[bid])
    env = _mk([ShiftSpec("dynamics", "scale", {"param": "body_mass", "index": "torso", "factor": 2.0})])
    assert np.isclose(env.unwrapped.model.body_mass[bid], 2.0 * nominal)


def test_mujoco_2d_fields():
    env = _mk([ShiftSpec("dynamics", "scale", {"param": "geom_friction", "index": 0, "factor": 2.0})])
    assert env.unwrapped.model.geom_friction[0, 0] > 0


# --- Maze: raw-action bridge + physics + transition noise --------------------
MAZE = "PointMaze_UMaze-v3"


def _maze(shifts):
    try:
        return make_robust(MAZE, shifts=shifts, seed=0)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"{MAZE} unavailable: {e}")


def test_maze_steps_without_dict_error():
    env = _maze([ShiftSpec("dynamics", "scale", {"param": "dof_damping", "index": 0, "factor": 3.0})])
    env.reset(seed=0)
    env.action_space.seed(0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(obs, dict)  # GoalEnv
    assert np.isclose(env.unwrapped.point_env.model.dof_damping[0], 3.0)


def test_dict_observation_shift_preserves_goal():
    base = _maze([]).reset(seed=0)[0]
    obs = _maze([ShiftSpec("observation", "gauss", {"sigma": 0.1})]).reset(seed=0)[0]
    assert not np.allclose(obs["observation"], base["observation"])  # state perturbed
    assert np.allclose(obs["desired_goal"], base["desired_goal"])    # goal left intact


def test_morphology_geom_size_by_name():
    import mujoco

    base = _mk([])
    gid = mujoco.mj_name2id(base.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
    nominal = float(base.unwrapped.model.geom_size[gid, 0])
    env = _mk([ShiftSpec("dynamics", "scale", {"param": "geom_size", "index": "foot_geom", "factor": 1.5})])
    assert np.isclose(env.unwrapped.model.geom_size[gid, 0], 1.5 * nominal)


def test_transition_noise_changes_and_reproduces():
    def traj(shifts):
        e = _maze(shifts)
        e.reset(seed=0)
        e.action_space.seed(0)
        xs = []
        for _ in range(12):
            e.step(e.action_space.sample())
            xs.append(e.unwrapped.point_env.data.qpos.copy())
        return np.stack(xs)

    base = traj([])
    noisy = traj([ShiftSpec("transition", "gauss", {"q_sigma": 0.05, "v_sigma": 0.05})])
    noisy2 = traj([ShiftSpec("transition", "gauss", {"q_sigma": 0.05, "v_sigma": 0.05})])
    assert not np.allclose(base, noisy)      # transition kernel actually perturbed
    assert np.allclose(noisy, noisy2)        # reproducible
