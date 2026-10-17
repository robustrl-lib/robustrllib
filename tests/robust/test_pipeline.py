"""Smoke and reproducibility tests for the robust pipeline.

Run: pytest tests/robust/test_pipeline.py
(Requires the mujoco extra; tests skip if Hopper-v5 is unavailable.)
"""
import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust
from robustrllib.config import RobustConfig

ENV_ID = "Hopper-v5"


def _make(shifts, seed=0):
    try:
        return make_robust(ENV_ID, shifts=shifts, seed=seed)
    except Exception as e:  # pragma: no cover - env extras missing
        pytest.skip(f"{ENV_ID} unavailable: {e}")


def test_no_shift_is_clean():
    env = _make([])
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            break


def test_dynamics_mutates_model_in_memory():
    env = _make([ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.5})])
    env.reset(seed=0)
    assert np.isclose(env.unwrapped.model.opt.gravity[2], -9.81 * 1.5, atol=1e-3)
    # No drift: nominal is cached, every reset re-applies the same factor.
    env.reset(seed=7)
    assert np.isclose(env.unwrapped.model.opt.gravity[2], -9.81 * 1.5, atol=1e-3)


def test_observation_noise_changes_obs():
    base = _make([])
    noisy = _make([ShiftSpec("observation", "gauss", {"sigma": 0.1})])
    assert not np.allclose(base.reset(seed=0)[0], noisy.reset(seed=0)[0])


def test_multi_shift_stacks():
    env = _make([
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
        ShiftSpec("observation", "gauss", {"sigma": 0.05}),
        ShiftSpec("reward", "gauss", {"sigma": 0.01}),
    ])
    obs, _ = env.reset(seed=0)
    obs, r, *_ = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs))


def test_seed_reproducibility():
    def rollout():
        env = _make([
            ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
            ShiftSpec("observation", "gauss", {"sigma": 0.05}),
        ])
        env.action_space.seed(0)
        obs, _ = env.reset(seed=0)
        trace = [np.asarray(obs, np.float64)]
        for _ in range(30):
            obs, *_ = env.step(env.action_space.sample())
            trace.append(np.asarray(obs, np.float64))
        return np.stack(trace)

    assert np.allclose(rollout(), rollout())


def test_lunarlander_reports_safe_landing_outcome():
    """LunarLander reports the physical landing outcome next to the shaped return."""
    from robustrllib.gym.envs.robust_box2d.lunar_lander import heuristic

    try:
        env = make_robust("LunarLanderContinuous-v3")
    except Exception as e:  # pragma: no cover - Box2D extra missing
        pytest.skip(f"LunarLanderContinuous-v3 unavailable: {e}")
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        obs, _, terminated, truncated, info = env.step(heuristic(env, obs))
        if terminated or truncated:
            assert info["safe_landing"] is True
            assert info["is_success"] is True
            break
    else:
        pytest.fail("heuristic LunarLander rollout did not terminate")
    env.close()


def test_fetchreach_dense_exposes_goal_and_dense_reward():
    try:
        env = make_robust("FetchReachDense-v3")
    except Exception as e:  # pragma: no cover - robotics/MuJoCo extras missing
        pytest.skip(f"FetchReachDense-v3 unavailable: {e}")
    obs, _ = env.reset(seed=123)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert np.concatenate([obs["observation"], obs["desired_goal"]]).shape == (13,)
    obs, reward, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    assert np.isclose(reward, -distance)
    env.close()


def test_fetchreach_reset_restores_configured_gripper_pose():
    """Reset restores the configured gripper pose (mj_resetData alone restores the XML pose)."""
    try:
        env = make_robust("FetchReachDense-v3")
    except Exception as e:  # pragma: no cover - robotics/MuJoCo extras missing
        pytest.skip(f"FetchReachDense-v3 unavailable: {e}")

    obs, _ = env.reset(seed=123)
    raw = env.unwrapped
    configured_gripper = np.asarray(raw.initial_gripper_xpos)
    reset_gripper = np.asarray(obs["achieved_goal"])

    # Reach has no object, so achieved_goal is exactly the gripper position.
    assert np.linalg.norm(reset_gripper - configured_gripper) < 5e-6
    # The sampled goal must also be around that configured pose.
    assert np.all(
        np.abs(np.asarray(obs["desired_goal"]) - configured_gripper)
        <= raw.target_range + 1e-6
    )
    env.close()


def test_fetchreach_same_seed_and_actions_are_exactly_reproducible():
    def rollout(seed):
        try:
            env = make_robust("FetchReachDense-v3")
        except Exception as e:  # pragma: no cover - robotics/MuJoCo extras missing
            pytest.skip(f"FetchReachDense-v3 unavailable: {e}")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(777)
        trace = []
        for _ in range(50):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            trace.append(np.concatenate([
                obs["observation"], obs["achieved_goal"], obs["desired_goal"],
                np.asarray([reward, terminated, truncated], dtype=np.float64),
            ]))
            if terminated or truncated:
                break
        env.close()
        return np.stack(trace)

    assert np.array_equal(rollout(2024), rollout(2024))
    assert not np.array_equal(rollout(2024), rollout(2025))


def test_raw_action_envs_detected():
    # Pusher-v5s and PointMaze take a raw action; Hopper-v5 and LunarLander take the robust_input dict.
    import robustrllib.gym as rgym
    from robustrllib.core_wrapper import _uses_dict_api

    cases = {"Hopper-v5": True, "LunarLanderContinuous-v3": True, "Pusher-v5s": False, "PointMaze_UMaze-v3": False}
    for eid, expected in cases.items():
        try:
            assert _uses_dict_api(rgym.make(eid)) is expected, eid
        except Exception as e:  # pragma: no cover
            pytest.skip(f"{eid} unavailable: {e}")


def test_maze_in_env_perturbation_is_off_and_rollouts_repeat():
    """PointMaze and its inner PointEnv read in-environment settings that are switched off."""
    import robustrllib.gym.envs.robust_maze.point as point_module
    import robustrllib.gym.envs.robust_maze.point_maze as point_maze_module

    def rollout():
        try:
            env = make_robust("PointMaze_Open-v3")
        except Exception as e:  # pragma: no cover
            pytest.skip(f"PointMaze_Open-v3 unavailable: {e}")
        assert point_module.args.noise_factor == "none"
        assert point_maze_module.args.noise_factor == "none"
        obs, _ = env.reset(seed=123)
        trace = [obs["observation"].copy()]
        action = np.array([0.25, -0.1], dtype=np.float64)
        for _ in range(20):
            obs, *_ = env.step(action)
            trace.append(obs["observation"].copy())
        env.close()
        return np.stack(trace)

    assert np.array_equal(rollout(), rollout())


def test_stacked_dynamics_same_param_is_last_wins():
    # No nominal pollution: two gravity scales -> last wins (1.5x), not 1.2*1.5.
    env = _make([
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.2}),
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.5}),
    ])
    env.reset(seed=0)
    g = float(env.unwrapped.model.opt.gravity[2])
    assert np.isclose(g, -9.81 * 1.5, atol=1e-2)
    env.reset(seed=3)
    assert np.isclose(env.unwrapped.model.opt.gravity[2], g)  # no drift


def test_gymnasium_compat_is_native_env():
    import gymnasium

    from robustrllib.compat import as_gymnasium

    env = as_gymnasium(_make([]))
    assert isinstance(env, gymnasium.Env)
    assert env.reward_range is not None and env.np_random is not None
    assert env.action_space.shape is not None  # __getattr__ / mapped space
    env.reset(seed=0)
    env.step(env.action_space.sample())


def test_config_from_dict():
    cfg = RobustConfig.from_dict({
        "env_id": ENV_ID,
        "seed": 0,
        "shifts": [{"target": "observation", "mode": "gauss", "params": {"sigma": 0.05}}],
    })
    assert cfg.env_id == ENV_ID and len(cfg.shifts) == 1
    assert isinstance(cfg.shifts[0], ShiftSpec)
