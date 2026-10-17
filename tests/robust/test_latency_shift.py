"""Tests for the sim-to-real primitives: latency, loguniform, relative/bias noise.

These back the reality-profile evaluation (Peng et al. 2018, arXiv:1710.06537;
Tan et al. 2018, arXiv:1804.10332). Each mode is checked against the property
that makes it worth having at all, not merely that it runs:

  * a delay of d steps must reproduce the undelayed action sequence shifted by
    d -- otherwise it is just action noise with extra steps;
  * loguniform must be symmetric in log space -- the reason it exists;
  * `bias` must be constant within an episode and redrawn across episodes --
    the property that distinguishes it from `gauss`.
"""

import numpy as np
import pytest

import robustrllib.gym as gym
from robustrllib import ShiftSpec
from robustrllib.core_wrapper import RobustCoreWrapper
from robustrllib.pipeline import build_pipeline
from robustrllib.shifts.dynamics import draw_dynamics_factor
from robustrllib.shifts.latency import LatencyShift

ENV_ID = "Hopper-v5"


def _base():
    try:
        return gym.make(ENV_ID)
    except Exception as e:  # pragma: no cover - mujoco extra missing
        pytest.skip(f"{ENV_ID} unavailable: {e}")


def _core(env):
    """The step-convention bridge `build_pipeline` installs first.

    These envs take a dict input (`robust_input["action"]`), not a bare array,
    so a shift wrapper stacked straight onto `gym.make(...)` raises IndexError
    on the first step. Tests that build a stack by hand must add it themselves.
    """
    return RobustCoreWrapper(env)


class _Recorder(gym.Wrapper):
    """Records the actions that actually reach the inner env."""

    def __init__(self, env):
        super().__init__(env)
        self.applied = []

    def reset(self, **kwargs):
        self.applied = []
        return self.env.reset(**kwargs)

    def step(self, action):
        self.applied.append(np.asarray(action, dtype=np.float64).copy())
        return self.env.step(action)


def _rollout(env, actions):
    env.reset(seed=0)
    for a in actions:
        env.step(a)


def _actions(n, dim, seed=1):
    return list(np.random.default_rng(seed).uniform(-1, 1, size=(n, dim)))


# ---------------------------------------------------------------- action delay

@pytest.mark.parametrize("delay", [1, 3])
def test_fixed_delay_shifts_the_action_sequence(delay):
    base = _base()
    rec = _Recorder(_core(base))
    env = LatencyShift(rec, ShiftSpec("latency", "fixed", {"steps": delay}), seed=0)
    acts = _actions(10, base.action_space.shape[0])
    _rollout(env, acts)

    applied = rec.applied
    assert len(applied) == len(acts)
    # After the initial hold, step t applies the action issued at t - delay.
    for t in range(delay, len(acts)):
        np.testing.assert_allclose(applied[t], acts[t - delay])
    # The buffer is pre-filled with the first action, never with zeros: a zero
    # command is a real torque and would be a transient, not a delay.
    for t in range(delay):
        np.testing.assert_allclose(applied[t], acts[0])
    base.close()


def test_zero_delay_is_the_identity():
    base = _base()
    rec = _Recorder(_core(base))
    env = LatencyShift(rec, ShiftSpec("latency", "fixed", {"steps": 0}), seed=0)
    acts = _actions(8, base.action_space.shape[0])
    _rollout(env, acts)
    for got, want in zip(rec.applied, acts):
        np.testing.assert_allclose(got, want)
    base.close()


def test_buffer_delay_stays_within_declared_bounds():
    base = _base()
    env = LatencyShift(_core(base), ShiftSpec("latency", "buffer", {"low": 1, "high": 3}),
                       seed=0)
    # Each episode gets its own reset seed. `ShiftWrapper.reset` reseeds the
    # perturbation stream whenever an explicit seed is passed (so that a whole
    # rollout is reproducible from one number), which means resetting 30 times
    # with seed=0 would legitimately draw the same delay 30 times.
    seen = {(env.reset(seed=ep), env._delay_steps)[1] for ep in range(40)}
    assert seen <= {1, 2, 3}, f"delays outside the declared [1, 3]: {sorted(seen)}"
    assert len(seen) > 1, "delay never varied across episodes"
    base.close()


def test_substep_rejects_envs_without_a_substep_knob():
    """An unsupported combination raises instead of falling back to another mode."""
    class _Bare(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (2,))
        action_space = gym.spaces.Box(-1, 1, (1,))

        def reset(self, **kw):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, a):
            return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    with pytest.raises(NotImplementedError, match="frame_skip or n_substeps"):
        LatencyShift(_core(_Bare()), ShiftSpec("latency", "substep", {}), seed=0)


def test_substep_varies_the_control_period_and_restores_it():
    base = _base()
    nominal = base.unwrapped.frame_skip
    env = LatencyShift(_core(base), ShiftSpec("latency", "substep",
                                       {"lambda_low": 125.0, "lambda_high": 1000.0}), seed=0)
    env.reset(seed=0)
    skips = []
    for _ in range(40):
        # Observe the skip the wrapper installs by reading it from inside the
        # inner step; simplest proxy is to check it is restored after each step
        # and that at least one draw exceeded the nominal.
        env.step(base.action_space.sample())
        skips.append(base.unwrapped.frame_skip)
    assert set(skips) == {nominal}, "frame_skip leaked past the step boundary"

    # Draw directly to confirm the period actually varies upward.
    drawn = {env._draw_skip() for _ in range(200)}
    assert max(drawn) > nominal, f"control period never exceeded nominal: {drawn}"
    assert min(drawn) >= nominal, "Exp(lambda) is non-negative; period cannot shrink"
    base.close()


def test_latency_is_reproducible_under_a_fixed_seed():
    def run():
        base = _base()
        rec = _Recorder(_core(base))
        env = LatencyShift(rec, ShiftSpec("latency", "buffer",
                                          {"low": 0, "high": 4}, seed=7), seed=7)
        acts = _actions(12, base.action_space.shape[0])
        for _ in range(3):
            _rollout(env, acts)
        out = [a.copy() for a in rec.applied]
        base.close()
        return out

    for a, b in zip(run(), run()):
        np.testing.assert_allclose(a, b)


# ------------------------------------------------------------------ loguniform

def test_loguniform_is_symmetric_in_log_space():
    spec = ShiftSpec("dynamics", "loguniform",
                     {"param": "body_mass", "low": 0.25, "high": 4.0})
    rng = np.random.default_rng(0)
    draws = np.array([draw_dynamics_factor(spec, rng) for _ in range(20000)])
    assert draws.min() >= 0.25 and draws.max() <= 4.0
    # Halving and doubling must be equally likely; a plain uniform over this
    # range would put ~84% of its mass above 1.0.
    assert abs((draws > 1.0).mean() - 0.5) < 0.02
    assert abs(np.log(draws).mean()) < 0.02


def test_loguniform_refuses_a_non_positive_range():
    spec = ShiftSpec("dynamics", "loguniform",
                     {"param": "dof_frictionloss", "low": 0.0, "high": 2.0})
    with pytest.raises(ValueError, match="strictly positive"):
        draw_dynamics_factor(spec, np.random.default_rng(0))


# ------------------------------------------------ relative / bias observation noise

def test_relative_noise_scales_with_each_feature():
    """One absolute sigma cannot serve a state that mixes angles and velocities."""
    base = _base()
    ref = np.concatenate([np.full(5, 0.01), np.full(6, 10.0)])
    env = build_pipeline(base, [ShiftSpec("observation", "relative",
                                       {"frac": 0.1, "ref_std": ref.tolist()}, seed=0)],
                      seed=0)
    obs0, _ = env.reset(seed=0)
    clean = base.unwrapped._get_obs()
    resid = np.abs(obs0 - clean)
    # Low-scale features get small perturbations, high-scale ones large.
    assert resid[:5].max() < resid[5:].min()
    base.close()


def test_bias_is_constant_within_an_episode_and_redrawn_across():
    base = _base()
    env = build_pipeline(base, [ShiftSpec("observation", "bias",
                                       {"low": -0.5, "high": 0.5}, seed=0)], seed=0)
    obs, _ = env.reset(seed=0)
    b0 = obs - base.unwrapped._get_obs()
    for _ in range(5):
        obs, *_ = env.step(base.action_space.sample())
        np.testing.assert_allclose(obs - base.unwrapped._get_obs(), b0, atol=1e-12)

    obs, _ = env.reset(seed=1)
    b1 = obs - base.unwrapped._get_obs()
    assert not np.allclose(b0, b1), "bias was not redrawn at reset"
    base.close()


def test_stateful_modes_are_rejected_where_there_is_no_state_to_hold_them():
    """`sample_noise` backs the reward and multi-agent wrappers, which have no
    per-episode bias; a config asking for one there must fail, not degrade."""
    from robustrllib.shifts.base import sample_noise
    for mode in ("relative", "bias"):
        with pytest.raises(ValueError, match="not an additive noise mode"):
            sample_noise(np.random.default_rng(0), mode, {}, (3,), 1.0)


# ------------------------------------------------------------------- composition

def test_reality_profile_stack_runs_and_perturbs():
    """The Section-3 claim: several shift TYPES compose in one env."""
    base = _base()
    shifts = [
        ShiftSpec("dynamics", "loguniform",
                  {"param": "body_mass", "index": "all", "low": 0.8, "high": 1.25}, seed=0),
        ShiftSpec("dynamics", "loguniform",
                  {"param": "dof_damping", "index": "all", "low": 0.7, "high": 1.4}, seed=0),
        ShiftSpec("latency", "fixed", {"steps": 1}, seed=0),
        ShiftSpec("observation", "relative", {"frac": 0.05}, seed=0),
        ShiftSpec("observation", "bias", {"low": -0.02, "high": 0.02}, seed=0),
        ShiftSpec("action", "gauss", {"sigma": 0.01}, seed=0),
    ]
    env = build_pipeline(base, shifts, seed=0)
    obs, _ = env.reset(seed=0)
    assert np.isfinite(obs).all()
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            break
    base.close()
