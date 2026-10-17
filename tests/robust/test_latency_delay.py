"""`delay` is a latency mode: the reward arrives late, nothing else moves.

`latency` + `delay` and `reward` + `delay` name the same shift: both build the
same wrapper and produce the same reward stream.
"""

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust
from robustrllib.shifts import RewardShift
from robustrllib.shifts.latency import LatencyShift


def _rollout(shifts, steps=250, seed=0):
    env = make_robust("Hopper-v5", shifts=shifts, seed=seed)
    obs, _ = env.reset(seed=seed)
    trace = [np.asarray(obs, dtype=np.float64)]
    rewards = []
    for _ in range(steps):
        obs, r, te, tr, _ = env.step(np.zeros(env.action_space.shape))
        trace.append(np.asarray(obs, dtype=np.float64))
        rewards.append(float(r))
        if te or tr:
            break
    env.close()
    return rewards, trace


@pytest.mark.parametrize("release", ["interval", "shift"])
@pytest.mark.parametrize("steps", [4, 16])
def test_latency_delay_equals_the_earlier_spelling(release, steps):
    params = {"steps": steps, "release": release}
    new, new_obs = _rollout([ShiftSpec("latency", "delay", params)])
    old, old_obs = _rollout([ShiftSpec("reward", "delay", params)])
    assert new == old
    assert all(np.array_equal(a, b) for a, b in zip(new_obs, old_obs))


def test_latency_delay_builds_the_reward_delay_wrapper():
    env = make_robust("Hopper-v5", shifts=[ShiftSpec("latency", "delay", {"steps": 4})])
    assert isinstance(env, RewardShift)
    env.close()


def test_latency_delay_moves_reward_and_nothing_else():
    base, base_obs = _rollout([])
    got, got_obs = _rollout([ShiftSpec("latency", "delay", {"steps": 16})])
    assert len(got) == len(base) < 250, "episode must terminate so the flush fires"
    assert np.isclose(sum(got), sum(base), rtol=1e-9)
    assert got != base
    assert all(np.array_equal(a, b) for a, b in zip(got_obs, base_obs))


def test_latency_delay_composes_with_an_execution_delay():
    shifts = [ShiftSpec("latency", "fixed", {"steps": 2}),
              ShiftSpec("latency", "delay", {"steps": 8})]
    got, _ = _rollout(shifts)
    ref, _ = _rollout(shifts[:1])
    assert np.isclose(sum(got), sum(ref), rtol=1e-9)


def test_cost_has_no_delay():
    with pytest.raises(ValueError, match="no cost form"):
        make_robust("Hopper-v5", shifts=[ShiftSpec("cost", "delay", {"steps": 4})])


def test_the_latency_class_points_at_the_registry():
    import robustrllib.gym as gym
    with pytest.raises(ValueError, match="make_robust"):
        LatencyShift(gym.make("Hopper-v5"), ShiftSpec("latency", "delay", {"steps": 4}))
