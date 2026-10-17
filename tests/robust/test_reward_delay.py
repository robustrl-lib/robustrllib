"""Reward delay must move reward in time without destroying it.

The undiscounted episode return is the invariant that makes a delayed-reward
training run comparable with its own control: if the wrapper leaked reward, a
lower score would confound "harder credit assignment" with "smaller objective".
"""

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust


def _rollout(shifts, steps=250, seed=0, terminate=True):
    """One rollout. `terminate=True` lets the episode end, which is what fires
    the flush; the return invariant is defined per EPISODE, so a run cut off
    mid-window legitimately still holds the unreleased tail in the accumulator."""
    env = make_robust("Hopper-v5", shifts=shifts, seed=seed,
                      terminate_when_unhealthy=terminate)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rs = []
    for _ in range(steps):
        _, r, te, tr, _ = env.step(np.zeros(env.action_space.shape))
        rs.append(float(r))
        if te or tr:
            break
    env.close()
    return rs


def _spec(**p):
    return [ShiftSpec("reward", "delay", p)]


def test_interval_release_preserves_the_episode_return():
    base = _rollout([])
    assert len(base) < 250, "episode must terminate for the flush to be exercised"
    for k in (4, 16, 64):
        got = _rollout(_spec(steps=k, release="interval"))
        assert len(got) == len(base)
        assert np.isclose(sum(got), sum(base), rtol=1e-9), (k, sum(got), sum(base))


def test_interval_release_holds_the_tail_when_the_episode_is_cut_off():
    """Cut mid-window, the unreleased remainder is still in the accumulator --
    withheld, not destroyed. This is why the invariant is per episode."""
    base = _rollout([], steps=50, terminate=False)
    got = _rollout(_spec(steps=16, release="interval"), steps=50, terminate=False)
    assert sum(got) < sum(base)
    assert np.isclose(sum(got), sum(base[:48]), rtol=1e-9)


def test_shift_release_preserves_the_episode_return():
    base = _rollout([])
    for k in (1, 8):
        got = _rollout(_spec(steps=k, release="shift"))
        assert np.isclose(sum(got), sum(base), rtol=1e-9), (k, sum(got), sum(base))


def test_interval_release_is_actually_sparse():
    """Reward arrives only on the release steps -- otherwise nothing is delayed."""
    got = _rollout(_spec(steps=16, release="interval"), steps=160, terminate=False)
    nz = [i for i, r in enumerate(got) if r != 0.0]
    assert len(nz) <= len(got) // 8, nz[:20]
    assert all((i + 1) % 16 == 0 for i in nz[:-1]), nz[:20]


def test_shift_release_pays_each_reward_late():
    base = _rollout([], steps=60, terminate=False)
    got = _rollout(_spec(steps=3, release="shift"), steps=60, terminate=False)
    assert got[:3] == [0.0, 0.0, 0.0]
    assert np.allclose(got[3:10], base[:7])


def test_delay_rejects_bad_configuration():
    import pytest
    with pytest.raises(ValueError):
        _rollout(_spec(steps=0), steps=2)
    with pytest.raises(ValueError):
        _rollout(_spec(steps=4, release="nonsense"), steps=2)
