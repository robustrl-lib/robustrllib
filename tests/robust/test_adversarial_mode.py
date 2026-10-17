"""`adversarial` is a mode the wrappers implement, on actions and observations."""

import numpy as np
import pytest

from robustrllib import ShiftSpec, bind_actor, make_robust

torch = pytest.importorskip("torch")


def _actions_applied(shifts, steps=20, seed=0):
    """Observations of a rollout under a fixed action sequence."""
    env = make_robust("Hopper-v5", shifts=shifts, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [np.asarray(obs, dtype=np.float64)]
    for _ in range(steps):
        a = rng.uniform(-1, 1, size=env.action_space.shape)
        obs, _, te, tr, _ = env.step(a)
        out.append(np.asarray(obs, dtype=np.float64))
        if te or tr:
            break
    env.close()
    return out


def _linear_actor(obs_dim=11, act_dim=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(act_dim, obs_dim, generator=g)
    return lambda o: torch.tanh(o @ w.T)


# ------------------------------------------------------------------ action
def test_adversarial_action_defaults_to_oppose():
    a = _actions_applied([ShiftSpec("action", "adversarial", {"eps": 0.2})])
    b = _actions_applied([ShiftSpec("action", "oppose", {"eps": 0.2})])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_adversarial_action_kind_selects_the_member():
    a = _actions_applied([ShiftSpec("action", "adversarial", {"kind": "oppose", "eps": 0.3})])
    b = _actions_applied([ShiftSpec("action", "oppose", {"eps": 0.3})])
    c = _actions_applied([])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_adversarial_action_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown adversarial action kind"):
        make_robust("Hopper-v5", shifts=[ShiftSpec("action", "adversarial", {"kind": "gauss"})])


def test_adversarial_action_keeps_the_members_own_limits():
    # rotate is defined on 2-D action spaces only; the class name does not lift that
    env = make_robust("Hopper-v5", shifts=[ShiftSpec("action", "adversarial", {"kind": "rotate"})])
    env.reset(seed=0)
    with pytest.raises(ValueError, match="2-D action space"):
        env.step(np.zeros(env.action_space.shape))
    env.close()


# ------------------------------------------------------------- observation
def _obs_pair(eps, steps=10, seed=0, actor=None, n=15):
    """(true observations, observations the policy is shown) for one rollout."""
    actor = actor or _linear_actor()
    clean = make_robust("Hopper-v5", seed=seed)
    adv = make_robust("Hopper-v5", seed=seed, shifts=[
        ShiftSpec("observation", "adversarial", {"eps": eps, "steps": steps, "actor": actor})])
    o, _ = clean.reset(seed=seed)
    p, _ = adv.reset(seed=seed)
    true, shown = [np.asarray(o, float)], [np.asarray(p, float)]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = rng.uniform(-1, 1, size=clean.action_space.shape)
        o, _, te, tr, _ = clean.step(a)
        p, _, _, _, _ = adv.step(a)
        true.append(np.asarray(o, float))
        shown.append(np.asarray(p, float))
        if te or tr:
            break
    clean.close(); adv.close()
    return np.array(true), np.array(shown)


def test_adversarial_observation_stays_inside_the_ball():
    eps = 0.05
    true, shown = _obs_pair(eps)
    d = np.abs(shown - true)
    # exact up to double-precision rounding of (o + eps) - o; the single-precision
    # search alone overshoots by about 3e-8, which this bound would catch
    assert d.max() <= eps + 1e-12
    assert d.max() > 0.5 * eps, "PGD should reach the boundary of the ball"


def test_adversarial_observation_does_not_touch_the_dynamics():
    # the environment evolves on the true state; only what the policy sees moves
    true, _ = _obs_pair(0.05)
    ref = np.array(_actions_applied([], steps=15))
    assert np.array_equal(true, ref[: len(true)])


def test_adversarial_observation_beats_random_noise_of_the_same_radius():
    actor, eps = _linear_actor(), 0.05
    true, shown = _obs_pair(eps, actor=actor)
    rng = np.random.default_rng(1)
    noisy = true + rng.uniform(-eps, eps, size=true.shape)

    def moved(x):
        with torch.no_grad():
            a0 = actor(torch.as_tensor(true, dtype=torch.float32))
            a1 = actor(torch.as_tensor(x, dtype=torch.float32))
        return float(((a1 - a0) ** 2).sum(dim=1).mean())

    assert moved(shown) > 2.0 * moved(noisy)


def test_adversarial_observation_is_reproducible_under_a_seed():
    _, a = _obs_pair(0.05, seed=3)
    _, b = _obs_pair(0.05, seed=3)
    assert np.array_equal(a, b)


def test_zero_radius_is_the_identity():
    true, shown = _obs_pair(0.0)
    assert np.array_equal(true, shown)


def test_adversarial_observation_needs_an_actor():
    env = make_robust("Hopper-v5", shifts=[ShiftSpec("observation", "adversarial", {"eps": 0.05})])
    with pytest.raises(ValueError, match="no actor"):
        env.reset(seed=0)
    env.close()


def test_bind_actor_attaches_to_a_specification_without_one():
    env = make_robust("Hopper-v5", seed=0, shifts=[
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.1}),
        ShiftSpec("observation", "adversarial", {"eps": 0.05})])
    assert bind_actor(env, _linear_actor()) == 1
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
    env.close()
    plain = make_robust("Hopper-v5", shifts=[ShiftSpec("observation", "gauss", {"sigma": 0.01})])
    assert bind_actor(plain, _linear_actor()) == 0
    plain.close()
