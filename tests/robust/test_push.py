"""PushShift audit: force only inside its window, cleared on reset/episode end,
paired seeds reproduce the push sequence, and the push actually moves the body
(Delta_P > 0 on Hopper's torso)."""

import numpy as np

from robustrllib import ShiftSpec, make_robust


def _push(**over):
    p = {"body": "torso", "interval": 20, "duration": 3, "force": 80.0, "direction": "random_x"}
    p.update(over)
    return ShiftSpec("dynamics", "push", p)


def _find_push(env):
    e = env
    while not hasattr(e, "push_log"):
        e = e.env
    return e


def test_force_only_inside_window_and_cleared_on_reset():
    env = make_robust("Hopper-v5", shifts=[_push()], seed=0, terminate_when_unhealthy=False)
    env.reset(seed=0)
    w = _find_push(env)
    bid = w._body
    data = env.unwrapped.data
    assert np.all(data.xfrc_applied[bid] == 0.0)
    active = 0
    for t in range(60):
        env.step(np.zeros(env.action_space.shape))
        f = data.xfrc_applied[bid, :3]
        # after step t the force reflects whether t was inside a window
        inside = any(t0 <= t < t0 + 3 for t0, _ in w.push_log)
        if inside:
            assert abs(f[0]) == 80.0 and f[1] == 0.0 and f[2] == 0.0
            active += 1
        else:
            assert np.all(f == 0.0)
    assert active == 3 * len(w.push_log) and len(w.push_log) >= 2
    env.reset(seed=1)
    assert np.all(data.xfrc_applied[bid] == 0.0)
    env.close()


def test_paired_seed_reproduces_push_sequence():
    a = make_robust("Hopper-v5", shifts=[_push()], seed=0, terminate_when_unhealthy=False)
    b = make_robust("Hopper-v5", shifts=[_push(), ShiftSpec("observation", "gauss", {"sigma": 0.05})],
                    seed=99, terminate_when_unhealthy=False)
    for env in (a, b):
        env.reset(seed=5)
        for _ in range(60):
            env.step(np.zeros(env.action_space.shape))
    la, lb = _find_push(a).push_log, _find_push(b).push_log
    assert [t for t, _ in la] == [t for t, _ in lb]
    assert all(np.allclose(u, v) for (_, u), (_, v) in zip(la, lb))
    a.close(); b.close()


def test_push_changes_the_transition():
    """Delta_P witness: same (s, a), with vs without the wrench -> different s'."""
    nom = make_robust("Hopper-v5", seed=0)
    sh = make_robust("Hopper-v5", shifts=[_push(interval=1, duration=1, direction=[1, 0, 0])], seed=0)
    nom.reset(seed=0); sh.reset(seed=0)
    q, v = nom.unwrapped.data.qpos.copy(), nom.unwrapped.data.qvel.copy()
    sh.unwrapped.set_state(q, v)
    a = np.zeros(nom.action_space.shape)
    nom.step(a); sh.step(a)
    d = np.linalg.norm(nom.unwrapped.data.qvel - sh.unwrapped.data.qvel)
    assert d > 1e-6, d
    nom.close(); sh.close()


def test_schedule_scales_force():
    env = make_robust("Hopper-v5", shifts=[ShiftSpec(
        "dynamics", "push",
        {"body": "torso", "interval": 1, "duration": 1, "force": 10.0, "direction": [0, 1, 0]},
        schedule={"type": "step", "start": 0.0, "end": 1.0, "t1": 2})],
        seed=0, terminate_when_unhealthy=False)
    env.reset(seed=0)
    bid = _find_push(env)._body
    data = env.unwrapped.data
    seen = []
    for _ in range(4):
        env.step(np.zeros(env.action_space.shape))
        seen.append(float(data.xfrc_applied[bid, 1]))
    assert seen == [0.0, 0.0, 10.0, 10.0], seen
    env.close()
