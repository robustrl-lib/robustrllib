"""PointMaze dynamics conditions change the transition P(s'|s, a).

A policy's score on a short maze can stay close to nominal even when the physics
changed, so these tests hold the same (s, a) fixed, step one transition in the nominal
and in each shifted environment, and compare the next states. They also check that each
condition writes the configured value into the live model.

PointMaze has zero gravity and a zero nominal ``dof_frictionloss``: ``scale`` of either
is a no-op, which is why the evaluation card sets dry friction with ``set``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robustrllib import ShiftSpec, make_robust
from robustrllib.adapters import get_adapter

ENV_ID = "PointMaze_UMaze-v3"
SEED = 0
EFFECT_ATOL = 1e-8
NOOP_ATOL = 1e-10
CARD = Path(__file__).resolve().parents[2] / "robustrllib/configs/eval/part1_pointmaze_umaze.yaml"
PROBE_ACTIONS = (
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.0]),
    np.array([0.5, 0.5]),
)


def _dynamics_conditions():
    grid = yaml.safe_load(CARD.read_text())["grid"]
    return [c for c in grid if c["shifts"] and all(s["target"] == "dynamics" for s in c["shifts"])]


def _make(shifts=()):
    try:
        return make_robust(ENV_ID, shifts=list(shifts), seed=SEED, max_episode_steps=300,
                           reward_type="sparse", continuing_task=True, reset_target=True)
    except Exception as exc:  # pragma: no cover - extra missing
        pytest.skip(f"{ENV_ID} unavailable: {exc}")


def _point(env):
    return env.unwrapped.point_env


def _physics(env) -> np.ndarray:
    inner = _point(env)
    return np.concatenate([inner.data.qpos, inner.data.qvel]).astype(np.float64)


def _read_param(env, param, index):
    arr, key = get_adapter(env)._locate(param, index)
    return np.asarray(arr[key], dtype=np.float64).copy()


def _transitions(env, states):
    out = []
    for qpos, qvel in states:
        for action in PROBE_ACTIONS:
            _point(env).set_state(qpos, qvel)
            env.step(action)
            out.append(_physics(env))
    return np.stack(out)


@pytest.fixture(scope="module")
def nominal():
    """Probe states (rest and three points of a short burst) and their nominal transitions."""
    env = _make()
    try:
        env.reset(seed=SEED)
        inner = _point(env)
        states = [(inner.data.qpos.copy(), inner.data.qvel.copy())]
        burst = [np.array([1.0, 0.0])] * 8 + [np.array([0.0, 1.0])] * 4
        for i, action in enumerate(burst):
            env.step(action)
            if i in (3, 7, 11):
                states.append((inner.data.qpos.copy(), inner.data.qvel.copy()))
        params = {p: _read_param(env, p, "all")
                  for p in ("actuator_gear", "dof_frictionloss", "dof_damping", "body_mass")}
        return states, _transitions(env, states), params
    finally:
        env.close()


def _shifted(shifts, states):
    env = _make(shifts)
    try:
        env.reset(seed=SEED)
        written = {s.params["param"]: _read_param(env, s.params["param"], s.params.get("index"))
                   for s in shifts}
        return written, _transitions(env, states)
    finally:
        env.close()


@pytest.mark.parametrize("cond", _dynamics_conditions(), ids=lambda c: c["name"])
def test_condition_writes_its_value_and_changes_the_transition(cond, nominal):
    states, nominal_next, params = nominal
    shifts = [ShiftSpec(**s) for s in cond["shifts"]]
    written, shifted_next = _shifted(shifts, states)
    for s in shifts:
        p = s.params
        if s.mode == "set":
            expected = np.full_like(params[p["param"]], float(p["value"]))
        else:
            expected = params[p["param"]] * float(p["factor"])
        np.testing.assert_allclose(written[p["param"]], expected, rtol=1e-5, atol=1e-8)
    assert np.linalg.norm(shifted_next - nominal_next, axis=1).mean() > EFFECT_ATOL


def test_identity_scale_matches_nominal(nominal):
    states, nominal_next, _ = nominal
    identity = [ShiftSpec("dynamics", "scale", {"param": "actuator_gear", "index": "all", "factor": 1.0})]
    _, shifted_next = _shifted(identity, states)
    assert np.abs(shifted_next - nominal_next).max() < NOOP_ATOL


@pytest.mark.parametrize("param,index", [("dof_frictionloss", "all"), ("gravity", None)])
def test_scale_of_a_zero_nominal_is_a_no_op(param, index, nominal):
    states, nominal_next, _ = nominal
    params = {"param": param, "factor": 5.0}
    if index is not None:
        params["index"] = index
    env = _make([ShiftSpec("dynamics", "scale", params)])
    try:
        env.reset(seed=SEED)
        shifted_next = _transitions(env, states)
    finally:
        env.close()
    assert np.abs(shifted_next - nominal_next).max() < NOOP_ATOL
