"""The FetchReach dynamics axes change the simulation they name."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from robustrllib import ShiftSpec, make_robust


REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "robustrllib/configs/eval/part1_fetchreach.yaml"
ARM_BODIES = {
    "robot0:shoulder_lift_link",
    "robot0:upperarm_roll_link",
    "robot0:elbow_flex_link",
    "robot0:forearm_roll_link",
    "robot0:wrist_flex_link",
    "robot0:wrist_roll_link",
    "robot0:gripper_link",
}


def _pack(obs):
    return np.concatenate(
        [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]
    ).astype(np.float64)


def _rollout(shifts, seed=2026, steps=12):
    env = make_robust("FetchReachDense-v3", shifts=shifts, seed=123)
    rng = np.random.default_rng(99)
    trace = []
    try:
        obs, _ = env.reset(seed=seed)
        trace.append(_pack(obs))
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            trace.append(
                np.concatenate(
                    [_pack(obs), [reward, float(terminated), float(truncated)]]
                )
            )
            if terminated or truncated:
                break
        return trace
    finally:
        env.close()


def _condition(name):
    cfg = yaml.safe_load(GRID.read_text())
    row = next(item for item in cfg["grid"] if item["name"] == name)
    return [ShiftSpec(**spec) for spec in row["shifts"]]


class FetchReachAxesTest(unittest.TestCase):
    def test_grid_has_four_scales_per_axis(self):
        cfg = yaml.safe_load(GRID.read_text())
        self.assertEqual(cfg["episodes"], 20)
        self.assertEqual(cfg["seed"], 10000)
        axes = [row.get("axis") for row in cfg["grid"] if row["name"] != "nominal"]
        self.assertEqual(sorted(set(axes)), ["gear", "morphology"])
        self.assertEqual(axes.count("gear"), 4)
        self.assertEqual(axes.count("morphology"), 4)

    def test_logical_gear_maps_to_fetch_mocap_gain(self):
        nominal = make_robust("FetchReachDense-v3")
        shifted = make_robust(
            "FetchReachDense-v3", shifts=_condition("gear_0.4")
        )
        try:
            nominal.reset(seed=7)
            shifted.reset(seed=7)
            self.assertEqual(nominal.unwrapped.model.nu, 0)
            self.assertAlmostEqual(nominal.unwrapped.position_action_scale, 0.05)
            self.assertAlmostEqual(shifted.unwrapped.position_action_scale, 0.02)
        finally:
            nominal.close()
            shifted.close()

    def test_factor_one_is_exactly_nominal(self):
        factor_one = [
            ShiftSpec(
                "dynamics",
                "scale",
                {"param": "actuator_gear", "index": "all", "factor": 1.0},
            )
        ]
        factor_one.extend(
            ShiftSpec(
                "dynamics",
                "scale",
                {"param": "body_pos_xyz", "index": body, "factor": 1.0},
            )
            for body in sorted(ARM_BODIES)
        )
        nominal_trace = _rollout([])
        shifted_trace = _rollout(factor_one)
        self.assertEqual(len(nominal_trace), len(shifted_trace))
        for nominal, shifted in zip(nominal_trace, shifted_trace):
            np.testing.assert_array_equal(nominal, shifted)

    def test_each_axis_changes_the_transition(self):
        nominal = _rollout([], steps=1)[1]
        gear = _rollout(_condition("gear_2"), steps=1)[1]
        morph = _rollout(_condition("morphology_1.2"), steps=1)[1]
        self.assertGreater(np.max(np.abs(gear - nominal)), 1e-4)
        self.assertGreater(np.max(np.abs(morph - nominal)), 1e-3)

    def test_morphology_changes_only_named_arm_offsets_and_preserves_state(self):
        import mujoco

        nominal = make_robust("FetchReachDense-v3")
        shifted = make_robust(
            "FetchReachDense-v3", shifts=_condition("morphology_1.2")
        )
        try:
            nominal.reset(seed=11)
            shifted.reset(seed=11)
            changed = set()
            for body_id in range(nominal.unwrapped.model.nbody):
                if not np.array_equal(
                    nominal.unwrapped.model.body_pos[body_id],
                    shifted.unwrapped.model.body_pos[body_id],
                ):
                    changed.add(
                        mujoco.mj_id2name(
                            nominal.unwrapped.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            body_id,
                        )
                    )
            self.assertEqual(changed, ARM_BODIES)
            np.testing.assert_array_equal(
                nominal.unwrapped.data.qpos, shifted.unwrapped.data.qpos
            )
            np.testing.assert_array_equal(
                nominal.unwrapped.data.qvel, shifted.unwrapped.data.qvel
            )
            self.assertFalse(
                np.array_equal(
                    nominal.unwrapped.model.dof_invweight0,
                    shifted.unwrapped.model.dof_invweight0,
                )
            )
        finally:
            nominal.close()
            shifted.close()

    def test_same_seed_and_actions_are_exactly_reproducible(self):
        shifts = _condition("gear_2") + _condition("morphology_1.2")
        first = _rollout(shifts)
        second = _rollout(shifts)
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
