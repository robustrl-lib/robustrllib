"""CPU tests of the task definitions (run inside the Isaac Lab Python environment)."""
import copy
from types import SimpleNamespace

import pytest
import torch

from isaac.methods.spidr import EnsembleVecEnv, restore_controller_targets
from isaac.tasks import drawer, g1


def test_g1_grid_has_31_conditions_from_the_suite():
    suite = g1.load_suite()
    names = [row["name"] for row in suite["conditions"] + suite["ood_conditions"]]
    assert len(names) == len(set(names))
    grid = g1.grid()
    assert len(grid) == 31 and grid.count("rough_nominal") == 1
    for entry in grid:
        terrain, name = entry.split("_", 1)
        assert terrain in ("flat", "rough") and name in names
    regions = [row["calibration_region"] for row in suite["conditions"] + suite["ood_conditions"]]
    assert regions.count("id") == 6 and regions.count("boundary") == 18 and regions.count("ood") == 5


def test_g1_ood_values_lie_outside_the_training_ranges():
    suite = g1.load_suite()
    ranges = {"friction": suite["calibration"]["friction"]["range"],
              "link_mass_scale": suite["calibration"]["link_mass_scale"]["range"],
              "control_delay_ms": suite["calibration"]["control_delay_ms"]["range"]}
    for row in suite["ood_conditions"]:
        if row["axis"] in ranges:
            low, high = ranges[row["axis"]]
            assert not low <= row["value"] <= high
        elif row["axis"] == "base_com_offset":
            assert abs(row["value"][0]) > suite["calibration"]["base_com_offset_m"]["range"][1]
        else:
            assert row["value"] > suite["calibration"]["torque_rfi_ratio"]["value"]


def _robot(q, names):
    data = SimpleNamespace(joint_pos=q, joint_pos_limits=torch.tensor([[-1.0, 1.0]]).expand(*q.shape, 2))
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data, joint_names=names)})


def test_g1_joint_limit_cost_is_continuous():
    env = _robot(torch.tensor([[0.0, 0.79], [0.0, 0.90], [-1.0, 0.0]]), ["left_hip", "right_knee"])
    assert torch.allclose(g1.joint_limit_cost(env), torch.tensor([0.0, 0.25, 0.5]))
    stats = g1.joint_limit_statistics(env)
    assert torch.allclose(stats["violation_fraction"], torch.tensor([0.0, 0.5, 0.5]))


def test_g1_cost_ignores_arm_joints_of_the_humanoid():
    env = _robot(torch.tensor([[0.0, 1.0]]), ["left_hip_joint", "left_shoulder_joint"])
    assert torch.equal(g1.joint_limit_cost(env), torch.tensor([0.0]))


def test_torque_noise_is_relative_to_motor_capacity():
    motor = SimpleNamespace(effort_limit=torch.tensor([23.5]))
    assert torch.allclose(0.1 * g1.motor_effort_limits(motor, device="cpu", dtype=torch.float32), torch.tensor([2.35]))
    with pytest.raises(ValueError):
        g1.motor_effort_limits(SimpleNamespace(effort_limit=float("inf")), device="cpu", dtype=torch.float32)


def test_drawer_has_two_controls_and_fourteen_exterior_conditions():
    assert len(drawer.CONDITIONS) == 16
    exterior = [c for c in drawer.CONDITIONS.values() if c["axis"]]
    assert len(exterior) == 14
    for c in exterior:
        assert all(c["value"] < low or c["value"] > high for low, high in drawer.AXES[c["axis"]]["ranges"])


def _drawer_cfg():
    events = SimpleNamespace()
    for spec in drawer.AXES.values():
        if spec["event"]:
            if not hasattr(events, spec["event"]):
                setattr(events, spec["event"], SimpleNamespace(params={}))
            getattr(events, spec["event"]).params.update(dict(zip(spec["keys"], map(tuple, spec["ranges"]))))
    return SimpleNamespace(events=events)


@pytest.mark.parametrize("name", [n for n, c in drawer.CONDITIONS.items() if c["axis"]])
def test_drawer_condition_changes_only_its_axis(name):
    cfg = _drawer_cfg()
    before = copy.deepcopy(cfg)
    condition = drawer.apply_condition(cfg, name)
    spec = drawer.AXES[condition["axis"]]
    for other in drawer.AXES.values():
        for key in other["keys"]:
            actual = getattr(cfg.events, other["event"]).params[key]
            if other["event"] == spec["event"] and key in spec["keys"]:
                assert actual == (condition["value"],) * 2
            else:
                assert actual == getattr(before.events, other["event"]).params[key]


def test_drawer_cost_ignores_the_gripper():
    q = torch.zeros(2, 9)
    q[:, 7:] = 1.0
    q[1, 0] = 0.9
    names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(
        joint_pos=q, joint_vel=torch.zeros_like(q), joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 9] * 2)))
    cabinet = SimpleNamespace(joint_names=["drawer_top_joint"], data=SimpleNamespace(
        joint_pos=torch.tensor([[0.0], [0.35]]), joint_vel=torch.zeros(2, 1)))
    measured = drawer.measure(SimpleNamespace(scene={"robot": robot, "cabinet": cabinet}))
    assert measured["cost"][0] == 0 and measured["cost"][1].item() == pytest.approx(0.5 / 7)
    assert measured["state"].shape == (2, 20) and measured["drawer_position"][1].item() == pytest.approx(0.35)


@pytest.mark.parametrize("lag", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("members", [0, 8])
def test_action_delay_returns_the_lagged_action(lag, members):
    env = object.__new__(EnsembleVecEnv)
    env.behavior_envs, env.device = 2, "cpu"
    env._delay = torch.full((2 * (members + 1),), lag)
    env._history = torch.zeros(max(4, lag + 1), 2, 8)
    for step in range(1, 8):
        out = env._delayed(torch.full((2, 8), float(step)))
        assert torch.equal(out, torch.full_like(out, max(0, step - lag)))


def test_members_receive_the_behaviour_controller_targets():
    assets = {}
    for name in ("robot", "cabinet"):
        q, v = torch.tensor([[0.2], [0.3], [9.0], [9.0]]), torch.tensor([[0.0], [0.1], [8.0], [8.0]])
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos_target=q, joint_vel_target=v))
        asset.set_joint_position_target_index = lambda target, env_ids, q=q: q.__setitem__(env_ids, target)
        asset.set_joint_velocity_target_index = lambda target, env_ids, v=v: v.__setitem__(env_ids, target)
        assets[name] = asset
    restore_controller_targets(SimpleNamespace(articulations=assets), 2, 1, torch.tensor([2, 3]))
    for asset in assets.values():
        assert torch.equal(asset.data.joint_pos_target[2:], asset.data.joint_pos_target[:2])
        assert torch.equal(asset.data.joint_vel_target[2:], asset.data.joint_vel_target[:2])


def test_drawer_success_is_the_final_opening():
    class Policy:
        def __call__(self, obs):
            return torch.zeros(2, 8)

        def reset(self, done):
            pass

    class Env:
        num_envs, device, max_episode_length = 2, "cpu", 2

        def reset(self):
            self.t = 0
            return {}, {}

        def step(self, action):
            self.t += 1
            opening = torch.tensor([0.35, 0.35]) if self.t == 1 else torch.tensor([0.20, 0.40])
            return {}, torch.ones(2), torch.tensor([self.t == 2] * 2), dict(drawer_position=opening,
                                                                             raw_cost=torch.zeros(2))

    result = drawer.rollout(Env(), Policy())
    assert result["final_open_m"] == pytest.approx([0.20, 0.40]) and result["max_open_m"] == pytest.approx([0.35, 0.40])
    assert result["ret"] == [2.0, 2.0] and result["steps"] == [2.0, 2.0]
