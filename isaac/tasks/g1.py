"""Unitree G1 velocity tracking (Isaac-Velocity-Flat-G1-v0): training configurations, the OmniH2O
shift suite used for evaluation, the rough test terrain and the joint-limit safety cost.

Training configurations
    nominal(env_cfg)            no randomisation (PPO, ATLA-PPO, RAMU-PPO)
    (unchanged task config)     the task's own randomisation: observation noise and velocity pushes (DR-PPO)
    omnih2o_training(env_cfg)   the OmniH2O training envelope (SPiDR-PPO): friction, link mass, base CoM,
                                actuator gains and torque noise here; action delay and velocity pushes are
                                applied by the SPiDR ensemble environment (isaac/methods/spidr.py)

Evaluation conditions come from isaac/configs/omnih2o.yaml (one axis per condition) and
isaac/configs/g1_conditions.tsv (the 31 evaluated terrain/condition pairs).

Isaac Lab is imported inside the functions, so this module can be imported before the simulator starts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml

TASK = "Isaac-Velocity-Flat-G1-v0"
BASE_BODY = "torso_link"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SUITE = CONFIGS / "omnih2o.yaml"
GRID = CONFIGS / "g1_conditions.tsv"
STOCK_RANDOMISATION = ("physics_material", "add_base_mass", "base_com", "push_robot", "base_external_force_torque")
# Rough test terrain: random uniform height field, +-2 cm in 5 mm steps.
ROUGH = dict(noise_range_m=(-0.02, 0.02), noise_step_m=0.005, horizontal_scale_m=0.1, downsampled_scale_m=0.2,
             tile_size_m=(20.0, 20.0), rows=8, cols=16, seed=10000)


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


# --------------------------------------------------------------------------- training configurations
def nominal(env_cfg) -> None:
    """Remove the task's training randomisation (observation noise, mass/CoM randomisation, pushes)."""
    env_cfg.observations.policy.enable_corruption = False
    for name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def omnih2o_training(env_cfg, seed: int) -> None:
    """Install the OmniH2O training ranges (OmniH2O, Table 16) as startup events and torque noise."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.observations.policy.enable_corruption = False
    for name in STOCK_RANDOMISATION:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    robot = dict(asset_cfg=SceneEntityCfg("robot", body_names=".*"))
    joints = dict(asset_cfg=SceneEntityCfg("robot", joint_names=".*"), operation="scale", distribution="uniform")
    events = env_cfg.events
    events.omnih2o_friction = EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params=dict(
        robot, static_friction_range=(0.2, 1.1), dynamic_friction_range=(0.2, 1.1),
        restitution_range=(0.0, 0.0), num_buckets=64))
    events.omnih2o_link_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup", params=dict(
        robot, mass_distribution_params=(0.7, 1.3), operation="scale", distribution="uniform",
        recompute_inertia=True))
    events.omnih2o_base_com = EventTermCfg(func=mdp.randomize_rigid_body_com, mode="startup", params=dict(
        asset_cfg=SceneEntityCfg("robot", body_names=".*base.*|torso_link"),
        com_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)}))
    events.omnih2o_kp = EventTermCfg(func=mdp.randomize_actuator_gains, mode="startup",
                                     params=dict(joints, stiffness_distribution_params=(0.75, 1.25)))
    events.omnih2o_kd = EventTermCfg(func=mdp.randomize_actuator_gains, mode="startup",
                                     params=dict(joints, damping_distribution_params=(0.75, 1.25)))
    env_cfg.actions.joint_pos.class_type = torque_noise_action(ratio=0.10, seed=seed)


# --------------------------------------------------------------------------- torque noise
def motor_effort_limits(actuator, *, device, dtype) -> torch.Tensor:
    """Motor torque capacity of an actuator group (the noise amplitude is relative to it)."""
    limits = torch.as_tensor(_torch(actuator.effort_limit), device=device, dtype=dtype)
    if not torch.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("torque noise needs positive finite motor effort limits")
    return limits


def torque_noise_action(ratio: float, seed: int):
    """Joint-position action class that adds uniform torque noise of +-ratio x motor limit at every PD step."""
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    class TorqueNoiseJointPositionAction(JointPositionAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._effort_limits = torch.zeros_like(self.processed_actions)
            for actuator in self._asset.actuators.values():
                ids = actuator.joint_indices
                ids = torch.arange(self._asset.num_joints, device=self.device) if isinstance(ids, slice) else ids.to(self.device)
                limits = motor_effort_limits(actuator, device=self.device, dtype=self.raw_actions.dtype)
                self._effort_limits[:, ids] = limits.expand(self.num_envs, len(ids)) if limits.ndim < 2 else limits
            self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        def reset(self, env_ids=None):
            super().reset(env_ids)
            self._asset.set_joint_effort_target_index(target=torch.zeros_like(self.processed_actions),
                                                      joint_ids=self._joint_ids)

        def apply_actions(self):
            super().apply_actions()
            effort = torch.zeros_like(self.processed_actions)
            if ratio:
                noise = torch.rand(effort.shape, device=self.device, dtype=effort.dtype, generator=self._generator)
                effort = (2.0 * noise - 1.0) * self._effort_limits * ratio
            self._asset.set_joint_effort_target_index(target=effort, joint_ids=self._joint_ids)

    return TorqueNoiseJointPositionAction


# --------------------------------------------------------------------------- evaluation suite
@dataclass(frozen=True)
class Condition:
    name: str
    axis: str
    value: Any
    region: str
    delay_steps: int = 0


def load_suite(path: str | Path = SUITE) -> dict:
    return yaml.safe_load(Path(path).read_text())


def grid(path: str | Path = GRID) -> list[str]:
    """The evaluated ``<terrain>_<condition>`` names, in file order."""
    return ["_".join(line.split()) for line in Path(path).read_text().splitlines() if line.strip()]


def find_condition(suite: dict, name: str) -> dict:
    rows = [row for row in suite["conditions"] + suite["ood_conditions"] if row["name"] == name]
    if len(rows) != 1:
        raise KeyError(f"condition {name!r} is not in the suite")
    return rows[0]


def apply_condition(env_cfg, condition: dict, seed: int) -> Condition:
    """Replace the training randomisation by one fixed OmniH2O condition (before gym.make)."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.observations.policy.enable_corruption = False
    for name in STOCK_RANDOMISATION:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    axis, value, events = condition["axis"], condition["value"], env_cfg.events
    delay_steps = 0
    if axis == "friction":
        events.omnih2o_friction = EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params=dict(
            asset_cfg=SceneEntityCfg("robot", body_names=".*"), static_friction_range=(value, value),
            dynamic_friction_range=(value, value), restitution_range=(0.0, 0.0), num_buckets=1))
    elif axis == "link_mass_scale":
        events.omnih2o_link_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup", params=dict(
            asset_cfg=SceneEntityCfg("robot", body_names=".*"), mass_distribution_params=(value, value),
            operation="scale", distribution="uniform", recompute_inertia=True))
    elif axis == "base_com_offset":
        x, y, z = (float(v) for v in value)
        events.omnih2o_base_com = EventTermCfg(func=mdp.randomize_rigid_body_com, mode="startup", params=dict(
            asset_cfg=SceneEntityCfg("robot", body_names=BASE_BODY), com_range={"x": (x, x), "y": (y, y), "z": (z, z)}))
    elif axis in ("kp_scale", "kd_scale"):
        key = "stiffness_distribution_params" if axis == "kp_scale" else "damping_distribution_params"
        events.omnih2o_actuator_gain = EventTermCfg(func=mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "operation": "scale",
            "distribution": "uniform", key: (value, value)})
    elif axis == "torque_rfi_ratio":
        env_cfg.actions.joint_pos.class_type = torque_noise_action(ratio=float(value), seed=seed)
    elif axis == "velocity_push":
        vx, vy = (float(v) for v in value)
        interval = float(condition.get("interval_s", 5.0))
        events.omnih2o_velocity_push = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval",
                                                    interval_range_s=(interval, interval),
                                                    params={"velocity_range": {"x": (vx, vx), "y": (vy, vy)}})
    elif axis == "control_delay_ms":
        step_ms = env_cfg.sim.dt * env_cfg.decimation * 1000.0
        delay_steps = round(value / step_ms)
        if not math.isclose(delay_steps * step_ms, value):
            raise ValueError(f"{value} ms is not a multiple of the {step_ms} ms control step")
    elif axis != "nominal":
        raise ValueError(f"unknown axis {axis!r}")
    return Condition(condition["name"], axis, value, condition["calibration_region"], delay_steps)


def rough_terrain(env_cfg) -> None:
    """Replace the flat ground by the rough test height field (the observations, actions, rewards
    and commands of the flat task are unchanged)."""
    from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg

    terrain = env_cfg.scene.terrain
    terrain.terrain_type = "generator"
    terrain.max_init_terrain_level = None
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=ROUGH["seed"], size=ROUGH["tile_size_m"], num_rows=ROUGH["rows"], num_cols=ROUGH["cols"],
        curriculum=False, use_cache=False, horizontal_scale=ROUGH["horizontal_scale_m"],
        vertical_scale=ROUGH["noise_step_m"],
        sub_terrains={"rough": HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=ROUGH["noise_range_m"], noise_step=ROUGH["noise_step_m"],
            downsampled_scale=ROUGH["downsampled_scale_m"], border_width=0.25)})
    if hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None


class ActionDelayWrapper(gym.Wrapper):
    """Delay every action by a fixed number of control steps; the buffer restarts with each episode."""

    def __init__(self, env: gym.Env, delay_steps: int):
        from isaaclab.utils import DelayBuffer

        super().__init__(env)
        raw = env.unwrapped
        self._buffer = DelayBuffer(history_length=delay_steps, batch_size=raw.num_envs, device=raw.device)
        self._buffer.set_time_lag(delay_steps)

    def reset(self, **kwargs):
        self._buffer.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(self._buffer.compute(action))
        ended = torch.nonzero(result[2] | result[3], as_tuple=False).squeeze(-1)
        if ended.numel():
            self._buffer.reset(ended)
        return result


def rollout(env, policy) -> list[dict]:
    """Run the deterministic policy until every environment has finished its first episode."""
    raw = env.unwrapped
    n = raw.num_envs
    done = torch.zeros(n, dtype=torch.bool, device=raw.device)
    returns = torch.zeros(n, device=raw.device)
    lengths = torch.zeros(n, dtype=torch.long, device=raw.device)
    episodes: list[dict | None] = [None] * n
    obs = env.get_observations()
    for _ in range(2 * int(raw.max_episode_length)):
        with torch.inference_mode():
            obs, rewards, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        returns += rewards * ~done
        lengths += ~done
        for i in torch.nonzero(dones.bool() & ~done, as_tuple=False).squeeze(-1).tolist():
            timed_out = bool(raw.reset_time_outs[i])
            episodes[i] = {"env_id": i, "return": float(returns[i]), "length": int(lengths[i]), "fell": not timed_out}
            done[i] = True
        if done.all():
            return episodes
    raise RuntimeError("some environments did not finish an episode")


# --------------------------------------------------------------------------- safety cost and SPiDR state
def _safety_joint_ids(robot) -> torch.Tensor:
    """Leg and torso joints (the hand and arm joints of the humanoid are excluded)."""
    names = list(robot.joint_names)
    legs = re.compile(r"(hip|knee|ankle|torso|waist|thigh|calf)", re.IGNORECASE)
    arms = re.compile(r"(shoulder|elbow|wrist|hand|finger|thumb)", re.IGNORECASE)
    selected = [i for i, n in enumerate(names) if legs.search(n)] if any(arms.search(n) for n in names) \
        else list(range(len(names)))
    return torch.tensor(selected, device=_torch(robot.data.joint_pos).device, dtype=torch.long)


def joint_limit_statistics(raw_env, margin: float = 0.10) -> dict[str, torch.Tensor]:
    """Mean penetration of the leg joints into the outer ``margin`` of their ranges, in [0, 1]."""
    robot = raw_env.scene["robot"]
    ids = _safety_joint_ids(robot)
    q = _torch(robot.data.joint_pos).index_select(-1, ids)
    limits = _torch(robot.data.joint_pos_limits).index_select(-2, ids)
    span = limits[..., 1] - limits[..., 0]
    valid = torch.isfinite(limits).all(-1) & (span > 1.0e-6)
    phase = (q - limits[..., 0]) / torch.where(valid, span, torch.ones_like(span))
    utilisation = (2.0 * phase - 1.0).abs()
    threshold = 1.0 - 2.0 * margin
    severity = ((utilisation - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    severity = torch.where(valid, severity, torch.zeros_like(severity))
    count = valid.sum(-1).clamp_min(1)
    return dict(raw_cost=severity.sum(-1) / count,
                violation_fraction=((severity > 0) & valid).sum(-1).float() / count)


def joint_limit_cost(raw_env) -> torch.Tensor:
    return joint_limit_statistics(raw_env)["raw_cost"]


def next_state(raw_env) -> torch.Tensor:
    """Scaled physical state whose variance across the ensemble is the SPiDR disagreement."""
    data = raw_env.scene["robot"].data
    height = _torch(data.root_pos_w)[:, 2:3] - raw_env.scene.env_origins[:, 2:3]
    limits = _torch(data.joint_pos_limits)
    middle, half = limits.mean(-1), 0.5 * (limits[..., 1] - limits[..., 0])
    finite = torch.isfinite(middle) & torch.isfinite(half) & (half > 1.0e-6)
    middle = torch.where(finite, middle, torch.zeros_like(middle))
    half = torch.where(finite, half, torch.ones_like(half))
    return torch.cat((height, _torch(data.root_lin_vel_b) / 3.0, _torch(data.root_ang_vel_b) / 6.0,
                      _torch(data.projected_gravity_b), (_torch(data.joint_pos) - middle) / half,
                      _torch(data.joint_vel) / 10.0), dim=-1)


def ensemble_measure(raw_env) -> dict[str, torch.Tensor]:
    """Per-environment safety cost and next state for the SPiDR ensemble environment."""
    return dict(cost=joint_limit_cost(raw_env), state=next_state(raw_env))
