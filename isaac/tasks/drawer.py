"""Franka drawer opening (Isaac-Open-Drawer-Franka-v0): training randomisation, the arm
joint-limit safety cost, the 16 evaluation conditions and the evaluation rollout.

Training randomisation (DR-PPO, ATLA-PPO, SPiDR-PPO): the task's robot and handle friction
randomisation, plus cabinet mass x U[0.8, 1.2], arm Kp and Kd x U[0.9, 1.1], drawer damping
x U[0.5, 1.5] and an action delay of 0-2 control steps per environment. PPO and RAMU-PPO train
without it. Success means the drawer is open more than 0.30 m at the end of the 8 s episode.

The 16 conditions are two controls (``nominal``: no randomisation; ``reference_dr``: the training
randomisation) and two values per axis outside its training range, the other axes keeping the
training randomisation.
"""
from __future__ import annotations

import torch

from isaac.methods.spidr import EnsembleVecEnv

TASK = "Isaac-Open-Drawer-Franka-v0"
SUCCESS_M = 0.30
EPISODE_COST_BUDGET = 20.0
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# Per-dimension scale of the ATLA perturbation of the 31-dimensional observation.
ATTACK_SCALE = [1.0] * 9 + [5.0] * 9 + [0.4, 1.0] + [0.5] * 3 + [1.0] * 8

AXES = {
    "robot_friction": dict(ranges=[[0.8, 1.25], [0.8, 1.25]], values=[0.4, 1.8], event="robot_physics_material",
                           keys=["static_friction_range", "dynamic_friction_range"]),
    "handle_friction": dict(ranges=[[1.0, 1.25], [1.25, 1.5]], values=[0.5, 2.0], event="cabinet_physics_material",
                            keys=["static_friction_range", "dynamic_friction_range"]),
    "cabinet_mass": dict(ranges=[[0.8, 1.2]], values=[0.6, 1.4], event="drawer_mass",
                         keys=["mass_distribution_params"]),
    "arm_kp": dict(ranges=[[0.9, 1.1]], values=[0.75, 1.25], event="drawer_robot_gains",
                   keys=["stiffness_distribution_params"]),
    "arm_kd": dict(ranges=[[0.9, 1.1]], values=[0.75, 1.25], event="drawer_robot_gains",
                   keys=["damping_distribution_params"]),
    "drawer_damping": dict(ranges=[[0.5, 1.5]], values=[0.25, 2.0], event="drawer_slide_damping",
                           keys=["damping_distribution_params"]),
    "action_delay": dict(ranges=[[0, 2]], values=[3, 4], event=None, keys=[]),
}
CONDITIONS = {"nominal": dict(axis=None, value=None), "reference_dr": dict(axis=None, value=None)}
CONDITIONS.update({f"{axis}_{value}".replace(".", "p"): dict(axis=axis, value=value)
                   for axis, spec in AXES.items() for value in spec["values"]})


def configure(env_cfg, num_envs: int, seed: int, randomise: bool) -> None:
    """Scene size, seed and training randomisation (call before gym.make)."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.observations.policy.enable_corruption = False
    if not randomise:
        env_cfg.events.robot_physics_material = None
        env_cfg.events.cabinet_physics_material = None
        return
    scale = dict(operation="scale", distribution="uniform")
    env_cfg.events.drawer_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup", params=dict(
        asset_cfg=SceneEntityCfg("cabinet", body_names=".*"), mass_distribution_params=(0.8, 1.2),
        recompute_inertia=True, **scale))
    env_cfg.events.drawer_robot_gains = EventTermCfg(func=mdp.randomize_actuator_gains, mode="startup", params=dict(
        asset_cfg=SceneEntityCfg("robot", joint_names="panda_joint[1-7]"),
        stiffness_distribution_params=(0.9, 1.1), damping_distribution_params=(0.9, 1.1), **scale))
    env_cfg.events.drawer_slide_damping = EventTermCfg(func=mdp.randomize_actuator_gains, mode="startup", params=dict(
        asset_cfg=SceneEntityCfg("cabinet", joint_names="drawer_top_joint"),
        damping_distribution_params=(0.5, 1.5), **scale))


def apply_condition(env_cfg, name: str) -> dict:
    """Fix the selected axis of an evaluation condition (call after configure, before gym.make)."""
    condition = dict(CONDITIONS[name], name=name)
    axis = condition["axis"]
    if axis and AXES[axis]["event"]:
        event = getattr(env_cfg.events, AXES[axis]["event"])
        for key in AXES[axis]["keys"]:
            event.params[key] = (condition["value"], condition["value"])
    return condition


def measure(raw_env) -> dict[str, torch.Tensor]:
    """Arm joint-limit cost, the next state for the SPiDR ensemble and the drawer opening.

    The cost is the mean over the seven arm joints of how far each joint is inside the outer 10 %
    of its range (0 at 80 % utilisation, 1 at the limit); the gripper fingers are excluded."""
    robot, cabinet = raw_env.scene["robot"], raw_env.scene["cabinet"]
    t = (lambda v: v.torch if hasattr(v, "torch") else v)
    q, qd = t(robot.data.joint_pos), t(robot.data.joint_vel)
    arm = [i for i, name in enumerate(robot.joint_names) if name in ARM_JOINTS]
    limits = t(robot.data.joint_pos_limits)
    normalised = (q - limits.mean(-1)) / ((limits[..., 1] - limits[..., 0]) * 0.5).clamp_min(1e-6)
    cost = ((normalised[:, arm].abs() - 0.8) / 0.2).clamp(0, 1).mean(-1)
    drawer = cabinet.joint_names.index("drawer_top_joint")
    opening = t(cabinet.data.joint_pos)[:, drawer]
    state = torch.cat((normalised, qd / 5.0, opening[:, None] / 0.4, t(cabinet.data.joint_vel)[:, drawer, None]), dim=-1)
    return dict(cost=cost, state=state, drawer_position=opening)


def arm_cost(raw_env) -> torch.Tensor:
    return measure(raw_env)["cost"]


def vec_env(env, behavior_envs: int, ensemble_size: int, pessimism: float, seed: int, randomise: bool,
            clip_actions=None, fixed_delay: int | None = None) -> EnsembleVecEnv:
    """RSL-RL environment for every Drawer method except RAMU-PPO (ensemble_size 0: no ensemble)."""
    return EnsembleVecEnv(env, behavior_envs, ensemble_size, measure, pessimism=pessimism,
                          delay_range=(0, 3) if randomise else None, fixed_delay=fixed_delay,
                          clip_actions=clip_actions, seed=seed)


def rollout(env, policy) -> dict[str, list]:
    """Run the deterministic policy for the first episode of every environment."""
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    totals = torch.zeros(n, 3, device=env.device)  # return, cost, steps
    max_open = torch.zeros(n, device=env.device)
    final_open = torch.zeros(n, device=env.device)
    with torch.inference_mode():
        obs, _ = env.reset()
        for _ in range(int(env.max_episode_length)):
            obs, reward, done, extras = env.step(policy(obs))
            opening = extras["drawer_position"]
            max_open = torch.where(alive, torch.maximum(opening, max_open), max_open)
            final_open = torch.where(alive, opening, final_open)
            totals += torch.stack((reward, extras["raw_cost"], torch.ones_like(reward)), dim=-1) * alive[:, None]
            alive &= ~done.bool()
            policy.reset(done)
            if not alive.any():
                break
    if alive.any():
        raise RuntimeError("some environments did not finish an episode")
    return dict(ret=totals[:, 0].tolist(), cost=totals[:, 1].tolist(), steps=totals[:, 2].tolist(),
                final_open_m=final_open.tolist(), max_open_m=max_open.tolist())
