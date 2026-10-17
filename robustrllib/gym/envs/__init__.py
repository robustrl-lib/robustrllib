"""Registers the environments of the benchmark.

Entry points are import strings, so an environment's simulator is imported only when
that environment is made.
"""
from robustrllib.gym.envs.registration import make, pprint_registry, register, registry, spec
from robustrllib.gym.envs.robust_maze.maps import OPEN, U_MAZE

# MuJoCo locomotion and manipulation
register(
    id="Hopper-v5",
    entry_point="robustrllib.gym.envs.robust_mujoco.hopper_v5:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="Walker2d-v5",
    entry_point="robustrllib.gym.envs.robust_mujoco.walker2d_v5:Walker2dEnv",
    max_episode_steps=1000,
)
register(
    id="HalfCheetah-v5",
    entry_point="robustrllib.gym.envs.robust_mujoco.half_cheetah_v5:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id="Pusher-v5s",
    entry_point="robustrllib.gym.envs.robust_mujoco.pusher_v5s:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

# Box2D
register(
    id="LunarLanderContinuous-v3",
    entry_point="robustrllib.gym.envs.robust_box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)
register(
    id="CarRacing-v2",
    entry_point="robustrllib.gym.envs.robust_box2d.car_racing:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Gymnasium-Robotics
register(
    id="FetchReachDense-v3",
    entry_point="robustrllib.gym.envs.robust_fetch.reach:MujocoFetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)
register(
    id="PointMaze_UMaze-v3",
    entry_point="robustrllib.gym.envs.robust_maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": U_MAZE, "reward_type": "sparse"},
    max_episode_steps=300,
)
register(
    id="PointMaze_Open-v3",
    entry_point="robustrllib.gym.envs.robust_maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": OPEN, "reward_type": "sparse"},
    max_episode_steps=300,
)
register(
    id="AdroitHandDoor-v1",
    entry_point="robustrllib.gym.envs.robust_adroit_hand.adroit_door:AdroitHandDoorEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=200,
)

# robosuite door whose latch height is bound to the door's rotation (doorcausal.py)
for _binding in ("hhl", "hhr"):
    register(
        id=f"DoorCausal{_binding.upper()}-v0",
        entry_point="robustrllib.gym.envs.doorcausal:DoorCausalEnv",
        kwargs={"spurious_type": _binding},
        max_episode_steps=300,
    )

__all__ = ["make", "pprint_registry", "register", "registry", "spec"]
