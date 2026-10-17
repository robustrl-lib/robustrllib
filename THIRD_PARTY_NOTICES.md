# Third-party notices

RobustRLlib contains code copied from, or adapted from, the projects below. Each is used
under its own licence, reproduced at the location given; the copyright of that code remains
with its authors.

## Code copied or adapted

| Where in this repository | Project | Licence |
|---|---|---|
| `robustrllib/gym/` (the environment API and the reference environments) | [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), Farama Foundation, and Robust-Gymnasium (Gu et al., ICLR 2025) | MIT |
| `robustrllib/gym/envs/robust_fetch/`, `robust_maze/`, `robust_adroit_hand/`, `robot_env.py`, `gymnasium_robotics_core.py`, `envs/utils/` | [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics); the maze and Adroit environments derive from [D4RL](https://github.com/Farama-Foundation/D4RL) | MIT (`robustrllib/gym/envs/LICENSE_GYMNASIUM_ROBOTICS`); Apache 2.0 headers in the D4RL-derived files |
| `robustrllib/gym/envs/robosuite/` (the parts DoorCausal uses) | [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT (`robustrllib/gym/envs/robosuite/LICENSE`) |
| `robustrllib/gym/envs/assets/` | the Fetch (OpenAI), Adroit and point-mass model files of Gymnasium-Robotics | see `robustrllib/gym/envs/assets/LICENSE.md` |
| `baselines/rfqi/core.py` | [RFQI](https://github.com/zaiyan-x/RFQI) | MIT (`baselines/rfqi/LICENSE`) |
| `baselines/rorl/lifelong_rl/` | [RORL](https://github.com/YangRui2015/RORL) | MIT (`baselines/rorl/LICENSE`) |
| `baselines/synther/diffusion.py` | [SynthER](https://github.com/conglu1997/SynthER) | MIT (`baselines/synther/LICENSE`) |
| `baselines/atla/{trainer,steps,models,torch_utils}.py` | [ATLA](https://github.com/huanzhang12/ATLA_robust_RL) (`src/policy_gradients`); `RunningStat` after [modular_rl](https://github.com/joschu/modular_rl) | the ATLA repository carries no licence file; its README is kept as `baselines/atla/UPSTREAM_README.md` |
| `baselines/atla/pixel.py` (`SAPPO.train`) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) `PPO.train` | MIT |
| `isaac/methods/spidr.py` (the primal-dual PPO solver) | [safe-learning](https://github.com/yardenas/safe-learning) | MIT |

## Methods reimplemented, and libraries used

IQL, TD3+BC, MOPO, RAMBO, ATLA-IQL, RSC-IQL, ROMB, FWM and PLR-PVL (`baselines/`), RARL, DR-SAC
and RSC-SAC are written from their papers; no files of their original code are copied. The
ATLA-SA schedule of the perturbation radius follows the `LinearScheduler` of
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) (BSD-3).

Used as libraries, not redistributed: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
and [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (MIT);
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
(BSD-3), with the G1 evaluation conditions taken from OmniH2O (He et al., 2024);
[openpi-baseline](https://github.com/posttraining-for-robotics/openpi-baseline) (Apache 2.0).
[policy_deployment](https://github.com/posttraining-for-robotics/policy_deployment) carries no
licence and none of its files is included; `vla/policy_deployment_mujoco237.patch` describes two
edits to a checkout of it.

## Data

The offline methods read the [Minari](https://github.com/Farama-Foundation/Minari) releases of the
D4RL tasks (as published). The VLA study uses the RSS 2026 post-training challenge dataset
(CC BY-NC 4.0), which is not redistributed here.
