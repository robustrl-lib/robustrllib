"""Train one Part-4 policy in Isaac Lab.

    ./isaaclab.sh -p isaac/train.py --task {g1,drawer} --method {ppo,dr,atla,ramu,spidr} --seed S \
        [--out DIR] [--assets DIR] [--iterations N] [--num_envs N] --headless

Methods: ``ppo`` (no randomisation), ``dr`` (training randomisation), ``atla`` (ATLA-PPO),
``ramu`` (RAMU-PPO), ``spidr`` (SPiDR-PPO). The default budget of every cell is the paper's
(iterations x environments x steps per environment); ``--iterations``/``--num_envs`` shrink it.
Checkpoints ``model_<iteration>.pt`` and ``params/{env,agent}.yaml`` are written to ``--out``
(default ``runs/isaac/<task>_<method>/seed<S>``). The Drawer task needs ``--assets`` (see
isaac/tasks/assets.py); for G1 it is optional.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_tasks.utils import add_launcher_args  # noqa: E402

TASKS = {"g1": "Isaac-Velocity-Flat-G1-v0", "drawer": "Isaac-Open-Drawer-Franka-v0"}
METHODS = ("ppo", "dr", "atla", "ramu", "spidr")
# (iterations, environments, steps per environment). ATLA collects two rollouts per iteration and
# SPiDR adds eight ensemble members per environment that do not count as training interactions.
BUDGET = {
    ("g1", "ppo"): (1500, 4096, 24), ("g1", "dr"): (1500, 4096, 24), ("g1", "atla"): (750, 4096, 24),
    ("g1", "ramu"): (1500, 4096, 24), ("g1", "spidr"): (900, 512, 320),
    ("drawer", "ppo"): (3200, 512, 96), ("drawer", "dr"): (3200, 512, 96), ("drawer", "atla"): (1600, 512, 96),
    ("drawer", "ramu"): (3200, 512, 96), ("drawer", "spidr"): (3200, 512, 96),
}
ENSEMBLE_SIZE, PESSIMISM = 8, 0.075

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--method", choices=METHODS, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--assets", type=Path, default=None, help="local asset copy (isaac/tasks/assets.py)")
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=None, help="training (behaviour) environments")
add_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401


def spidr_recipe(agent_cfg) -> None:
    """Network and PPO settings of the SPiDR reference implementation (G1)."""
    for model in (agent_cfg.actor, agent_cfg.critic):
        model.hidden_dims, model.activation, model.obs_normalization = [512, 256, 128], "swish", True
    alg = agent_cfg.algorithm
    alg.learning_rate, alg.schedule, alg.gamma, alg.lam = 3e-4, "fixed", 0.97, 0.95
    alg.entropy_coef, alg.num_learning_epochs, alg.num_mini_batches = 0.01, 4, 32
    alg.clip_param, alg.max_grad_norm = 0.3, 1.0


@hydra_task_config(TASKS[args.task], "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    task, method, seed = args.task, args.method, args.seed
    iterations, num_envs, steps = BUDGET[(task, method)]
    iterations, num_envs = args.iterations or iterations, args.num_envs or num_envs
    out = (args.out or Path("runs/isaac") / f"{task}_{method}" / f"seed{seed}").resolve()
    members = ENSEMBLE_SIZE if method == "spidr" else 0
    agent_cfg.seed, agent_cfg.max_iterations, agent_cfg.num_steps_per_env = seed, iterations, steps
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs * (members + 1)

    with launch_simulation(env_cfg, args):
        from isaac.methods.atla import ATLARunner
        from isaac.methods.ramu import RAMUVecEnv
        from isaac.methods.spidr import EnsembleVecEnv
        from isaac.tasks import drawer, g1
        from isaac.tasks.assets import use_local_assets

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.sim.device = args.device or env_cfg.sim.device
        agent_cfg.device = env_cfg.sim.device
        use_local_assets(env_cfg, task, args.assets)
        if task == "g1":
            if method == "spidr":
                g1.omnih2o_training(env_cfg, seed)
                spidr_recipe(agent_cfg)
            elif method != "dr":
                g1.nominal(env_cfg)
        else:
            drawer.configure(env_cfg, num_envs * (members + 1), seed, randomise=method not in ("ppo", "ramu"))
            agent_cfg.algorithm.schedule = "fixed"
            agent_cfg.algorithm.learning_rate = 1e-4 if method == "ramu" else 5e-4

        raw = gym.make(TASKS[task], cfg=env_cfg)
        clip = agent_cfg.clip_actions
        if method == "ramu":
            env = RAMUVecEnv(raw, g1.joint_limit_cost if task == "g1" else drawer.arm_cost, clip)
        elif task == "g1" and method == "spidr":
            env = EnsembleVecEnv(raw, num_envs, members, g1.ensemble_measure, pessimism=PESSIMISM, delay_range=(1, 4),
                                 push_interval_s=5.0, clip_actions=clip, seed=seed)
        elif task == "drawer":
            env = drawer.vec_env(raw, num_envs, members, PESSIMISM if members else 0.0, seed,
                                 randomise=method != "ppo", clip_actions=clip)
        else:
            env = RslRlVecEnvWrapper(raw, clip_actions=clip)

        cfg = agent_cfg.to_dict()
        episode_length = int(env.max_episode_length)
        if method == "ramu":
            cfg["algorithm"].update(class_name="isaac.methods.ramu:RAMUPPO", feature_masks=env.feature_masks,
                                    sampling_seed=seed + 100000, episode_length=episode_length)
        elif method == "spidr" or task == "drawer":
            cfg["algorithm"].update(class_name="isaac.methods.spidr:PrimalDualPPO", constrained=method == "spidr",
                                    episode_length=episode_length, unroll_length=20 if task == "g1" else 24)
        out.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(out / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(out / "params" / "agent.yaml"), cfg)

        # rsl_rl consumes the class names of the configuration, so the runner gets a copy.
        torch.manual_seed(seed)
        if method == "atla":
            runner = ATLARunner(env, copy.deepcopy(cfg), log_dir=str(out), device=agent_cfg.device,
                                attack_scale=drawer.ATTACK_SCALE if task == "drawer" else None,
                                timeout_value="current" if task == "drawer" else "next")
        else:
            runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=str(out), device=agent_cfg.device)
        torch.manual_seed(seed)
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        env.close()


if __name__ == "__main__":
    main()
