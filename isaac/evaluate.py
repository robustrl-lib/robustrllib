"""Evaluate one checkpoint of any Part-4 method under its task's shift conditions.

    ./isaaclab.sh -p isaac/evaluate.py --task {g1,drawer} --checkpoint RUN/model_N.pt \
        [--training_seed S] [--condition NAME|all] [--out DIR] [--assets DIR] --headless

The policy acts deterministically (action mean) in 128 environments with a fixed evaluation seed,
and each environment contributes its first episode. ``--condition all`` evaluates the task's grid
(G1: the 31 pairs of isaac/configs/g1_conditions.tsv, named ``<terrain>_<condition>``; Drawer: the
16 conditions of isaac/tasks/drawer.py), one simulator process per condition. Each condition writes
``<out>/<condition>.json`` (default ``<checkpoint dir>/eval``). The actor architecture is read from
``params/agent.yaml`` next to the checkpoint, so checkpoints of every method load the same way.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_tasks.utils import add_launcher_args  # noqa: E402

TASKS = {"g1": "Isaac-Velocity-Flat-G1-v0", "drawer": "Isaac-Open-Drawer-Franka-v0"}
EVAL_SEED, EVAL_ENVS = 10000, 128
# The Drawer evaluation keeps the training scene layout of eight ensemble members per environment.
DRAWER_MEMBERS, DRAWER_PESSIMISM = 8, 0.075

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--training_seed", type=int, default=0, help="seed of the evaluated run, recorded in the output")
parser.add_argument("--condition", default="all")
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--assets", type=Path, default=None, help="local asset copy (isaac/tasks/assets.py)")
add_launcher_args(parser)
ARGV = list(sys.argv)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
out_dir = (args.out or args.checkpoint.parent / "eval").resolve()


def condition_names(task: str) -> list[str]:
    if task == "g1":
        from isaac.tasks.g1 import grid
        return grid()
    from isaac.tasks.drawer import CONDITIONS
    return list(CONDITIONS)


def evaluate_all() -> None:
    """Run this script once per condition (each condition needs a fresh simulator)."""
    argv, skip = [], False
    for item in ARGV:
        if skip or item.startswith("--condition="):
            skip = False
        elif item == "--condition":
            skip = True
        else:
            argv.append(item)
    for name in condition_names(args.task):
        subprocess.run([sys.executable, *argv, "--condition", name], check=False)
        if not (out_dir / f"{name}.json").exists():
            raise RuntimeError(f"condition {name} produced no result")


def policy_config(agent_cfg, checkpoint: Path) -> dict:
    """The runner configuration with the actor and critic of the trained run."""
    import yaml

    def with_class_names(saved, default):
        # Older configuration dumps can lack the class names that rsl_rl consumes.
        if isinstance(saved, dict) and isinstance(default, dict):
            saved = dict(saved)
            if "class_name" not in saved and "class_name" in default:
                saved["class_name"] = default["class_name"]
            for key in saved.keys() & default.keys():
                if key != "class_name":
                    saved[key] = with_class_names(saved[key], default[key])
        return saved

    saved = yaml.safe_load((checkpoint.parent / "params" / "agent.yaml").read_text())
    cfg = agent_cfg.to_dict()
    for key in ("actor", "critic", "obs_groups", "clip_actions"):
        cfg[key] = with_class_names(saved[key], cfg.get(key))
    return cfg


def main_single() -> None:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401

    task, name = args.task, args.condition

    @hydra_task_config(TASKS[task], "rsl_rl_cfg_entry_point")
    def run(env_cfg, agent_cfg):
        env_cfg.seed = EVAL_SEED
        env_cfg.scene.num_envs = EVAL_ENVS * (DRAWER_MEMBERS + 1 if task == "drawer" else 1)
        with launch_simulation(env_cfg, args):
            from isaac.tasks import drawer, g1
            from isaac.tasks.assets import use_local_assets

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
            env_cfg.sim.device = args.device or env_cfg.sim.device
            agent_cfg.device = env_cfg.sim.device
            use_local_assets(env_cfg, task, args.assets)
            cfg = policy_config(agent_cfg, args.checkpoint)
            if task == "g1":
                terrain, condition_name = name.split("_", 1)
                condition = g1.apply_condition(env_cfg, g1.find_condition(g1.load_suite(), condition_name), EVAL_SEED)
                if terrain == "rough":
                    g1.rough_terrain(env_cfg)
                raw = gym.make(TASKS[task], cfg=env_cfg)
                if condition.delay_steps:
                    raw = g1.ActionDelayWrapper(raw, condition.delay_steps)
                env = RslRlVecEnvWrapper(raw, clip_actions=cfg["clip_actions"])
                info = dict(terrain=terrain, condition=dict(name=condition.name, axis=condition.axis,
                            value=condition.value, region=condition.region, delay_steps=condition.delay_steps))
            else:
                drawer.configure(env_cfg, env_cfg.scene.num_envs, EVAL_SEED, randomise=name != "nominal")
                condition = drawer.apply_condition(env_cfg, name)
                delay = condition["value"] if condition["axis"] == "action_delay" else None
                env = drawer.vec_env(gym.make(TASKS[task], cfg=env_cfg), EVAL_ENVS, DRAWER_MEMBERS, DRAWER_PESSIMISM,
                                     EVAL_SEED, randomise=name != "nominal", clip_actions=cfg["clip_actions"],
                                     fixed_delay=delay)
                info = dict(condition=condition)

            # Every method and condition starts the rollout from the same random state.
            torch.manual_seed(EVAL_SEED)
            runner = OnPolicyRunner(env, cfg, log_dir=None, device=agent_cfg.device)
            runner.load(str(args.checkpoint), load_cfg={"actor": True, "critic": True})
            torch.manual_seed(EVAL_SEED)
            policy = runner.get_inference_policy(device=env.device)
            result = dict(task=TASKS[task], checkpoint=str(args.checkpoint.resolve()),
                          checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          training_seed=args.training_seed, seed=EVAL_SEED, **info)
            if task == "g1":
                episodes = g1.rollout(env, policy)
                result.update(episodes=len(episodes),
                              mean_return=sum(e["return"] for e in episodes) / len(episodes),
                              survival_rate=sum(not e["fell"] for e in episodes) / len(episodes),
                              per_episode=episodes)
            else:
                per = drawer.rollout(env, policy)
                n = len(per["ret"])
                per_step_budget = drawer.EPISODE_COST_BUDGET / int(env.max_episode_length)
                result.update(episodes=n, mean_return=sum(per["ret"]) / n,
                              success_rate=sum(x > drawer.SUCCESS_M for x in per["final_open_m"]) / n,
                              within_budget_rate=sum(c / s <= per_step_budget for c, s in zip(per["cost"], per["steps"])) / n,
                              per_episode=per)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
            env.close()

    run()


if __name__ == "__main__":
    if args.condition == "all":
        evaluate_all()
    else:
        main_single()
