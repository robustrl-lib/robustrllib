#!/usr/bin/env python
"""Train PPO or SAC (Stable-Baselines3) on one task.

    python baselines/train.py -c robustrllib/configs/experiment/ppo_hopper.yaml --seed 0

Reads a run configuration (``baselines/common.py``). The algorithm's ``hparams`` are
Stable-Baselines3 constructor arguments, plus four keys handled here:

``normalize``          running normalisation of observations and rewards (``VecNormalize``);
                       the observation statistics are stored in the checkpoint and applied,
                       frozen, by the loader
``n_envs``             number of training environments (default 1)
``learning_rate_end``  decay the learning rate linearly from ``learning_rate`` to this value
``policy``             network; default ``CnnPolicy`` for image observations, else ``MlpPolicy``

The training environments come from ``robustrllib.tasks.make_env`` with the run's
training-time shifts. Training stops after ``train.steps`` environment steps and
writes ``<out>/ckpt/step_<steps>.zip``; ``<out>/progress.csv`` logs the mean return
of recent training episodes.
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from baselines.common import load_run_config
from robustrllib.tasks import make_env

ALGORITHMS = {"ppo": PPO, "sac": SAC}
OWN_KEYS = ("normalize", "n_envs", "learning_rate_end", "policy")


class Budget(BaseCallback):
    """Stop after ``steps`` environment steps; log recent training returns."""

    def __init__(self, steps: int, progress: Path, every: int = 10_000):
        super().__init__()
        self.steps, self.progress, self.every = int(steps), Path(progress), int(every)
        self.episodes = 0
        self.logged = 0

    def _on_training_start(self) -> None:
        with open(self.progress, "w", newline="") as f:
            csv.writer(f).writerow(["step", "episodes", "return_mean"])

    def _on_step(self) -> bool:
        self.episodes += sum("episode" in info for info in self.locals.get("infos", []))
        done = self.num_timesteps >= self.steps
        if done or self.num_timesteps // self.every > self.logged // self.every:
            self.logged = self.num_timesteps
            returns = [ep["r"] for ep in self.model.ep_info_buffer]
            with open(self.progress, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, self.episodes,
                                        f"{np.mean(returns):.4f}" if returns else ""])
        return not done


def training_env(task, shifts, seed: int, n_envs: int = 1, normalize: bool = False,
                 gamma: float = 0.99, resample=None):
    """``n_envs`` copies of the task (seeds ``seed .. seed + n_envs - 1``), optionally normalised."""
    venv = VecMonitor(DummyVecEnv([
        partial(make_env, task, shifts=shifts, seed=seed + i, resample=resample)
        for i in range(n_envs)
    ]))
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=gamma)
    return venv


def build_model(algorithm: str, venv, hparams: dict, seed: int, device: str = "auto"):
    kwargs = {k: v for k, v in hparams.items() if k not in OWN_KEYS}
    if hparams.get("learning_rate_end") is not None:
        kwargs["learning_rate"] = LinearSchedule(float(kwargs["learning_rate"]),
                                                 float(hparams["learning_rate_end"]), 1.0)
    if kwargs.get("optimize_memory_usage"):
        kwargs.setdefault("replay_buffer_kwargs", {"handle_timeout_termination": False})
    policy = hparams.get("policy") or ("CnnPolicy" if len(venv.observation_space.shape) == 3
                                       else "MlpPolicy")
    return ALGORITHMS[algorithm](policy, venv, seed=seed, device=device, verbose=0, **kwargs)


def train(cfg: dict, algorithm: str, hparams: dict | None = None, shifts=(), resample=None,
          device: str = "auto") -> Path:
    """Train on the run configuration ``cfg``; ``shifts`` are stacked before the run's own."""
    hparams = dict(cfg["algorithm"].get("hparams") or {}) if hparams is None else dict(hparams)
    seed = int(cfg["train"]["seed"])
    steps = int(cfg["train"]["steps"])
    out = Path(cfg["train"]["out_dir"])
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    venv = training_env(cfg["task"], list(shifts) + list(cfg["train"].get("shifts") or []), seed,
                        n_envs=int(hparams.get("n_envs", 1)),
                        normalize=bool(hparams.get("normalize", False)),
                        gamma=float(hparams.get("gamma", 0.99)), resample=resample)
    model = build_model(algorithm, venv, hparams, seed, device)
    print(f"[{algorithm}] {cfg['name']} seed {seed}: {steps} steps, "
          f"observation {venv.observation_space.shape}, action {venv.action_space}", flush=True)
    model.learn(total_timesteps=steps, callback=Budget(steps, out / "progress.csv"))

    if isinstance(venv, VecNormalize):
        model.obs_normalization = {
            "mean": venv.obs_rms.mean.tolist(), "var": venv.obs_rms.var.tolist(),
            "clip": float(venv.clip_obs), "epsilon": float(venv.epsilon),
        }
    path = out / "ckpt" / f"step_{model.num_timesteps}.zip"
    model.save(path)
    venv.close()
    print(f"[{algorithm}] wrote {path}", flush=True)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration or experiment card")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_run_config(args.config)
    train(cfg, cfg["algorithm"]["name"], device=args.device)


if __name__ == "__main__":
    main()
