#!/usr/bin/env python
"""Train ATLA-PPO or ATLA-SA-PPO on one task.

    python baselines/train.py -c robustrllib/configs/experiment/atla_hopper.yaml --seed 0
    python baselines/atla/train.py --config runs/atla_hopper/seed0/config.yaml

``mode: adv_ppo`` is ATLA-PPO and ``mode: adv_sa_ppo`` is ATLA-SA-PPO. Vector
observations train with the ATLA trainer (``trainer.py``): ``train.steps // t``
iterations of one protagonist and one adversary rollout of ``t`` steps each.
Image observations train with the pixel trainer (``pixel.py``). The final
checkpoint is ``<out>/ckpt/step_<protagonist steps>.pt``; ``progress.csv`` logs
the mean episode return of each iteration.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baselines.common import load_run_config  # noqa: E402
from robustrllib.tasks import make_env  # noqa: E402


def train_vector(task, shifts, hparams, steps, seed, out: Path) -> Path:
    from baselines.atla.trainer import Trainer

    iterations = int(steps) // int(hparams["t"])
    params = {**hparams, "train_steps": iterations}
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    trainer = Trainer(params, lambda: make_env(task, shifts=shifts, seed=seed))
    if params["initial_std"] != 1.0:
        trainer.policy_model.log_stdev.data[:] = np.log(params["initial_std"])

    with open(out / "progress.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["iteration", "steps", "return", "adversary_rollout_return"])
        for i in range(iterations):
            ret, adv_ret = trainer.train_step()
            log.writerow([i + 1, (i + 1) * trainer.T, f"{ret:.3f}", f"{adv_ret:.3f}"])
            f.flush()
            print(f"[atla] iteration {i + 1}/{iterations}  return {ret:9.2f}  "
                  f"under the adversary's rollout {adv_ret:9.2f}", flush=True)

    ckpt = vector_checkpoint(trainer, params)
    path = out / "ckpt" / f"step_{ckpt['steps']}.pt"
    torch.save(ckpt, path)
    trainer.envs[0].close()
    return path


def vector_checkpoint(trainer, hparams) -> dict:
    """The trainer's networks, its observation normaliser and its hyperparameters."""
    from baselines.atla.torch_utils import ZFilter

    state_filter = trainer.envs[0].state_filter
    obs_filter = None
    if isinstance(state_filter, ZFilter):
        rs = state_filter.rs
        obs_filter = {"n": int(rs.n), "mean": torch.as_tensor(rs.mean), "sum_sq": torch.as_tensor(rs._S),
                      "clip": float(state_filter.clip or 0.0)}
    return {
        "backend": "vector",
        "hparams": dict(hparams),
        "obs_dim": int(trainer.NUM_FEATURES),
        "act_dim": int(trainer.NUM_ACTIONS),
        "steps": int(trainer.n_steps * trainer.T),
        "policy": trainer.policy_model.state_dict(),
        "value": trainer.val_model.state_dict(),
        "adversary_policy": trainer.adversary_policy_model.state_dict(),
        "adversary_value": trainer.adversary_val_model.state_dict(),
        "obs_filter": obs_filter,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration (or an experiment card)")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    train, task = cfg["train"], cfg["task"]
    hparams = dict(cfg["algorithm"]["hparams"])
    seed = int(train["seed"])
    out = Path(train["out_dir"])
    out = out if out.is_absolute() else REPO / out
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    probe = make_env(task, seed=seed)
    pixels = len(probe.observation_space.shape) > 1
    probe.close()
    print(f"[atla] {cfg['name']}  mode {hparams['mode']}  seed {seed}  "
          f"{'pixel' if pixels else 'vector'} observations  {int(train['steps'])} steps", flush=True)
    if pixels:
        from baselines.atla.pixel import train_pixel

        path = train_pixel(task, train["shifts"], hparams, int(train["steps"]), seed, out)
    else:
        path = train_vector(task, train["shifts"], hparams, int(train["steps"]), seed, out)
    print(f"[atla] saved {path}", flush=True)


if __name__ == "__main__":
    main()
