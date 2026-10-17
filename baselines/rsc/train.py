#!/usr/bin/env python
"""Train RSC-SAC on one task.

    python baselines/train.py -c robustrllib/configs/experiment/rsc_hopper.yaml --seed 0

``baselines/train.py`` resolves the experiment card and runs this script on the run
configuration. The training environment is ``robustrllib.tasks.make_env`` of the task
card with the run's training-time shifts. Writes ``<out>/ckpt/step_<steps>.zip`` (the
SAC policy), ``<out>/causal_model.pt`` and ``<out>/progress.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from baselines.common import REPO, load_run_config  # noqa: E402
from baselines.rsc.rsc import RSCSAC  # noqa: E402
from robustrllib.tasks import make_env  # noqa: E402

RSC_KEYS = ("actor_lr", "critic_lr", "lr_final", "warmup", "ramp", "cooldown", "mix",
            "selected_fraction", "reference_size", "protect_last", "dynamics_lr", "dynamics")


def build(env, hp: dict, seed: int, device: str) -> RSCSAC:
    images = len(env.observation_space.shape) == 3
    policy_kwargs = {"net_arch": list(hp["net_arch"])}
    if hp.get("log_std_init") is not None:
        policy_kwargs["log_std_init"] = float(hp["log_std_init"])
    target_entropy = hp.get("target_entropy", "auto")
    return RSCSAC(
        "CnnPolicy" if images else "MlpPolicy",
        env,
        rsc={k: hp.get(k) for k in RSC_KEYS},
        buffer_size=int(hp["buffer_size"]),
        batch_size=int(hp["batch_size"]),
        learning_starts=int(hp["learning_starts"]),
        train_freq=int(hp["train_freq"]),
        gradient_steps=int(hp["gradient_steps"]),
        gamma=float(hp["gamma"]),
        tau=float(hp["tau"]),
        target_entropy=target_entropy if target_entropy == "auto" else float(target_entropy),
        use_sde=bool(hp.get("use_sde", False)),
        sde_sample_freq=int(hp.get("sde_sample_freq", -1)),
        use_sde_at_warmup=bool(hp.get("use_sde", False)),
        # images are stored once (the next observation is the following entry)
        optimize_memory_usage=images,
        replay_buffer_kwargs={"handle_timeout_termination": False} if images else None,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration (or an experiment card)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    hp, train = cfg["algorithm"]["hparams"], cfg["train"]
    seed, steps = int(train["seed"]), int(train["steps"])
    out = Path(train["out_dir"])
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    env = make_env(cfg["task"], shifts=train.get("shifts") or [], seed=seed)
    model = build(env, hp, seed, device)
    model.set_logger(configure(str(out), ["csv"]))
    print(f"[rsc] {cfg['name']} seed {seed}: {steps} steps on {device}", flush=True)
    model.learn(total_timesteps=steps, log_interval=10)
    model.logger.dump(model.num_timesteps)

    (out / "ckpt").mkdir(exist_ok=True)
    path = out / "ckpt" / f"step_{model.num_timesteps}.zip"
    model.save(path)
    torch.save(model.causal_model.state_dict(), out / "causal_model.pt")
    env.close()
    print(f"[rsc] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
