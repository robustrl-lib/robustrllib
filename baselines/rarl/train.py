#!/usr/bin/env python
"""Train RARL: a protagonist and an adversary, PPO or TRPO, in alternating update blocks.

    python baselines/train.py -c robustrllib/configs/experiment/rarl_ppo_hopper.yaml --seed 0

Each cycle runs ``update_ratio[0]`` protagonist blocks, then ``update_ratio[1]``
adversary blocks; a block is one rollout of ``n_steps x n_envs`` steps followed by one
update. The adversary stays off for the first ``warmup`` protagonist steps. The budget
``train.steps`` counts protagonist environment steps; when it is reached the protagonist
is saved to ``<out>/ckpt/step_<steps>.zip`` (with ``ckpt/vecnorm.pkl`` when the card asks
for observation normalisation) and training stops.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baselines.common import load_run_config
from baselines.rarl.game import Player, PlayerView
from robustrllib.tasks import make_env

AGENT_KEYS = {
    "ppo": ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "max_grad_norm"),
    "trpo": ("n_steps", "batch_size", "learning_rate", "gamma", "gae_lambda", "target_kl",
             "n_critic_updates", "cg_max_steps", "cg_damping"),
}


def guarded_trpo():
    """TRPO that keeps the previous policy when an update yields an invalid distribution."""
    from sb3_contrib import TRPO

    class GuardedTRPO(TRPO):
        def train(self) -> None:
            policy = copy.deepcopy(self.policy.state_dict())
            optimizer = copy.deepcopy(self.policy.optimizer.state_dict())
            try:
                super().train()
                ok = all(torch.isfinite(p).all() for p in self.policy.parameters())
            except ValueError:      # non-finite parameters make the action distribution invalid
                ok = False
            if not ok:
                self.policy.load_state_dict(policy)
                self.policy.optimizer.load_state_dict(optimizer)

    return GuardedTRPO


def make_agent(hp, venv, seed, learning_rate):
    kwargs = {k: hp[k] for k in AGENT_KEYS[hp["base"]] if k in hp}
    kwargs["learning_rate"] = learning_rate
    cls = PPO if hp["base"] == "ppo" else guarded_trpo()
    policy = "CnnPolicy" if len(venv.observation_space.shape) == 3 else "MlpPolicy"
    return cls(policy, venv, policy_kwargs=dict(hp.get("policy_kwargs") or {}),
               seed=seed, verbose=0, **kwargs)


class SaveAtBudget(BaseCallback):
    """Save the protagonist when it has taken ``steps`` environment steps, then stop."""

    def __init__(self, steps, ckpt_dir: Path, vecnorm=None):
        super().__init__()
        self.steps, self.ckpt_dir, self.vecnorm = int(steps), ckpt_dir, vecnorm
        self.saved = None

    def _on_step(self) -> bool:
        if self.num_timesteps < self.steps:
            return True
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.saved = self.ckpt_dir / f"step_{self.num_timesteps}.zip"
        self.model.save(self.saved)
        if self.vecnorm is not None:
            self.vecnorm.save(str(self.ckpt_dir / "vecnorm.pkl"))
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration (or an experiment card)")
    cfg = load_run_config(p.parse_args().config)
    hp, task, train = cfg["algorithm"]["hparams"], cfg["task"], cfg["train"]
    seed, steps, out = int(train["seed"]), int(train["steps"]), Path(train["out_dir"])
    n_envs = int(hp["n_envs"])
    set_random_seed(seed)

    protagonist, adversary = Player(), Player()

    def view(role, other, env_seed):
        def build():
            env = make_env(task, shifts=train.get("shifts") or [], seed=env_seed)
            env = PlayerView(env, hp["adversary"], role, other,
                             zero_force_prob=hp["zero_force_prob"] if role == "protagonist" else 0.0,
                             seed=env_seed)
            return Monitor(env) if role == "protagonist" else env
        return build

    # the two players get disjoint environment seeds: seed + i and seed + n_envs + i
    pro_venv = DummyVecEnv([view("protagonist", adversary, seed + i) for i in range(n_envs)])
    adv_venv = DummyVecEnv([view("adversary", protagonist, seed + n_envs + i) for i in range(n_envs)])
    if hp["normalize"]:
        pro_venv = VecNormalize(pro_venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hp["gamma"])
        protagonist.vecnorm = pro_venv
    lr = float(hp["learning_rate"])
    protagonist.model = make_agent(hp, pro_venv, seed, lr)
    adversary.model = make_agent(hp, adv_venv, seed + n_envs, float(hp.get("adversary_lr") or lr))
    pro, adv = protagonist.model, adversary.model

    budget = SaveAtBudget(steps, out / "ckpt", protagonist.vecnorm)
    block = int(hp["n_steps"]) * n_envs
    n_pro, n_adv = (int(k) for k in hp["update_ratio"])
    floor = hp.get("log_std_floor")
    warmup = int(hp["warmup"])
    out.mkdir(parents=True, exist_ok=True)
    progress = open(out / "progress.csv", "w", newline="")
    log = csv.writer(progress)
    log.writerow(["steps", "episode_return"])
    print(f"[rarl] {cfg['name']}  {hp['base']}  seed {seed}  {steps} protagonist steps  "
          f"adversary {hp['adversary']}  ratio {n_pro}:{n_adv}  warmup {warmup}  "
          f"p0 {hp['zero_force_prob']}  block {block}", flush=True)

    def update(agent, callback=None):
        agent.learn(total_timesteps=block, callback=callback, reset_num_timesteps=False)
        if floor is not None and hasattr(agent.policy, "log_std"):
            with torch.no_grad():
                agent.policy.log_std.clamp_(min=float(floor))

    while budget.saved is None:
        adversary.active = pro.num_timesteps >= warmup
        for _ in range(n_pro):
            update(pro, budget)
            if budget.saved is not None:
                break
            ret = np.mean([e["r"] for e in pro.ep_info_buffer]) if pro.ep_info_buffer else float("nan")
            log.writerow([pro.num_timesteps, f"{ret:.4f}"])
            progress.flush()
            print(f"[rarl] steps {pro.num_timesteps:>9}  episode return {ret:10.2f}", flush=True)
        if budget.saved is None and pro.num_timesteps >= warmup:
            adversary.active = True
            for _ in range(n_adv):
                update(adv)
    progress.close()
    pro_venv.close()
    adv_venv.close()
    print(f"[rarl] saved {budget.saved}", flush=True)


if __name__ == "__main__":
    main()
