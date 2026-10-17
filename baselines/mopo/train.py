#!/usr/bin/env python
"""Train MOPO (Yu et al., 2020) on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/mopo_hopper.yaml --seed 0

1. fit the dynamics ensemble on the dataset;
2. run ``train.updates`` SAC updates; every ``model_train_freq`` updates, branch
   ``rollout_batch_size`` rollouts of ``rollout_length`` steps from dataset states into a
   pool that keeps ``model_retain_epochs`` refreshes, with the reward penalised by
   ``penalty_coeff`` times the ensemble uncertainty;
3. each SAC batch takes a fraction ``real_ratio`` from the dataset and the rest from the pool.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from baselines.mopo.dynamics import EnsembleDynamics, ModelEnv, ModelPool, rollout, termination_fn
from baselines.mopo.policy import save_policy
from baselines.mopo.sac import SACAgent
from baselines.offline.common import Mean, Progress, evaluate, make_env, parse_config, setup
from baselines.offline.data import load_dataset

TRAIN_DEFAULTS = {"updates": 1_000_000, "log_every": 1000, "eval_every": 5000,
                  "eval_episodes": 1, "eval_seed": 0, "save_every": 100_000}


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    hp, t = cfg["algorithm"]["hparams"], {**TRAIN_DEFAULTS, **cfg["train"]}
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    rng = np.random.default_rng(seed)

    agent = SACAgent(data.state_dim, data.action_dim, float(env.action_space.high[0]),
                     gamma=hp["gamma"], tau=hp["tau"], critic_lr=hp["lr"], actor_lr=hp["lr"], alpha_lr=hp["lr"],
                     target_entropy=hp["target_entropy"], device=device)
    model = EnsembleDynamics(data.state_dim, data.action_dim, members=hp["num_networks"], elites=hp["num_elites"],
                             live_mask=(data.next_states - data.states).std(0) > 1e-8, device=device)
    inputs = np.concatenate([data.states, data.actions], -1)
    targets = np.concatenate([data.rewards, data.next_states - data.states], -1)
    print(f"[dynamics] {model.fit(inputs, targets, rng, max_epochs=hp.get('model_max_epochs'))}", flush=True)

    model_env = ModelEnv(model, termination_fn(cfg["task"]["env_id"]), hp["penalty_coeff"],
                         hp.get("penalty_type", "max_aleatoric"))
    length, starts = int(hp["rollout_length"]), int(hp["rollout_batch_size"])
    pool = ModelPool(starts * length * int(hp["model_retain_epochs"]), data.state_dim, data.action_dim)
    batch = int(hp["batch_size"])
    n_real = int(batch * float(hp["real_ratio"]))
    progress, stats = Progress(out), Mean()
    for step in range(1, int(t["updates"]) + 1):
        if (step - 1) % int(hp["model_train_freq"]) == 0:
            start_states = torch.as_tensor(data.states[rng.integers(0, len(data), size=starts)], device=device)
            rollout(model_env, agent.policy, pool, start_states, length)
        real = data.sample(n_real)
        fake = pool.sample(batch - n_real, rng)
        s, a, r, s2, d = (torch.as_tensor(np.concatenate([x, y]), device=device) for x, y in
                          zip((real.states, real.actions, real.rewards, real.next_states, real.dones), fake))
        stats.add(agent.update(s, a, r, s2, d))
        if step % int(t["log_every"]) == 0 or step == int(t["updates"]):
            row = {"update": step, "return": math.nan, **stats.pop()}
            if step % int(t["eval_every"]) == 0 or step == int(t["updates"]):
                row["return"] = evaluate(agent.act, env, t["eval_episodes"], t["eval_seed"])
            progress.write(row)
        if step % int(t["save_every"]) == 0 or step == int(t["updates"]):
            save_policy(out / "ckpt" / f"update_{step}.pt", agent, hp, data.state_dim, data.action_dim)


if __name__ == "__main__":
    main()
