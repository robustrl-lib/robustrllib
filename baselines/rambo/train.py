#!/usr/bin/env python
"""Train RAMBO (Rigter et al., 2022) on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/rambo_hopper.yaml --seed 0

On normalised observations: fit the dynamics ensemble by maximum likelihood,
behaviour-clone the policy for ``bc_epochs``, then repeat epochs of ``epoch_length`` SAC
updates on batches mixing dataset (``real_ratio``) and model transitions (rollouts of
``rollout_length`` steps refreshed every ``model_train_freq`` updates, no penalty),
followed by ``epoch_length`` adversarial model updates (baselines/rambo/adversary.py).
With ``bc_reg_weight > 0`` the actor loss keeps a behaviour-cloning term on the dataset
part of each batch. Training stops early when |mean Q| exceeds ``divergence_q`` after
``divergence_check_after`` updates; the last checkpoint written before that is the
final one.
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
from baselines.rambo.adversary import AdversarialModel, StateNormalizer, pretrain_bc

TRAIN_DEFAULTS = {"updates": 2_000_000, "log_every": 1000, "eval_every": 10_000,
                  "eval_episodes": 1, "eval_seed": 0, "save_every": 100_000}


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    hp, t = cfg["algorithm"]["hparams"], {**TRAIN_DEFAULTS, **cfg["train"]}
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    rng = np.random.default_rng(seed)
    terminate = termination_fn(cfg["task"]["env_id"])

    norm = StateNormalizer(data.states, device)
    obs, next_obs = norm.normalize(data.states), norm.normalize(data.next_states)
    target_entropy = hp.get("target_entropy", "auto")
    agent = SACAgent(data.state_dim, data.action_dim, float(env.action_space.high[0]), gamma=hp["gamma"],
                     tau=hp["tau"], critic_lr=hp["critic_lr"], actor_lr=hp["actor_lr"], alpha_lr=hp["alpha_lr"],
                     target_entropy=-data.action_dim if target_entropy == "auto" else float(target_entropy),
                     device=device)
    model = EnsembleDynamics(data.state_dim, data.action_dim, members=hp["num_networks"], elites=hp["num_elites"],
                             lr=hp["model_lr"], max_logvar=hp["model_max_logvar"], min_logvar=hp["model_min_logvar"],
                             logvar_bound_coeff=hp["model_bound_reg_coeff"],
                             live_mask=(next_obs - obs).std(0) > 1e-8, device=device)
    mle_in = np.concatenate([obs, data.actions], -1)
    mle_tg = np.concatenate([data.rewards, next_obs - obs], -1)
    print(f"[dynamics] {model.fit(mle_in, mle_tg, rng, holdout_ratio=hp['model_holdout_ratio'], patience=hp['model_patience'], max_epochs=hp.get('model_max_epochs'))}", flush=True)
    if int(hp["bc_epochs"]) > 0:
        pretrain_bc(agent, obs, data.actions, rng, epochs=hp["bc_epochs"], batch_size=hp["batch_size"],
                    lr=hp["bc_lr"], objective=hp.get("bc_objective", "nll"), device=device)

    model_env = ModelEnv(model, terminate, unnormalize=norm.unnormalize)
    adversary = AdversarialModel(model, agent, norm, terminate, weight=hp["adversary_weight"], lr=hp["adv_lr"])
    length, starts = int(hp["rollout_length"]), int(hp["rollout_batch_size"])
    pool = ModelPool(starts * length * int(hp["model_retain_epochs"]), data.state_dim, data.action_dim)
    batch, epoch_length = int(hp["batch_size"]), int(hp["epoch_length"])
    n_real = int(batch * float(hp["real_ratio"]))
    bc_weight = float(hp.get("bc_reg_weight", 0.0))
    act = lambda o: agent.act(norm.normalize(np.asarray(o, dtype=np.float32)))
    progress, stats = Progress(out), Mean()
    for step in range(1, int(t["updates"]) + 1):
        if (step - 1) % int(hp["model_train_freq"]) == 0:
            rollout(model_env, agent.policy, pool,
                    torch.as_tensor(obs[rng.integers(0, len(data), size=starts)], device=device), length)
        i = rng.integers(0, len(data), size=n_real)
        fake = pool.sample(batch - n_real, rng)
        s, a, r, s2, d = (torch.as_tensor(np.concatenate([x, y]), device=device) for x, y in
                          zip((obs[i], data.actions[i], data.rewards[i], next_obs[i], data.dones[i]), fake))
        stats.add(agent.update(s, a, r, s2, d, bc_obs=s[:n_real], bc_act=a[:n_real], bc_weight=bc_weight))

        if step % epoch_length == 0 and hp["adversary_weight"]:
            done = 0
            while done < epoch_length:
                states = torch.as_tensor(obs[rng.integers(0, len(data), size=batch)], device=device)
                for _ in range(length):
                    j = rng.integers(0, len(data), size=(model.members, batch))
                    states, info = adversary.step(states, mle_in[j], mle_tg[j], rng)
                    stats.add(info)
                    done += 1
                    if done >= epoch_length:
                        break
        if step % int(t["log_every"]) == 0 or step == int(t["updates"]):
            row = {"update": step, "return": math.nan, **stats.pop()}
            if step % int(t["eval_every"]) == 0 or step == int(t["updates"]):
                row["return"] = evaluate(act, env, t["eval_episodes"], t["eval_seed"])
            progress.write(row)
            if step >= int(hp["divergence_check_after"]) and abs(row["q1_mean"]) > float(hp["divergence_q"]):
                print(f"[train] |mean Q| {abs(row['q1_mean']):.3g} > {hp['divergence_q']} at update {step}; stopping",
                      flush=True)
                break
        if step % int(t["save_every"]) == 0 or step == int(t["updates"]):
            save_policy(out / "ckpt" / f"update_{step}.pt", agent, hp, data.state_dim, data.action_dim,
                        norm.mean, norm.std)


if __name__ == "__main__":
    main()
