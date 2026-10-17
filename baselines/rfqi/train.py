#!/usr/bin/env python
"""Train RFQI on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/rfqi_hopper.yaml --seed 0

``train.updates`` updates of batches of ``hparams.batch_size``; every ``eval_every`` updates
the policy's nominal return over ``eval_episodes`` episodes is measured, and the policy
with the highest one so far is kept as ``ckpt/best.pt`` (the checkpoint RFQI reports).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from baselines.offline.common import Mean, Progress, evaluate, make_env, parse_config, setup
from baselines.offline.data import load_dataset
from baselines.rfqi.core import RFQI

TRAIN_DEFAULTS = {"updates": 500_000, "log_every": 1000, "eval_every": 5000,
                  "eval_episodes": 10, "eval_seed": 0, "save_every": 100_000}


def save(path, agent, hparams) -> None:
    torch.save({"hparams": dict(hparams), "dims": [agent.state_dim, agent.action_dim],
                "max_action": agent.max_action.cpu().numpy(), "agent": agent.state_dict()}, path)


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    hp, t = cfg["algorithm"]["hparams"], {**TRAIN_DEFAULTS, **cfg["train"]}
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = RFQI(data.state_dim, data.action_dim, env.action_space.high, device, **hp)
    batch, best = int(hp["batch_size"]), -math.inf
    progress, stats = Progress(out), Mean()
    for step in range(1, int(t["updates"]) + 1):
        b = data.sample(batch)
        stats.add(agent.update(*(torch.as_tensor(x, device=device) for x in
                                 (b.states, b.actions, b.next_states, b.rewards))))
        if step % int(t["log_every"]) == 0 or step == int(t["updates"]):
            row = {"update": step, "return": math.nan, **stats.pop()}
            if step % int(t["eval_every"]) == 0 or step == int(t["updates"]):
                row["return"] = evaluate(agent.act, env, t["eval_episodes"], t["eval_seed"])
                if row["return"] > best:
                    best = row["return"]
                    save(out / "ckpt" / "best.pt", agent, hp)
            progress.write(row)
        if step % int(t["save_every"]) == 0 or step == int(t["updates"]):
            save(out / "ckpt" / f"update_{step}.pt", agent, hp)


if __name__ == "__main__":
    main()
