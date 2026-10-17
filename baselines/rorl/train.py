#!/usr/bin/env python
"""Train RORL on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/rorl_hopper.yaml --seed 0

``train.updates`` updates of the released RORL trainer on batches of ``hparams.batch_size``
dataset transitions, with observations normalised by the dataset mean and standard deviation.
"""
from __future__ import annotations

import math

from baselines.offline.common import Mean, Progress, evaluate, make_env, parse_config, setup
from baselines.offline.data import load_dataset
from baselines.rorl.agent import RORLAgent

TRAIN_DEFAULTS = {"updates": 3_000_000, "log_every": 1000, "eval_every": 5000,
                  "eval_episodes": 1, "eval_seed": 0, "save_every": 500_000}


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    hp, t = cfg["algorithm"]["hparams"], {**TRAIN_DEFAULTS, **cfg["train"]}
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = RORLAgent(env, data.states.mean(0), data.states.std(0), hp, device)
    progress, stats = Progress(out), Mean()
    for step in range(1, int(t["updates"]) + 1):
        b = data.sample(int(hp["batch_size"]))
        stats.add(agent.update({"observations": b.states, "actions": b.actions, "rewards": b.rewards,
                                "terminals": b.dones, "next_observations": b.next_states}))
        if step % int(t["log_every"]) == 0 or step == int(t["updates"]):
            row = {"update": step, "return": math.nan, **stats.pop()}
            if step % int(t["eval_every"]) == 0 or step == int(t["updates"]):
                row["return"] = evaluate(agent.act, env, t["eval_episodes"], t["eval_seed"])
            progress.write(row)
        if step % int(t["save_every"]) == 0 or step == int(t["updates"]):
            agent.save(out / "ckpt" / f"update_{step}.pt")


if __name__ == "__main__":
    main()
