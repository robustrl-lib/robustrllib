#!/usr/bin/env python
"""Train IQL on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/iql_hopper.yaml --seed 0

``fit`` is the training loop of every IQL-backbone method: ``train.updates`` gradient
updates of batches of ``hparams.batch_size`` transitions; every ``log_every`` updates a
row of mean statistics goes to ``progress.csv`` (with the nominal return every
``eval_every`` updates), and ``ckpt/update_<n>.pt`` is written every ``save_every``
updates and at the end.
"""
from __future__ import annotations

import math

from baselines.iql.agent import IQLAgent
from baselines.offline.common import Mean, Progress, evaluate, make_env, parse_config, setup
from baselines.offline.data import load_dataset

TRAIN_DEFAULTS = {"updates": 1_000_000, "log_every": 1000, "eval_every": 5000,
                  "eval_episodes": 1, "eval_seed": 0, "save_every": 100_000}


def agent_kwargs(cfg, data, env, seed, device) -> dict:
    return dict(state_dim=data.state_dim, action_dim=data.action_dim,
                max_action=float(env.action_space.high[0]), device=device, seed=seed,
                **cfg["algorithm"]["hparams"])


def fit(agent, data, cfg, out, env, updates=None, start=0, progress=None) -> int:
    """Run ``updates`` updates (default ``train.updates``) after ``start``; returns the last update."""
    t = {**TRAIN_DEFAULTS, **cfg["train"]}
    hp = cfg["algorithm"]["hparams"]
    batch_size = int(hp.get("batch_size", 256))
    end = start + int(updates if updates is not None else t["updates"])
    progress = progress or Progress(out)
    stats = Mean()
    for step in range(start + 1, end + 1):
        stats.add(agent.update(data.sample(batch_size)))
        if step % int(t["log_every"]) == 0 or step == end:
            row = {"update": step, "return": math.nan, **stats.pop()}
            if step % int(t["eval_every"]) == 0 or step == end:
                row["return"] = evaluate(agent.act, env, t["eval_episodes"], t["eval_seed"])
            progress.write(row)
        if step % int(t["save_every"]) == 0 or step == end:
            agent.save(out / "ckpt" / f"update_{step}.pt", hp)
    return end


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = IQLAgent(**agent_kwargs(cfg, data, env, seed, device))
    fit(agent, data, cfg, out, env)


if __name__ == "__main__":
    main()
