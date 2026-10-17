#!/usr/bin/env python
"""Train ROMB on a task's offline dataset: fit the world model, then the IQL training loop.

    python baselines/train.py -c robustrllib/configs/experiment/romb_hopper.yaml --seed 0
"""
from __future__ import annotations

from baselines.iql.train import agent_kwargs, fit
from baselines.offline.common import make_env, parse_config, setup
from baselines.offline.data import load_dataset
from baselines.romb.agent import ROMBAgent


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = ROMBAgent(env_id=cfg["task"]["env_id"], **agent_kwargs(cfg, data, env, seed, device))
    agent.prepare(data)
    fit(agent, data, cfg, out, env)


if __name__ == "__main__":
    main()
