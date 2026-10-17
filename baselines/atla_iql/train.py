#!/usr/bin/env python
"""Train ATLA-IQL on a task's offline dataset (the IQL training loop, baselines/iql/train.py).

    python baselines/train.py -c robustrllib/configs/experiment/atla_iql_hopper.yaml --seed 0
"""
from __future__ import annotations

from baselines.atla_iql.agent import ATLAIQLAgent
from baselines.iql.train import agent_kwargs, fit
from baselines.offline.common import make_env, parse_config, setup
from baselines.offline.data import load_dataset


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    fit(ATLAIQLAgent(**agent_kwargs(cfg, data, env, seed, device)), data, cfg, out, env)


if __name__ == "__main__":
    main()
