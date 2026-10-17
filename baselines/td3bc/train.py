#!/usr/bin/env python
"""Train TD3+BC on a task's offline dataset (the training loop of baselines/iql/train.py).

    python baselines/train.py -c robustrllib/configs/experiment/td3bc_hopper.yaml --seed 0
"""
from __future__ import annotations

from baselines.iql.train import agent_kwargs, fit
from baselines.offline.common import make_env, parse_config, setup
from baselines.offline.data import load_dataset
from baselines.td3bc.agent import TD3BCAgent


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = TD3BCAgent(**agent_kwargs(cfg, data, env, seed, device))
    if cfg["algorithm"]["hparams"].get("normalize_states", True):
        agent.set_normalization(data.states.mean(0), data.states.std(0) + 1e-3)
    fit(agent, data, cfg, out, env)


if __name__ == "__main__":
    main()
