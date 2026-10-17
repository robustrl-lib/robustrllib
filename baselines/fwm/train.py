#!/usr/bin/env python
"""Train FWM on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/fwm_hopper.yaml --seed 0

The flow model is trained first on the run's dataset (``<run>/flow``, settings under
``hparams.flow``), unless ``hparams.flow_dir`` holds one. With
``hparams.warm_start_updates: N`` the agent first trains N updates of plain IQL and then
``train.updates`` updates with the FWM mechanism on; otherwise FWM runs from the start.
"""
from __future__ import annotations

from baselines.fwm.agent import FWMAgent
from baselines.fwm.flow import flow_models
from baselines.iql.train import agent_kwargs, fit
from baselines.offline.common import Progress, make_env, parse_config, setup
from baselines.offline.data import load_dataset


def run(cfg, agent_cls, flow_count: int) -> None:
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    agent = agent_cls(**agent_kwargs(cfg, data, env, seed, device))
    agent.prepare(data, flow_models(cfg, data, out, flow_count, seed, device))
    warm = int(cfg["algorithm"]["hparams"].get("warm_start_updates", 0))
    progress, start = Progress(out), 0
    if warm:
        agent.mechanism = False
        start = fit(agent, data, cfg, out, env, updates=warm, progress=progress)
        agent.mechanism = True
    fit(agent, data, cfg, out, env, start=start, progress=progress)


def main() -> None:
    run(parse_config(__doc__), FWMAgent, flow_count=1)


if __name__ == "__main__":
    main()
