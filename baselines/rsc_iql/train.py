#!/usr/bin/env python
"""Train RSC-IQL on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/rsc_iql_hopper.yaml --seed 0

The causal mask is fitted first on the run's dataset (``<run>/causal_mask.npz``, settings
under ``hparams.mask``), unless ``hparams.mask_path`` names an existing one; then the
IQL training loop of baselines/iql/train.py runs with the rewriting agent.
"""
from __future__ import annotations

from baselines.iql.train import agent_kwargs, fit
from baselines.offline.common import make_env, parse_config, setup
from baselines.offline.data import load_dataset
from baselines.rsc_iql.agent import RSCIQLAgent
from baselines.rsc_iql.mask import fit_mask


def main() -> None:
    cfg = parse_config(__doc__)
    seed, device, out = setup(cfg)
    data = load_dataset(cfg)
    env = make_env(cfg["task"], seed)
    kwargs = agent_kwargs(cfg, data, env, seed, device)
    if not kwargs.get("mask_path"):
        kwargs["mask_path"] = str(fit_mask(data, out / "causal_mask.npz", seed, device, **kwargs.pop("mask", {})))
    fit(RSCIQLAgent(**kwargs), data, cfg, out, env)


if __name__ == "__main__":
    main()
