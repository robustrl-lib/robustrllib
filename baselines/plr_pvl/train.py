#!/usr/bin/env python
"""Train PLR-PVL on a task's offline dataset.

    python baselines/train.py -c robustrllib/configs/experiment/plr_pvl_hopper.yaml --seed 0

The ``hparams.n_flow_models`` flow models are trained first on the run's dataset
(``<run>/flow``, settings under ``hparams.flow``), unless ``hparams.flow_dir`` holds them;
then the IQL training loop runs with the PLR-PVL agent (baselines/fwm/train.py).
"""
from __future__ import annotations

from baselines.fwm.train import run
from baselines.offline.common import parse_config
from baselines.plr_pvl.agent import PLRPVLAgent


def main() -> None:
    cfg = parse_config(__doc__)
    run(cfg, PLRPVLAgent, flow_count=int(cfg["algorithm"]["hparams"].get("n_flow_models", 3)))


if __name__ == "__main__":
    main()
