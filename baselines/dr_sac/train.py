#!/usr/bin/env python
"""Train DR-SAC: SAC whose training environment redraws its physical parameters.

    python baselines/train.py -c robustrllib/configs/experiment/dr_sac_hopper.yaml --seed 0

Every parameter is drawn independently at each reset and held for the episode. Two
``hparams`` describe the training distribution:

``randomize``  dynamics shifts with ``mode: uniform`` (MuJoCo, PointMaze, FetchReach,
               Pusher), stacked on the training environment by ``make_env``
``resample``   ``{attribute: [low, high]}`` for parameters an environment reads when it
               builds an episode (CarRacing friction, morphology and engine power)

The learner is SAC exactly as in ``baselines/sb3/train.py``; the checkpoints load with
``baselines.sb3.policy:SB3Policy``. Evaluation uses the nominal task and its shift grid.
"""
from __future__ import annotations

import argparse

from baselines.common import load_run_config
from baselines.sb3.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration or experiment card")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_run_config(args.config)
    hparams = dict(cfg["algorithm"].get("hparams") or {})
    randomize = hparams.pop("randomize", None) or []
    resample = hparams.pop("resample", None) or None
    for shift in randomize:
        if shift.get("target") != "dynamics" or shift.get("mode") != "uniform":
            raise ValueError(f"randomize expects dynamics shifts with mode uniform, got {shift}")
    train(cfg, "sac", hparams=hparams, shifts=randomize, resample=resample, device=args.device)


if __name__ == "__main__":
    main()
