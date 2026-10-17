#!/usr/bin/env python
"""Train one method on one task: one experiment card, one seed.

    python baselines/train.py -c robustrllib/configs/experiment/iql_hopper.yaml --seed 0
    python baselines/train.py -c robustrllib/configs/experiment/rarl_ppo_hopper.yaml --seed 0 \
        --train-shift reward_delay_16

The card is resolved into ``<out>/config.yaml`` (default ``runs/<card>/seed<k>``), and
the training script that the algorithm card names under ``entry`` runs on it. The
checkpoints land in ``<out>/ckpt/``; ``baselines/evaluate.py --run <out>`` scores them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.common import REPO, apply_overrides, resolve, write_run_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", required=True, help="experiment card")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="run directory (default runs/<card>/seed<k>)")
    p.add_argument("--train-shift", default=None,
                   help="a training-time arm of robustrllib/configs/train_shifts.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override dotted keys of the run configuration, e.g. train.steps=5000 "
                        "algorithm.hparams.learning_rate=1e-4")
    p.add_argument("--dry-run", action="store_true", help="write the run configuration and stop")
    args, extra = p.parse_known_args()

    cfg = apply_overrides(resolve(args.config, seed=args.seed, out=args.out, train_shift_name=args.train_shift),
                          args.set)
    entry = cfg["algorithm"].get("entry")
    if not entry:
        p.error(f"the algorithm card of {args.config} names no training script (entry)")
    path = write_run_config(cfg, REPO / cfg["train"]["out_dir"])
    print(f"run configuration: {path}")
    if args.dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    cmd = [sys.executable, str(REPO / entry), "--config", str(path), *extra]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
