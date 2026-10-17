#!/usr/bin/env python
"""Train an algorithm written against the ``Algo`` interface (``robustrllib.algos``).

    python baselines/train.py -c examples/reinforce_hopper.yaml --seed 0

The algorithm card names this script under ``entry`` and the class under ``class``::

    name: reinforce
    entry: baselines/train_algo.py
    class: robustrllib.algos.template_reinforce:Reinforce
    hparams: {lr: 0.0003, gamma: 0.99, hidden: [64, 64]}

The class is built with the card's ``hparams``, trained on the task card's environment
(with any training-time shifts) for ``train.steps`` environment steps, and saved to
``<out>/ckpt/step_<steps>.pt``. ``baselines/evaluate.py`` loads it with the class's ``load``.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.common import REPO, load_run_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration (or an experiment card)")
    cfg = load_run_config(p.parse_args().config)

    from robustrllib.algos import build_algo
    from robustrllib.tasks import make_env

    algo_card, train = cfg["algorithm"], cfg["train"]
    seed, steps = int(train["seed"]), int(train["steps"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(cfg["task"], shifts=train.get("shifts") or [], seed=seed)
    algo = build_algo(algo_card["name"], cls=algo_card["class"], **(algo_card.get("hparams") or {}))
    print(f"[train] {cfg['name']}  {algo_card['class']}  {steps} steps  seed {seed}", flush=True)
    algo.train(env, total_timesteps=steps, seed=seed)

    ckpt = REPO / train["out_dir"] / "ckpt" / f"step_{steps}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(ckpt))
    print(f"[train] saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
