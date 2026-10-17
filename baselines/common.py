"""Experiment cards: resolve one into the configuration of a single run.

An experiment card names one cell of the benchmark, a method on a task::

    name: rarl_ppo_hopper
    algorithm: robustrllib/configs/algorithm/rarl_ppo.yaml
    task: robustrllib/configs/task/hopper.yaml
    eval: robustrllib/configs/eval/part1_mujoco.yaml
    hparams: {force: 5.0, bodies: [foot]}      # merged over the algorithm card's hparams
    train: {steps: 2000000}

``algorithm``, ``task`` and ``eval`` are paths from the root of the repository or
inline mappings. ``resolve`` inlines them, merges ``hparams`` over the algorithm
card's, and adds the seed, the output directory and the training-time shifts of the
run, so that a method's training script reads one self-contained file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
TRAIN_SHIFTS = REPO / "robustrllib/configs/train_shifts.yaml"


def read_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _card(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return read_yaml(path)


def train_shift(name: str) -> list[dict]:
    """The shifts of one training-time arm of ``train_shifts.yaml``."""
    arms = read_yaml(TRAIN_SHIFTS)
    if name not in arms:
        raise KeyError(f"unknown training-time arm {name!r}; choose from {sorted(arms)}")
    return arms[name]


def resolve(card, seed: int = 0, out: str | None = None, train_shift_name: str | None = None) -> dict[str, Any]:
    """The run configuration of ``card`` for one seed."""
    exp = _card(card)
    algorithm = _card(exp.get("algorithm"))
    algorithm["hparams"] = {**(algorithm.get("hparams") or {}), **(exp.get("hparams") or {})}
    train = dict(exp.get("train") or {})
    name = exp.get("name") or Path(str(card)).stem
    if train_shift_name:
        name = f"{name}_{train_shift_name}"
    train["seed"] = int(seed)
    train["out_dir"] = str(out or f"runs/{name}/seed{seed}")
    # shifts the card itself trains under, then the named training-time arm
    train["shifts"] = list(train.get("shifts") or []) + (train_shift(train_shift_name) if train_shift_name else [])
    grid = _card(exp.get("eval"))
    if isinstance(exp.get("eval"), str):
        grid.setdefault("name", Path(exp["eval"]).stem)
    return {
        "name": name,
        "algorithm": algorithm,
        "task": _card(exp.get("task")),
        "eval": grid,
        "train": train,
    }


def apply_overrides(cfg: dict, assignments) -> dict:
    """Set dotted keys of a run configuration: ``train.steps=5000``, ``algorithm.hparams.lr=1e-4``."""
    for item in assignments or []:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_run_config(path) -> dict[str, Any]:
    """Read a run configuration; an experiment card is resolved with seed 0."""
    cfg = read_yaml(path)
    if isinstance(cfg.get("algorithm"), dict) and "seed" in (cfg.get("train") or {}):
        return cfg
    return resolve(path)


def write_run_config(cfg: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
