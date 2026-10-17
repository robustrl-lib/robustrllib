"""Run setup, progress logging and the nominal progress evaluation of the offline trainers."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from baselines.common import load_run_config


def parse_config(description: str | None = None) -> dict:
    """``--config <run configuration or experiment card>`` -> the resolved run configuration."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", required=True)
    return load_run_config(parser.parse_args().config)


def setup(cfg) -> tuple[int, torch.device, Path]:
    """Seed every generator from the run seed; return (seed, device, run directory)."""
    seed = int(cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg["train"].get("threads", 4)))
    device = torch.device(cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(cfg["train"]["out_dir"])
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    print(f"[run] {cfg['name']} seed {seed} on {device} -> {out}", flush=True)
    return seed, device, out


def make_env(task, seed: int):
    from robustrllib.tasks import make_env as _make_env

    return _make_env(task, seed=seed)


def evaluate(act, env, episodes: int, seed: int) -> float:
    """Mean nominal return of ``act(obs) -> action``; episode ``i`` resets with ``seed + i``."""
    returns = []
    for i in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed) + i)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(obs))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


class Progress:
    """Rows of training statistics in ``<run>/progress.csv``."""

    def __init__(self, out: Path):
        self.path = Path(out) / "progress.csv"
        self.fields: list[str] = []
        self.rows: list[dict] = []

    def write(self, row: dict) -> None:
        self.rows.append(row)
        new = [k for k in row if k not in self.fields]
        if new:  # a statistic appears for the first time: rewrite the file with the wider header
            self.fields += new
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, self.fields).writerow(row)
        shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"[train] {shown}", flush=True)


class Mean:
    """Running mean of the numeric statistics returned by the updates of one logging window."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, stats: dict) -> None:
        for key, value in stats.items():
            if isinstance(value, (int, float, np.floating)):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def pop(self) -> dict:
        out = {k: self.sums[k] / self.counts[k] for k in self.sums}
        self.sums, self.counts = {}, {}
        return out
