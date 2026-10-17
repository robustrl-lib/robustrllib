"""The algorithm interface: train on an environment, act, save, load.

``baselines/train_algo.py`` trains a subclass named in an algorithm card, and
``baselines/evaluate.py`` loads it with ``load(path, env=..., device=..., config=...)``
and acts with ``predict``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Algo(ABC):
    name: str = "base"

    def __init__(self, **hparams):
        self.hparams = hparams

    @abstractmethod
    def train(self, env, total_timesteps: int, seed: int | None = None) -> None:
        """Fit the policy on ``env``."""

    @abstractmethod
    def predict(self, obs, deterministic: bool = True):
        """Return an action for a single observation."""

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "Algo":
        """Restore a saved policy; the evaluator passes ``env``, ``device`` and the run ``config``."""
