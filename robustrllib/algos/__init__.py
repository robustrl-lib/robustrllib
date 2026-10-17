"""The algorithm interface and the loaders the launcher and the evaluator use.

An algorithm is named by import path in its algorithm card; the library needs no edit::

    # robustrllib/configs/algorithm/myalgo.yaml
    name: myalgo
    entry: baselines/train_algo.py           # or the method's own training script
    class: mypackage.myalgo:MyAlgo           # "module.path:ClassName"
    hparams: {lr: 0.0003}

``build_algo`` instantiates the class with ``**hparams``; ``load_algo`` calls its
``load`` classmethod on a checkpoint.
"""
import importlib

from robustrllib.algos.base import Algo


def _import_obj(path: str):
    """Import ``module.path:ClassName`` (or ``module.path.ClassName``)."""
    module, _, attr = path.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(module), attr)


def build_algo(name: str, cls: str, **hparams):
    """A new instance of the class ``cls`` (``module.path:ClassName``) of algorithm ``name``."""
    if not cls:
        raise ValueError(f"algorithm {name!r} names no class (`class: module:ClassName` in its card)")
    return _import_obj(cls)(**hparams)


def load_algo(name: str, path: str, cls: str, env=None, **kwargs):
    """The trained policy of algorithm ``name`` stored at ``path``, through ``cls.load``."""
    if not cls:
        raise ValueError(f"algorithm {name!r} names no class (`class: module:ClassName` in its card)")
    return _import_obj(cls).load(path, env=env, **kwargs)


__all__ = ["Algo", "build_algo", "load_algo"]
