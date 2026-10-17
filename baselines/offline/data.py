"""Offline datasets: load a task's dataset, apply training-time shifts, sample batches.

A task card names its dataset under ``dataset``: a Minari id (``mujoco/hopper/medium-v0``,
``D4RL/door/expert-v2``) or a ``.npz`` file with D4RL keys (``observations``, ``actions``,
``rewards``, ``next_observations``, ``terminals`` and ``timeouts``). Dict observations are
flattened with the card's ``observation.keys``, in the same order as
``robustrllib.tasks.make_env``, so the dataset and the evaluation environment share one
observation layout.

Offline training has no environment to shift, so the training-time arms of Part 2 act on
the dataset rewards (``apply_shifts``):

* ``reward`` with ``gauss`` / ``uniform`` / ``shift``: additive noise on every recorded
  reward, drawn once from the run seed with the sampler the online reward shift uses;
* ``latency`` (or ``reward``) with ``delay``: within each episode the reward is released
  every ``steps`` steps (``release: interval``) or ``steps`` steps late
  (``release: shift``); the rest is paid at the end of the episode, so every episode keeps
  its undiscounted return.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray      # [B, 1]
    next_states: np.ndarray
    dones: np.ndarray        # [B, 1], true terminations only


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray      # [T]
    next_observations: np.ndarray
    terminals: np.ndarray    # [T]


def _flatten(obs, keys):
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32)
    parts = []
    for key in keys or ["observation"]:
        value = np.asarray(obs[key], dtype=np.float32)
        parts.append(value.reshape(value.shape[0], -1))
    return np.concatenate(parts, axis=-1)


def _npz_episodes(path) -> list[Episode]:
    with np.load(path) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    ends = arrays["terminals"].reshape(-1) > 0
    if "timeouts" in arrays:
        ends |= arrays["timeouts"].reshape(-1) > 0
    cuts = list(np.flatnonzero(ends) + 1)
    if not cuts or cuts[-1] != len(ends):
        cuts.append(len(ends))
    episodes, start = [], 0
    for end in cuts:
        episodes.append(Episode(
            observations=arrays["observations"][start:end].astype(np.float32),
            actions=arrays["actions"][start:end].astype(np.float32),
            rewards=arrays["rewards"][start:end].reshape(-1).astype(np.float32),
            next_observations=arrays["next_observations"][start:end].astype(np.float32),
            terminals=arrays["terminals"][start:end].reshape(-1).astype(np.float32),
        ))
        start = end
    return episodes


def load_episodes(task) -> list[Episode]:
    """The episodes of the task card's dataset."""
    source = str(task["dataset"])
    if source.endswith(".npz"):
        if not Path(source).exists():
            raise FileNotFoundError(f"dataset {source} not found; the task card says how to create it")
        return _npz_episodes(source)
    import minari

    try:
        dataset = minari.load_dataset(source)
    except (FileNotFoundError, ValueError):
        dataset = minari.load_dataset(source, download=True)
    keys = (task.get("observation") or {}).get("keys")
    episodes = []
    for ep in dataset.iterate_episodes():
        obs = _flatten(ep.observations, keys)
        episodes.append(Episode(
            observations=obs[:-1], actions=np.asarray(ep.actions, dtype=np.float32),
            rewards=np.asarray(ep.rewards, dtype=np.float32).reshape(-1),
            next_observations=obs[1:],
            terminals=np.asarray(ep.terminations, dtype=np.float32).reshape(-1),
        ))
    return episodes


def _delay(rewards, steps: int, release: str):
    out = np.zeros_like(rewards)
    if release == "interval":
        acc = 0.0
        for t, r in enumerate(rewards):
            acc += float(r)
            if (t + 1) % steps == 0:
                out[t], acc = acc, 0.0
        out[-1] += acc
    elif release == "shift":
        out[steps:] = rewards[:-steps] if steps < len(rewards) else 0.0
        out[-1] += rewards[max(len(rewards) - steps, 0):].sum()
    else:
        raise ValueError(f"unknown reward-delay release {release!r}")
    return out


def apply_shifts(episodes: list[Episode], shifts, seed: int) -> list[Episode]:
    """Apply training-time reward shifts to the dataset rewards (see the module docstring)."""
    from robustrllib.shifts.base import sample_noise

    for spec in shifts or []:
        target, mode, params = spec["target"], spec["mode"], dict(spec.get("params") or {})
        if mode == "delay" and target in ("latency", "reward"):
            steps = int(params.get("steps", 1))
            if steps < 1:
                raise ValueError(f"reward delay needs steps >= 1, got {steps}")
            for ep in episodes:
                ep.rewards = _delay(ep.rewards, steps, params.get("release", "interval"))
        elif target == "reward" and mode in ("gauss", "uniform", "shift"):
            sizes = [len(ep.rewards) for ep in episodes]
            rng = np.random.default_rng(int(seed))
            noise = np.asarray(sample_noise(rng, mode, params, (sum(sizes),), 1.0), dtype=np.float32)
            for ep, part in zip(episodes, np.split(noise, np.cumsum(sizes)[:-1])):
                ep.rewards = (ep.rewards + part).astype(np.float32)
        else:
            raise ValueError(f"offline training applies reward and reward-delay shifts only, got {target}/{mode}")
    return episodes


class Dataset:
    """Transitions of a list of episodes, sampled uniformly with replacement."""

    def __init__(self, episodes: list[Episode], seed: int = 0):
        self.episodes = episodes
        self.states = np.concatenate([ep.observations for ep in episodes]).astype(np.float32)
        self.actions = np.concatenate([ep.actions for ep in episodes]).astype(np.float32)
        self.rewards = np.concatenate([ep.rewards for ep in episodes]).reshape(-1, 1).astype(np.float32)
        self.next_states = np.concatenate([ep.next_observations for ep in episodes]).astype(np.float32)
        self.dones = np.concatenate([ep.terminals for ep in episodes]).reshape(-1, 1).astype(np.float32)
        self.rng = np.random.default_rng(int(seed))

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def batch(self, idx) -> Batch:
        return Batch(self.states[idx], self.actions[idx], self.rewards[idx],
                     self.next_states[idx], self.dones[idx])

    def sample(self, batch_size: int) -> Batch:
        return self.batch(self.rng.integers(0, len(self), size=int(batch_size)))


def load_dataset(cfg) -> Dataset:
    """The run's dataset: the task card's data with the run's training-time shifts applied."""
    seed = int(cfg["train"]["seed"])
    episodes = apply_shifts(load_episodes(cfg["task"]), cfg["train"].get("shifts"), seed)
    data = Dataset(episodes, seed=seed)
    print(f"[data] {cfg['task']['dataset']}: {len(data)} transitions, {len(episodes)} episodes, "
          f"state {data.state_dim}, action {data.action_dim}"
          + (f", shifts {cfg['train']['shifts']}" if cfg["train"].get("shifts") else ""), flush=True)
    return data
