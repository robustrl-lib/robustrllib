"""YAML <-> dataclass config for declarative benchmark runs.

Example YAML::

    env_id: Hopper-v5
    seed: 0
    make_kwargs: {max_episode_steps: 1000}
    shifts:
      - {target: dynamics, mode: scale, params: {param: gravity, factor: 1.3}}
      - {target: observation, mode: gauss, params: {sigma: 0.05}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robustrllib.spec import ShiftSpec


@dataclass
class RobustConfig:
    env_id: str
    shifts: list[ShiftSpec] = field(default_factory=list)
    seed: int | None = None
    make_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RobustConfig":
        shifts = [ShiftSpec(**s) for s in d.get("shifts", [])]
        return cls(
            env_id=d["env_id"],
            shifts=shifts,
            seed=d.get("seed"),
            make_kwargs=d.get("make_kwargs", {}),
        )

    def make(self):
        from robustrllib.pipeline import make_robust

        return make_robust(self.env_id, self.shifts, seed=self.seed, **self.make_kwargs)


def load_config(path: str) -> RobustConfig:
    import yaml

    with open(path) as f:
        return RobustConfig.from_dict(yaml.safe_load(f))
