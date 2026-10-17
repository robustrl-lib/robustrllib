"""Every evaluation card and training-time arm parses into valid shifts."""
from __future__ import annotations

import glob
import os

import pytest
import yaml

from robustrllib.spec import ShiftSpec

_CONFIGS = os.path.join(os.path.dirname(__file__), "..", "..", "robustrllib", "configs")
_EVAL_CARDS = sorted(
    glob.glob(os.path.join(_CONFIGS, "eval", "part*.yaml"))
    + glob.glob(os.path.join(_CONFIGS, "eval", "nominal.yaml"))
)


@pytest.mark.parametrize("path", _EVAL_CARDS, ids=os.path.basename)
def test_eval_card_conditions_are_valid(path):
    with open(path) as f:
        card = yaml.safe_load(f)
    assert int(card["episodes"]) > 0 and "seed" in card
    names = [c["name"] for c in card["grid"]]
    assert len(names) == len(set(names)), "condition names must be unique"
    for cond in card["grid"]:
        for s in cond.get("shifts") or []:
            ShiftSpec(**s)


def test_train_shift_arms_are_valid():
    path = os.path.join(_CONFIGS, "train_shifts.yaml")
    if not os.path.exists(path):
        pytest.skip("no train_shifts.yaml")
    with open(path) as f:
        arms = yaml.safe_load(f)
    assert arms
    for name, shifts in arms.items():
        assert shifts, name
        for s in shifts:
            ShiftSpec(**s)
