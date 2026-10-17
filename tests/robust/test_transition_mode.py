"""The transition target has one form, Gaussian state noise, and says so."""

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust


def test_gauss_perturbs_the_simulator_state():
    clean = make_robust("Hopper-v5", seed=0)
    noisy = make_robust("Hopper-v5", seed=0, shifts=[
        ShiftSpec("transition", "gauss", {"q_sigma": 0.01, "v_sigma": 0.02})])
    clean.reset(seed=0); noisy.reset(seed=0)
    a = np.zeros(clean.action_space.shape)
    clean.step(a); noisy.step(a)
    assert not np.array_equal(clean.unwrapped.data.qpos, noisy.unwrapped.data.qpos)
    clean.close(); noisy.close()


@pytest.mark.parametrize("mode", ["uniform", "shift", "adversarial", "delay", "scale"])
def test_any_other_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="write mode 'gauss'"):
        make_robust("Hopper-v5", shifts=[ShiftSpec("transition", mode, {"sigma": 0.01})])
