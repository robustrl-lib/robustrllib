"""Two properties of DynamicsShift on reset.

1. The first observation reflects the shift. ``DynamicsShift.reset`` returns an
   observation computed after the shift is applied; Adroit Door's observation carries
   ``handle_pos = site_xpos[handle]``, so a frame translation must change it.

2. ``translate`` adds its offset to the pose the environment sampled at reset.
   Adroit Door randomises ``body_pos[frame]`` in ``reset_model``; the offset is added
   to that sample, not to the XML nominal.

Both tests skip if the Adroit extra is absent; the accumulation test uses Hopper.
"""

import numpy as np
import pytest

from robustrllib import ShiftSpec, make_robust


def _adroit_available():
    try:
        make_robust("AdroitHandDoor-v1").close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _translate(body, offset):
    return ShiftSpec(target="dynamics", mode="translate",
                     params={"param": "body_pos_xyz", "index": body, "offset": list(offset)})


@pytest.mark.skipif(not _adroit_available(), reason="Adroit extra not installed")
def test_translate_offsets_the_env_sampled_pose_not_the_xml_default():
    """Bug 2: the shifted frame must sit at (env's own reset sample + offset)."""
    import mujoco
    off = np.array([0.02, 0.0, 0.0])
    nominal = make_robust("AdroitHandDoor-v1")
    shifted = make_robust("AdroitHandDoor-v1", shifts=[_translate("frame", off)])
    for seed in (10000, 10001, 10002):
        nominal.reset(seed=seed)
        shifted.reset(seed=seed)
        mn, ms = nominal.unwrapped.model, shifted.unwrapped.model
        fid = mujoco.mj_name2id(mn, mujoco.mjtObj.mjOBJ_BODY, "frame")
        env_sample = np.array(mn.body_pos[fid])
        got = np.array(ms.body_pos[fid])
        # Same seed -> same env randomisation; the shift must ADD to it.
        np.testing.assert_allclose(got, env_sample + off, atol=1e-9,
            err_msg=f"seed {seed}: translate did not add to the env's own sample")
    # The env sample itself must vary across seeds (i.e. the env really does
    # randomise this body) -- otherwise the assertion above proves nothing.
    fid = mujoco.mj_name2id(nominal.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    nominal.reset(seed=10000); p0 = np.array(nominal.unwrapped.model.body_pos[fid])
    nominal.reset(seed=10001); p1 = np.array(nominal.unwrapped.model.body_pos[fid])
    assert not np.allclose(p0, p1), "Adroit Door should randomise frame pose per reset"
    nominal.close(); shifted.close()


@pytest.mark.skipif(not _adroit_available(), reason="Adroit extra not installed")
def test_reset_observation_reflects_the_shift():
    """Bug 1: the first observation must be computed AFTER the shift lands.
    Adroit Door's obs includes handle_pos, so a translated frame must change it."""
    off = np.array([0.05, 0.0, 0.0])
    nominal = make_robust("AdroitHandDoor-v1")
    shifted = make_robust("AdroitHandDoor-v1", shifts=[_translate("frame", off)])
    o0, _ = nominal.reset(seed=10000)
    o1, _ = shifted.reset(seed=10000)
    o0, o1 = np.asarray(o0, float), np.asarray(o1, float)
    changed = int((np.abs(o1 - o0) > 1e-9).sum())
    assert changed > 0, "reset observation identical to nominal after a 5 cm frame translate (stale first frame)"
    # The recomputed obs must also agree with what the env would report now.
    fresh = np.asarray(shifted.unwrapped._get_obs(), float)
    np.testing.assert_allclose(o1, fresh, atol=1e-9)
    nominal.close(); shifted.close()


def test_translate_does_not_accumulate_across_resets():
    """Per-episode base snapshot: N resets must give the same pose, not N*offset.
    Hopper never randomises foot pose, so any drift here is accumulation."""
    pytest.importorskip("mujoco")
    import mujoco
    off = np.array([0.06, 0.0, 0.0])
    env = make_robust("Hopper-v5", shifts=[_translate("foot", off)])
    m = env.unwrapped.model
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")
    poses = []
    for _ in range(3):
        env.reset(seed=1)
        poses.append(np.array(env.unwrapped.model.body_pos[fid]))
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-12)
    np.testing.assert_allclose(poses[0], poses[2], atol=1e-12)
    env.close()


def test_adapter_without_read_fails_loudly():
    """A translate on an adapter lacking read() must raise, not fall back to
    the XML nominal (which is exactly the clobbering behaviour)."""
    from robustrllib.adapters.base import DynamicsAdapter

    class _NoRead(DynamicsAdapter):
        def list_params(self): return ("x",)
        def get_nominal(self, param, index=None): return 0.0
        def set_param(self, param, value, index=None): pass

    with pytest.raises(NotImplementedError):
        _NoRead().read("x")
