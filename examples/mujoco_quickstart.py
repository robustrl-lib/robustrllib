"""MuJoCo quick start: declare shifts as data and stack them on an environment.

    python examples/mujoco_quickstart.py               # Hopper-v5
    python examples/mujoco_quickstart.py Walker2d-v5

Every perturbation is a ``ShiftSpec`` given to ``make_robust`` at construction; the
environment keeps the standard ``reset`` / ``step(action)`` interface.
"""
import sys

import numpy as np

from robustrllib import ShiftSpec, make_robust

ENV = sys.argv[1] if len(sys.argv) > 1 else "Hopper-v5"
TORSO_GEOM = "torso_geom"


def hr(title):
    print(f"\n{'-'*66}\n{title}\n{'-'*66}")


def short_rollout(env, seed=0, n=50):
    """A standard gymnasium loop."""
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            obs, _ = env.reset()
    return total


def main():
    print(f"env = {ENV}")

    # 1) One-liner: gym.make + perturbation pipeline, standard step(action) API.
    hr("1) make_robust: one env, standard API")
    env = make_robust(ENV, shifts=[ShiftSpec("observation", "gauss", {"sigma": 0.05})], seed=0)
    print(f"  return over 50 steps = {short_rollout(env):.2f}")

    # 2) Each shift target MuJoCo supports, declared independently.
    hr("2) every shift target, one at a time")
    catalogue = {
        "observation": ShiftSpec("observation", "gauss", {"sigma": 0.05}),
        "action":      ShiftSpec("action", "uniform", {"low": -0.1, "high": 0.1}),
        "reward":      ShiftSpec("reward", "shift", {"shift": -0.01}),
        "dynamics":    ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
        "transition":  ShiftSpec("transition", "gauss", {"q_sigma": 0.01, "v_sigma": 0.02}),
    }
    for name, spec in catalogue.items():
        env = make_robust(ENV, shifts=[spec], seed=0)
        print(f"  {name:12} return = {short_rollout(env):.2f}")

    # 3) Stack any number -- order = env -> out (dynamics first feeds the obs).
    hr("3) stack multiple shifts")
    env = make_robust(ENV, shifts=[
        ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
        ShiftSpec("dynamics", "scale", {"param": "body_mass", "index": "torso", "factor": 1.5}),
        ShiftSpec("observation", "gauss", {"sigma": 0.05}),
        ShiftSpec("reward", "gauss", {"sigma": 0.01}),
    ], seed=0)
    print(f"  4 stacked shifts, return = {short_rollout(env):.2f}")

    # 4) Dynamics is edited in memory (no XML rewrite); addressable by name.
    hr("4) dynamics shift edits the live model (by name)")
    clean = make_robust(ENV, shifts=[], seed=0); clean.reset(seed=0)
    shifted = make_robust(ENV, shifts=[
        ShiftSpec("dynamics", "scale", {"param": "geom_size", "index": TORSO_GEOM, "factor": 1.4}),
    ], seed=0); shifted.reset(seed=0)
    import mujoco
    gid = mujoco.mj_name2id(clean.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, TORSO_GEOM)
    print(f"  {TORSO_GEOM} size: {clean.unwrapped.model.geom_size[gid,0]:.4f} -> "
          f"{shifted.unwrapped.model.geom_size[gid,0]:.4f} (x1.4)")

    # 5) Non-stationary ("dynamic") shift: intensity varies over time via schedule.
    hr("5) non-stationary schedule: gravity ramps over steps")
    env = make_robust(ENV, shifts=[ShiftSpec(
        "dynamics", "scale", {"param": "gravity", "factor": 1.0},
        schedule={"type": "linear", "start": 1.0, "end": 2.0, "t0": 0, "t1": 40},
    )], seed=0)
    env.reset(seed=0)
    g = clean.unwrapped.model.opt.gravity[2]
    for t in (0, 20, 40):
        for _ in range(20 if t else 0):
            env.step(env.action_space.sample())
        print(f"  step {t:2d}: gravity_z = {env.unwrapped.model.opt.gravity[2]:.3f} "
              f"(nominal {g:.3f})")

    # 6) Reproducibility: same seed -> identical rollout across the whole stack.
    hr("6) reproducibility")
    def run():
        e = make_robust(ENV, shifts=[
            ShiftSpec("dynamics", "scale", {"param": "gravity", "factor": 1.3}),
            ShiftSpec("observation", "gauss", {"sigma": 0.05}),
        ], seed=0)
        e.action_space.seed(0)
        obs, _ = e.reset(seed=0)
        tr = [np.asarray(obs, np.float64)]
        for _ in range(30):
            obs, *_ = e.step(e.action_space.sample())
            tr.append(np.asarray(obs, np.float64))
        return np.stack(tr)
    print(f"  two seeded rollouts identical: {np.allclose(run(), run())}")

    print("\nAll sections ran.")


if __name__ == "__main__":
    main()
