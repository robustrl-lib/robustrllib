#!/usr/bin/env python
"""Size of each input-side shift, measured in the signal it corrupts.

  image axes        ||shifted - clean||_2 / ||clean||_2 over frames sampled
                    uniformly from the episode, all three cameras
  noise axes        sigma / standard deviation of that signal over the episode
                    (arm joints; recorded commands for command noise, recorded
                    joint state for proprioception noise)
  execution delay   RMS(a_t - a_{t-d}) / standard deviation of the arm
                    commands, i.e. how far the command moves during the delay

The dynamics axes perturb the world rather than a signal the policy reads or
writes; their size is the physical setting itself (ood_eval.AXES).

  python vla/dose.py --bundle ep0.pkl --out results/dose.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ood_eval import ARM, AXES, shifted_image


def image_dose(bundle: dict, frames: np.ndarray, cond: dict) -> float:
    num = den = 0.0
    for frames_of_camera in bundle["images"].values():
        for i in frames:
            clean = shifted_image(frames_of_camera[i], {}).astype(np.float64)
            shifted = shifted_image(frames_of_camera[i], cond).astype(np.float64)
            num += float(((shifted - clean) ** 2).sum())
            den += float((clean ** 2).sum())
    return float(np.sqrt(num / den))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", type=Path, required=True, help="bundle written by make_bundle.py")
    p.add_argument("--frames", type=int, default=48,
                   help="frames sampled per camera (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output")
    a = p.parse_args()

    with open(a.bundle, "rb") as f:
        bundle = pickle.load(f)
    actions = np.asarray(bundle["actions"], dtype=np.float64)[:, ARM]
    joints = np.asarray(bundle["joint_positions"], dtype=np.float64)[:, ARM]
    frames = np.linspace(0, len(actions) - 1, a.frames).astype(int)
    action_std = float(actions.std(0).mean())
    state_std = float(joints.std(0).mean())

    def delay_dose(d: int) -> float:
        return float(np.sqrt(((actions[d:] - actions[:-d]) ** 2).mean())) / action_std

    dose = {
        "brightness": {f"{v:g}": image_dose(bundle, frames, {"brightness": v})
                       for v in AXES["brightness"].rungs},
        "blur": {f"{v:g}": image_dose(bundle, frames, {"blur": v}) for v in AXES["blur"].rungs},
        "act_noise": {f"{v:g}": v / action_std for v in AXES["act_noise"].rungs},
        "proprio_noise": {f"{v:g}": v / state_std for v in AXES["proprio_noise"].rungs},
        "latency": {f"{v:g}": delay_dose(v) for v in AXES["latency"].rungs},
    }
    for axis, rungs in dose.items():
        print(f"{axis:<14}" + "  ".join(f"{r}: {100 * v:5.1f}%" for r, v in rungs.items()))
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"arm_action_std": action_std, "arm_state_std": state_std,
                                     "frames_per_camera": a.frames, "dose": dose}, indent=1) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
