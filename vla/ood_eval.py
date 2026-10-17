#!/usr/bin/env python
"""Shift evaluation of a served VLA policy on the bimanual insert-mouse-battery task.

The policy is scored by joint-space tracking against a recorded demonstration in
the dual-YAM MuJoCo scene of policy_deployment
(https://github.com/posttraining-for-robotics/policy_deployment), whose scene
helpers and WebSocket client are imported from a local checkout. For every
evaluation window of the bundle and every shift condition:

  1. the scene is set to the recorded joint state at the window start;
  2. the policy is queried with the recorded camera frames of the current step
     and the simulator's live joint state; observation shifts (image brightness,
     image blur, proprioception noise) are applied to what the policy receives;
  3. the returned action chunk is executed in a scene that carries the
     condition's dynamics shift (gravity, joint damping, dry friction), with
     command noise and execution delay applied at execution time;
  4. steps 2-3 repeat `--chunks` times, and the simulated joint trajectory is
     compared with the recorded joint trajectory over the same steps.

The camera frames are the recorded ones (the scene has no mouse or battery to
render), so image shifts change what the policy sees, while the other shifts
change what its actions do.

Metrics per condition (mean and s.d. over windows):
  track_rmse   RMSE between simulated and recorded 14-D joint positions
  end_l2       L2 distance between the two at the window end
  chunk_rmse   RMSE between each predicted chunk and the recorded actions
Per axis and rung, `rungs` holds the mean track_rmse over the rung's settings and
the retention nominal_track_rmse / rung_track_rmse.

Run in the openpi environment with a policy server listening (see README.md):

  python vla/ood_eval.py --policy-deployment $POLICY_DEPLOYMENT \\
      --bundle ep0.pkl --out results/ood_eval.json
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import pickle
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from PIL import Image, ImageFilter


class Axis(NamedTuple):
    source: str      # shift source the axis belongs to
    unit: str        # what a setting means
    nominal: float   # setting of the unshifted task
    rungs: tuple     # nearest -> farthest; a tuple rung is a group of settings scored by their mean


# The eight shift axes. Input-side axes have four rungs of one setting each.
# Dynamics axes have three rungs (near, far, extreme), each a group of four
# settings on both sides of nominal. The hardware operating point measured from
# the demonstration logs is the second rung of `latency` (5 steps = 83 ms) and
# of `act_noise` (0.020 rad).
AXES: dict[str, Axis] = {
    "latency":       Axis("latency",     "control steps at 60 Hz the chunk arrives late", 0,
                          (2, 5, 10, 20)),
    "act_noise":     Axis("action",      "Gaussian sigma on arm joint commands, rad", 0.0,
                          (0.01, 0.02, 0.05, 0.10)),
    "proprio_noise": Axis("observation", "Gaussian sigma on arm joint state, rad", 0.0,
                          (0.005, 0.01, 0.02, 0.05)),
    "brightness":    Axis("semantic",    "factor on pixel intensity, all cameras", 1.0,
                          (0.75, 0.5, 0.25, 0.1)),
    "blur":          Axis("semantic",    "Gaussian blur radius in pixels, all cameras", 0.0,
                          (1, 2, 3, 4)),
    "gravity":       Axis("dynamics",    "factor on gravity", 1.0,
                          ((0.8, 0.9, 1.1, 1.2), (0.6, 0.7, 1.3, 1.4), (0.25, 0.5, 1.5, 2.0))),
    "damping":       Axis("dynamics",    "factor on every joint's damping", 1.0,
                          ((2, 3, 4, 5), (8, 10, 15, 20), (50, 100, 200, 500))),
    "frictionloss":  Axis("dynamics",    "added to every joint's dry friction (frictionloss)", 0.0,
                          ((0.25, 0.5, 0.75, 1.0), (1.5, 2, 3, 4), (8, 16, 32, 64))),
}

ARM = np.r_[0:6, 7:13]   # arm joints of the 14-D layout; the grippers carry no noise
CAMERAS = {"top_camera": "cam_high", "left_camera": "cam_left_wrist",
           "right_camera": "cam_right_wrist"}   # bundle name -> policy input name
PROMPT = "Insert the battery into the mouse."
SCENE = "sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml"


def rung_settings(axis: str) -> list[tuple]:
    return [r if isinstance(r, tuple) else (r,) for r in AXES[axis].rungs]


def condition_name(axis: str, value) -> str:
    return f"{axis}_{value:g}"


def conditions(axes: list[str]) -> dict[str, dict]:
    """nominal plus one condition per setting of every rung of `axes`."""
    out = {"nominal": {}}
    for axis in axes:
        for settings in rung_settings(axis):
            for value in settings:
                out[condition_name(axis, value)] = {axis: value}
    return out


# ---------------------------------------------------------------------------
# Shift operators
# ---------------------------------------------------------------------------

def shifted_model(scene: Path, cond: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    if "gravity" in cond:
        model.opt.gravity[:] = model.opt.gravity * cond["gravity"]
    if "damping" in cond:
        model.dof_damping[:] = model.dof_damping * cond["damping"]
    if "frictionloss" in cond:
        model.dof_frictionloss[:] = model.dof_frictionloss + cond["frictionloss"]
    return model, mujoco.MjData(model)


def shifted_image(jpeg: bytes, cond: dict) -> np.ndarray:
    """Decode one recorded JPEG frame, apply brightness then blur, return CHW uint8."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if "brightness" in cond:
        arr = np.asarray(img, dtype=np.float32) * cond["brightness"]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if "blur" in cond:
        img = img.filter(ImageFilter.GaussianBlur(radius=cond["blur"]))
    return np.ascontiguousarray(np.transpose(np.asarray(img, dtype=np.uint8), (2, 0, 1)))


def execute_chunk(sandbox, model, data, ctrl_idxs, qpos_addrs, chunk: np.ndarray, *,
                  cond: dict, rng: np.random.Generator, ctrl_hz: float) -> np.ndarray:
    """Run one action chunk; return the (H, 14) simulated joint trajectory.

    With an execution delay of d steps the previous command is held for the
    first d control steps and the chunk then runs d steps late, so its last d
    actions are not executed.
    """
    inner_steps = max(1, int(round((1.0 / ctrl_hz) / model.opt.timestep)))
    delay = int(cond.get("latency", 0))
    sigma = float(cond.get("act_noise", 0.0))
    traj = np.empty((len(chunk), 14), dtype=np.float64)
    for h in range(len(chunk)):
        if h >= delay:
            a = np.array(chunk[h - delay], dtype=np.float64)
            if sigma > 0.0:
                a[ARM] += rng.normal(0.0, sigma, size=len(ARM))
            sandbox.action14_to_ctrl(a, ctrl_idxs, data)
        for _ in range(inner_steps):
            mujoco.mj_step(model, data)
        traj[h] = sandbox.read_state14(qpos_addrs, data)
    return traj


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_sandbox(root: Path):
    """Import sim/check_in_sim.py (scene helpers and WebSocket client) from a policy_deployment checkout."""
    path = root / "sim" / "check_in_sim.py"
    if not path.is_file():
        raise SystemExit(f"{root} is not a policy_deployment checkout (no sim/check_in_sim.py)")
    spec = importlib.util.spec_from_file_location("check_in_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_condition(name: str, cond: dict, bundle: dict, query, sandbox, *, scene: Path,
                       prompt: str, horizon: int, chunks: int, ctrl_hz: float, seed: int) -> dict:
    model, data = shifted_model(scene, cond)
    qpos_addrs, ctrl_idxs = sandbox._resolve_indices(model)
    joints = np.asarray(bundle["joint_positions"], dtype=np.float64)
    actions = np.asarray(bundle["actions"], dtype=np.float64)
    images = bundle["images"]
    rng = np.random.default_rng(seed)
    sigma_p = float(cond.get("proprio_noise", 0.0))

    track, end_l2, chunk_rmse = [], [], []
    for window in bundle["samples"]:
        k0 = int(window["frame_idx"])
        total = min(horizon * chunks, len(joints) - k0)
        if total < horizon:
            continue
        sandbox._snap_to_state14(model, data, qpos_addrs, ctrl_idxs,
                                 np.asarray(window["state"], dtype=np.float64))
        parts, errors = [], []
        k = k0
        while k + horizon <= k0 + total:
            state = sandbox.read_state14(qpos_addrs, data).astype(np.float64)
            if sigma_p > 0.0:
                state[ARM] += rng.normal(0.0, sigma_p, size=len(ARM))
            obs = {"images": {CAMERAS[c]: shifted_image(images[c][k], cond) for c in CAMERAS},
                   "state": np.ascontiguousarray(state), "prompt": prompt}
            chunk = np.asarray(query(obs, k), dtype=np.float64)[:horizon]
            diff = chunk - actions[k:k + len(chunk)]
            errors.append(float(np.sqrt(np.mean(diff * diff))))
            parts.append(execute_chunk(sandbox, model, data, ctrl_idxs, qpos_addrs, chunk,
                                       cond=cond, rng=rng, ctrl_hz=ctrl_hz))
            k += horizon
        traj = np.concatenate(parts, axis=0)
        ref = joints[k0:k0 + len(traj)]
        d = traj - ref
        track.append(float(np.sqrt(np.mean(d * d))))
        end_l2.append(float(np.linalg.norm(traj[-1] - ref[-1])))
        chunk_rmse.append(float(np.mean(errors)))

    out = {"track_rmse_mean": float(np.mean(track)), "track_rmse_std": float(np.std(track)),
           "end_l2_mean": float(np.mean(end_l2)), "end_l2_std": float(np.std(end_l2)),
           "chunk_rmse_mean": float(np.mean(chunk_rmse)), "n_windows": len(track)}
    print(f"{name:<22} track_rmse={out['track_rmse_mean']:.4f}  end_l2={out['end_l2_mean']:.4f}"
          f"  chunk_rmse={out['chunk_rmse_mean']:.4f}", flush=True)
    return out


def rung_summary(per_condition: dict, axes: list[str]) -> dict:
    nominal = per_condition["nominal"]["track_rmse_mean"]
    summary = {}
    for axis in axes:
        rows = []
        for i, settings in enumerate(rung_settings(axis), start=1):
            m = float(np.mean([per_condition[condition_name(axis, v)]["track_rmse_mean"]
                               for v in settings]))
            rows.append({"rung": i, "settings": list(settings), "track_rmse": m,
                         "retention": nominal / m})
        summary[axis] = rows
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy-deployment", type=Path, default=os.environ.get("POLICY_DEPLOYMENT"),
                   help="policy_deployment checkout (default: $POLICY_DEPLOYMENT)")
    p.add_argument("--bundle", type=Path, required=True, help="bundle written by make_bundle.py")
    p.add_argument("--scene", type=Path, default=None,
                   help=f"MuJoCo scene (default: <policy-deployment>/{SCENE})")
    p.add_argument("--host", default="127.0.0.1", help="policy server host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8000, help="policy server port (default: %(default)s)")
    p.add_argument("--api-key", default=None, help="API key, if the server requires one")
    p.add_argument("--prompt", default=PROMPT, help="language instruction (default: %(default)r)")
    p.add_argument("--action-horizon", type=int, default=50,
                   help="actions executed per policy query (default: %(default)s)")
    p.add_argument("--chunks", type=int, default=4,
                   help="policy queries chained per window (default: %(default)s)")
    p.add_argument("--ctrl-hz", type=float, default=60.0, help="control rate (default: %(default)s)")
    p.add_argument("--seed", type=int, default=10000, help="seed of the noise shifts (default: %(default)s)")
    p.add_argument("--axes", nargs="+", choices=list(AXES), default=list(AXES), metavar="AXIS",
                   help=f"shift axes to evaluate, from {', '.join(AXES)} (default: all); "
                        "nominal is always included")
    p.add_argument("--replay", action="store_true",
                   help="execute the recorded actions instead of querying a policy server "
                        "(checks the scene, the shifts and the metrics without a policy)")
    p.add_argument("--out", type=Path, required=True, help="output JSON")
    a = p.parse_args()

    if a.policy_deployment is None:
        p.error("pass --policy-deployment or set POLICY_DEPLOYMENT")
    root = a.policy_deployment.resolve()
    sandbox = load_sandbox(root)
    scene = a.scene or root / SCENE
    with open(a.bundle, "rb") as f:
        bundle = pickle.load(f)
    if len(bundle["images"]["top_camera"]) < len(bundle["actions"]):
        raise SystemExit(f"{a.bundle} has no camera frame for every step; build it with make_bundle.py")

    if a.replay:
        recorded = np.asarray(bundle["actions"], dtype=np.float64)
        policy_name = "recorded actions"

        def query(obs, k):
            return recorded[k:k + a.action_horizon]
    else:
        client = sandbox.WebsocketPolicyClient(host=a.host, port=a.port, api_key=a.api_key)
        policy_name = client.metadata.get("policy_name", "")

        def query(obs, k):
            return client.infer(obs)["actions"]

    per_condition = {}
    for name, cond in conditions(a.axes).items():
        per_condition[name] = evaluate_condition(
            name, cond, bundle, query, sandbox, scene=scene, prompt=a.prompt,
            horizon=a.action_horizon, chunks=a.chunks, ctrl_hz=a.ctrl_hz, seed=a.seed)

    result = {
        "task": "insert-mouse-battery",
        "policy": policy_name,
        "bundle": a.bundle.name,
        "prompt": a.prompt,
        "action_horizon": a.action_horizon,
        "chunks": a.chunks,
        "ctrl_hz": a.ctrl_hz,
        "seed": a.seed,
        "per_condition": per_condition,
        "rungs": rung_summary(per_condition, a.axes),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
