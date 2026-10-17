#!/usr/bin/env python
"""Build an evaluation bundle from one episode of a LeRobot (v2.1) dataset.

The bundle is a pickled dict in the format of policy_deployment's sim bundles,
with a camera frame for every step:

  actions          (T, 14) float32   recorded commands (`action`)
  joint_positions  (T, 14) float32   recorded joint state (`observation.state`)
  timestamps_ns    (T,)    int64
  images           {top_camera | left_camera | right_camera: [JPEG bytes] * T}
  samples          one entry per evaluation window, starting every `--stride` steps:
                   {frame_idx, state (14,), action_chunk (horizon, 14), images}
  meta             {episode, fps, n_frames}

  python vla/make_bundle.py --root <dataset>/insert-mouse-battery/expert-data \\
      --episode 0 --out ep0.pkl
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path

import av
import numpy as np
import pandas as pd

CAMERAS = {"observation.images.cam_high": "top_camera",
           "observation.images.cam_left_wrist": "left_camera",
           "observation.images.cam_right_wrist": "right_camera"}


def decode_jpegs(mp4: Path, quality: int) -> list[bytes]:
    frames = []
    with av.open(str(mp4)) as container:
        for frame in container.decode(video=0):
            buf = io.BytesIO()
            frame.to_image().save(buf, format="JPEG", quality=quality)
            frames.append(buf.getvalue())
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, required=True,
                   help="LeRobot dataset directory (the one holding meta/info.json)")
    p.add_argument("--episode", type=int, default=0, help="episode index (default: %(default)s)")
    p.add_argument("--out", type=Path, required=True, help="output pickle")
    p.add_argument("--stride", type=int, default=250,
                   help="steps between window starts (default: %(default)s)")
    p.add_argument("--horizon", type=int, default=50,
                   help="length of each window's action_chunk (default: %(default)s)")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality of the stored frames (default: %(default)s)")
    a = p.parse_args()

    info = json.loads((a.root / "meta" / "info.json").read_text())
    chunk = a.episode // info["chunks_size"]
    table = pd.read_parquet(a.root / info["data_path"].format(episode_chunk=chunk, episode_index=a.episode))
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(table["action"].to_numpy()).astype(np.float32)
    timestamps = (table["timestamp"].to_numpy() * 1e9).astype(np.int64)
    T = len(table)

    images = {}
    for key, name in CAMERAS.items():
        mp4 = a.root / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=a.episode)
        frames = decode_jpegs(mp4, a.jpeg_quality)
        if len(frames) < T:
            raise SystemExit(f"{mp4}: {len(frames)} frames for {T} steps")
        images[name] = frames[:T]

    samples = [{"frame_idx": k, "state": state[k].copy(),
                "action_chunk": action[k:k + a.horizon].copy(),
                "images": {name: images[name][k] for name in CAMERAS.values()}}
               for k in range(0, T - a.horizon, a.stride)]
    bundle = {"actions": action, "joint_positions": state, "timestamps_ns": timestamps,
              "images": images, "samples": samples,
              "meta": {"episode": a.episode, "fps": float(info["fps"]), "n_frames": T}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    print(f"wrote {a.out}: {T} steps, {len(samples)} windows")


if __name__ == "__main__":
    main()
