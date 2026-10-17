"""pi0.5 policy for the policy_deployment WebSocket server, backed by an openpi checkpoint.

Serve a fine-tuned checkpoint from the openpi environment:

    PYTHONPATH=$POLICY_DEPLOYMENT:vla python $POLICY_DEPLOYMENT/scripts/launch.py \\
        --policy pi05_policy:Pi05Policy \\
        --policy-kwargs checkpoint_path=<checkpoint_base_dir>/pi05_imb_full/<exp_name>/199999 \\
        --host 127.0.0.1 --port 8000

`checkpoint_path` is the openpi step directory (the one holding params/ and
assets/). The openpi training config is the directory two levels above the step
(<checkpoint_base_dir>/<config>/<exp_name>/<step>) unless `config=<name>` is passed.

A request carries three CHW uint8 camera images (cam_high, cam_left_wrist,
cam_right_wrist), the 14-D joint state [left arm 1-6, left gripper, right arm 1-6,
right gripper] and the prompt. openpi's YamInputs takes exactly these camera
names and converts CHW images itself; YamOutputs returns absolute 14-D joint
actions with grippers in the dataset's normalised [0, 1] range, which is the
convention of the dual-YAM scene. Both are passed through unchanged.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import numpy as np

from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class Pi05Policy(BasePolicy):
    def __init__(self, checkpoint_path: str, config: str | None = None, default_prompt: str = "",
                 action_horizon: int = 50, action_dim: int = 14, state_dim: int = 14) -> None:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        self.checkpoint_path = pathlib.Path(checkpoint_path).resolve()
        if not (self.checkpoint_path / "params").is_dir():
            raise FileNotFoundError(f"{self.checkpoint_path} has no params/; pass the step directory")
        self.config_name = config or self.checkpoint_path.parent.parent.name
        self.default_prompt = default_prompt
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.model = policy_config.create_trained_policy(
            openpi_config.get_config(self.config_name), self.checkpoint_path,
            default_prompt=default_prompt)

    @property
    def metadata(self) -> ServerMetadata:
        return {
            "protocol_version": "1.0",
            "policy_name": f"openpi:{self.config_name}",
            "control_mode": "joints",
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "image_keys": list(CAMERAS),
            "image_shape": [3, 224, 224],
            "expects_prompt": True,
        }

    def infer(self, obs: Dict[str, Any]) -> InferenceResponse:
        images = obs.get("images") or {}
        missing = [c for c in CAMERAS if c not in images]
        if missing:
            # YamInputs would replace a missing camera by a masked black image.
            raise ValueError(f"request is missing cameras {missing}")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(f"state has {state.size} dims, expected {self.state_dim}")
        out = self.model.infer({
            "images": {c: np.asarray(images[c], dtype=np.uint8) for c in CAMERAS},
            "state": state,
            "prompt": obs.get("prompt") or self.default_prompt,
        })
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise RuntimeError(f"policy returned actions of shape {actions.shape}")
        return {"actions": actions[:self.action_horizon]}
