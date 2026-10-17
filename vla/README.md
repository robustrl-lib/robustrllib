# VLA evaluation (Part 4)

A single π0.5 model is fine-tuned on the expert demonstrations of the bimanual
*insert-mouse-battery* task of the RSS 2026 post-training challenge, then
evaluated by joint-space tracking against a recorded demonstration under eight
shift axes from five shift sources.

| File | What it does |
|---|---|
| `ood_eval.py` | the evaluation: the shift table `AXES`, the shift operators, the closed-loop tracking windows, the metrics and the retention per rung |
| `make_bundle.py` | builds the evaluation bundle (recorded commands, joint states and camera frames) from one dataset episode |
| `pi05_policy.py` | policy class for the policy_deployment WebSocket server that serves an openpi checkpoint |
| `dose.py` | size of each input-side shift, relative to the signal it corrupts |
| `policy_deployment_mujoco237.patch` | two changes that let policy_deployment run under MuJoCo 2.3.7 |

## Requirements

| Component | Version | Role |
|---|---|---|
| [openpi-baseline](https://github.com/posttraining-for-robotics/openpi-baseline) (the challenge's fork of openpi, Apache 2.0) | commit `3cf4747fcc2d9b8023eb43b681b141f93527d65e` | fine-tuning and inference; its environment (Python 3.11, JAX, LeRobot, MuJoCo 2.3.7, PyAV, pandas, Pillow, websockets, msgpack) runs every script in this folder |
| [policy_deployment](https://github.com/posttraining-for-robotics/policy_deployment) | commit `029eebc3cbb5c50cab19d6108489d62b59642e4d` | WebSocket policy server and client, and the dual-YAM MuJoCo scene with its helpers (`sim/check_in_sim.py`) |
| [Challenge-phase1-dataset](https://huggingface.co/datasets/Posttraining-RFM-RSS2026/Challenge-phase1-dataset) (CC BY-NC 4.0) | `insert-mouse-battery/expert-data` | 831 demonstrations, LeRobot v2.1, 60 Hz, 14-D joint state and actions, three 180×320 cameras |
| GPU | | fine-tuning and serving; the evaluator itself runs on CPU |

policy_deployment carries no licence, so no file of it is copied here: clone
it, apply the patch and point `POLICY_DEPLOYMENT` at the checkout.

```bash
git clone https://github.com/posttraining-for-robotics/openpi-baseline.git
git -C openpi-baseline checkout 3cf4747fcc2d9b8023eb43b681b141f93527d65e
git -C openpi-baseline submodule update --init --recursive
(cd openpi-baseline && GIT_LFS_SKIP_SMUDGE=1 uv sync)
export OPENPI_PY=$PWD/openpi-baseline/.venv/bin/python

git clone https://github.com/posttraining-for-robotics/policy_deployment.git
git -C policy_deployment checkout 029eebc3cbb5c50cab19d6108489d62b59642e4d
git -C policy_deployment apply <this repository>/vla/policy_deployment_mujoco237.patch
export POLICY_DEPLOYMENT=$PWD/policy_deployment

huggingface-cli download Posttraining-RFM-RSS2026/Challenge-phase1-dataset \
    --repo-type dataset --include "insert-mouse-battery/*" --local-dir data
```

The patch makes two changes to policy_deployment:

1. `sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml`: the compiler
   element becomes `<compiler autolimits="true" angle="radian"/>`. MuJoCo 2.3.7
   refuses a joint `range` without `limited` unless `autolimits` is set; with
   it, every joint range of the scene is enforced.
2. `sim/check_in_sim.py`: `_VideoSink.close()` calls `self.renderer.close()`
   only when the renderer has that method (MuJoCo 3). This only matters for
   the `--output` video mode of `check_in_sim.py`; the evaluation never renders.

## Fine-tuning

The fine-tune is the challenge's baseline recipe for this task
(`pi05_insert-mouse-battery` in openpi-baseline) on the full
`insert-mouse-battery/expert-data` split: π0.5 initialised from the
`pi05_base` checkpoint, 200,000 steps at batch size 32, arm actions as deltas
to the current joint state and grippers absolute, the dataset's task string as
the prompt. Register it in `_CONFIGS` of
`openpi-baseline/src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_imb_full",
    model=pi0_config.Pi0Config(pi05=True),
    data=DualYamDataConfig(
        repo_id="insert-mouse-battery/expert-data",
        base_config=DataConfig(
            prompt_from_task=True,
            local_files_path="<data>/insert-mouse-battery/expert-data",
        ),
        use_delta_joint_actions=True,
        adapt_to_pi=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=200_000,
    batch_size=32,
    num_workers=16,
    save_interval=10_000,
    keep_period=None,
    fsdp_devices=8,  # GPUs the model is sharded over
),
```

```bash
cd openpi-baseline
uv run scripts/compute_norm_stats.py --config-name pi05_imb_full
uv run scripts/train.py pi05_imb_full --exp-name=<exp_name> --seed=<seed> \
    --fsdp-devices=<number of GPUs> --no-wandb-enabled
```

The final checkpoint is `checkpoints/pi05_imb_full/<exp_name>/199999`.
LeRobot decodes the videos with torchcodec whenever that package imports; on
a machine where torchcodec cannot load FFmpeg's shared libraries, pass
`video_backend="pyav"` to `LeRobotDataset` in `create_torch_dataset`
(`src/openpi/training/data_loader.py`).

## Evaluation bundle

```bash
$OPENPI_PY vla/make_bundle.py --root data/insert-mouse-battery/expert-data --episode 0 --out ep0.pkl
```

Episode 0 has 2,710 steps; windows start every 250 steps, which gives 11
windows of 200 control steps each.

## Serving

```bash
PYTHONPATH=$POLICY_DEPLOYMENT:vla $OPENPI_PY $POLICY_DEPLOYMENT/scripts/launch.py \
    --policy pi05_policy:Pi05Policy \
    --policy-kwargs checkpoint_path=openpi-baseline/checkpoints/pi05_imb_full/<exp_name>/199999 \
    --host 127.0.0.1 --port 8000
```

The training config is read from the checkpoint path; pass
`--policy-kwargs config=<name>` when the checkpoint lives elsewhere.

## Evaluation

```bash
$OPENPI_PY vla/ood_eval.py --policy-deployment $POLICY_DEPLOYMENT \
    --bundle ep0.pkl --port 8000 --out results/ood_eval.json
```

For each window and each condition the scene is set to the recorded joint
state at the window start; the policy is queried with the recorded camera
frames of the current step and the simulator's live joint state; the returned
50-step chunk is executed at 60 Hz; and this repeats for four chunks. The
score is the RMSE between the simulated and the recorded 14-D joint
trajectory over the 200 steps (`track_rmse`, lower is better), averaged over
the windows. The camera frames are the recorded ones, so image shifts change
what the policy sees and the other shifts change what its actions do.

The defaults are the paper's protocol (action horizon 50, four chunks, 60 Hz,
prompt "Insert the battery into the mouse.", noise seed 10000, all eight axes
plus nominal). `--axes` restricts the run to some axes. `--replay` executes the
recorded actions instead of querying a server, which checks the scene, the
shifts and the metrics without a policy.

The output holds, per condition, `track_rmse_mean/std`, `end_l2_mean/std`
(distance to the recorded joint state at the window end) and
`chunk_rmse_mean` (predicted against recorded chunk), and, per axis and rung,
`track_rmse` (mean over the rung's settings) and
`retention = nominal track_rmse / rung track_rmse`, the quantity the paper
plots against the rung's place in its ladder.

## Shift axes

Defined once, in `AXES` of `ood_eval.py`. Each axis has its own ladder, ordered
outward from nominal; rungs index severity within an axis, not across axes.
Input-side axes have four rungs (Q1–Q4) of one setting each. Dynamics axes
have three rungs (near, far, extreme), each a group of four settings scored by
their mean `track_rmse`.

| Source | Axis (`--axes`) | Applied to | Unit | Nominal | Rungs, nearest → farthest |
|---|---|---|---|---|---|
| latency | execution delay (`latency`) | executed chunk: the previous command is held for *d* steps and the chunk runs *d* steps late | control steps at 60 Hz | 0 | 2, **5**, 10, 20 |
| action | command noise (`act_noise`) | arm joint commands, every step | Gaussian σ, rad | 0 | 0.01, **0.02**, 0.05, 0.10 |
| observation | proprioception noise (`proprio_noise`) | arm joint state sent to the policy | Gaussian σ, rad | 0 | 0.005, 0.01, 0.02, 0.05 |
| semantic | image brightness (`brightness`) | all three camera frames | × pixel intensity | 1 | 0.75, 0.5, 0.25, 0.1 |
| semantic | image blur (`blur`) | all three camera frames | Gaussian radius, px | 0 | 1, 2, 3, 4 |
| dynamics | gravity (`gravity`) | the scene's gravity | × | 1 | {0.8, 0.9, 1.1, 1.2}, {0.6, 0.7, 1.3, 1.4}, {0.25, 0.5, 1.5, 2.0} |
| dynamics | joint damping (`damping`) | every joint's damping | × | 1 | {2, 3, 4, 5}, {8, 10, 15, 20}, {50, 100, 200, 500} |
| dynamics | dry friction (`frictionloss`) | every joint's MuJoCo `frictionloss` | + | 0 | {0.25, 0.5, 0.75, 1.0}, {1.5, 2, 3, 4}, {8, 16, 32, 64} |

Bold rungs are the operating point of the real platform, read from its
demonstration logs: 83 ms of command-to-execution lag (5 steps at 60 Hz) and
0.020 rad of joint tracking error. Noise never touches the grippers.

`dose.py` measures the input-side rungs in the units of the signal they
corrupt: the relative L2 change of the camera frames for the image axes,
σ over the signal's standard deviation for the noise axes, and the RMS command
change over the delay, over the same standard deviation, for execution delay.

```bash
$OPENPI_PY vla/dose.py --bundle ep0.pkl --out results/dose.json
```
