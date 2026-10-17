# Methods

One directory per method of the paper, two shared entry points, and the helpers the
offline methods share.

```bash
python baselines/train.py -c robustrllib/configs/experiment/<method>_<task>.yaml --seed 0
python baselines/evaluate.py --run runs/<method>_<task>/seed0
```

`train.py` resolves the experiment card into `<run>/config.yaml` and runs the training script
its algorithm card names under `entry`; `evaluate.py` loads the checkpoint the run's rule
selects (`last` by default, `last:K` or `best` from the card's `checkpoint` key or
`--checkpoint`) with the class the algorithm card names under `class` and scores it on every
condition of the card's evaluation grid (another grid with `--eval robustrllib/configs/eval/<grid>.yaml`).
`--train-shift <arm>` trains under one of the training-time arms of
`robustrllib/configs/train_shifts.yaml`.

| Directory | Methods | Regime |
|---|---|---|
| `iql/` | IQL (the backbone of ATLA-IQL, RSC-IQL, ROMB, FWM, PLR-PVL and SynthER) | offline |
| `td3bc/` | TD3+BC | offline |
| `mopo/` | MOPO | offline |
| `synther/` | SynthER (diffusion-generated data, then IQL) | offline |
| `rfqi/` | RFQI | offline |
| `rorl/` | RORL | offline |
| `atla_iql/` | ATLA-IQL | offline |
| `rsc_iql/` | RSC-IQL | offline |
| `rambo/` | RAMBO | offline |
| `romb/` | ROMB | offline |
| `fwm/` | FWM | offline |
| `plr_pvl/` | PLR-PVL | offline |
| `sb3/` | PPO, SAC (Stable-Baselines3) | online |
| `dr_sac/` | DR-SAC | online |
| `rarl/` | RARL-PPO, RARL-TRPO | online |
| `rsc/` | RSC-SAC | online |
| `atla/` | ATLA (ATLA-PPO), ATLA-SA (ATLA-SA-PPO) | online |
| `offline/` | dataset loading, training-time shifts on the dataset rewards, the LunarLander dataset generator | shared |

`train_algo.py` trains any class written against the `Algo` interface of
`robustrllib.algos` (see `examples/reinforce_hopper.yaml`). The robust safe methods (RAMU,
SPiDR) run on Isaac Lab and live in `isaac/`.

The SAC experiments use the following hyperparameters. H/HC/W2D denotes Hopper,
HalfCheetah and Walker2d. Shared settings are in
[`sac.yaml`](../robustrllib/configs/algorithm/sac.yaml), with CarRacing overrides in
[`sac_carracing.yaml`](../robustrllib/configs/experiment/sac_carracing.yaml).

| Parameter | H/HC/W2D | PointMaze | FetchReach | CarRacing |
|---|---|---|---|---|
| Learning rate | 3 × 10⁻⁴ | 3 × 10⁻⁴ | 3 × 10⁻⁴ | 10⁻⁴ |
| Hidden layers (actor / critic) | 2 × 256 | 2 × 256 | 2 × 256 | NatureCNN + 2 × 256 |
| Target smoothing τ | 0.005 | 0.005 | 0.005 | 0.02 |
| Replay-buffer size | 10⁶ | 10⁶ | 10⁶ | 2 × 10⁵ |
| Learning starts | 100 | 100 | 100 | 10⁴ |
| Train frequency / gradient steps | 1 / 1 | 1 / 1 | 1 / 1 | 8 / 8 |
| Discount factor γ | 0.99 | 0.99 | 0.99 | 0.99 |
| Target entropy | automatic | automatic | automatic | automatic (−3) |
| Minibatch size | 256 | 256 | 256 | 256 |

Both actor and critics explicitly use two hidden layers of 256 units. CarRacing
uses `CnnPolicy` with its NatureCNN feature extractor before those layers.
`target_entropy: auto` sets the entropy target to minus the action dimension
(−3 for CarRacing); `ent_coef: auto` tunes the entropy coefficient during training.
