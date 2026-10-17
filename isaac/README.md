# Part 4: robust safe methods on Isaac Lab

Unitree G1 velocity tracking (`Isaac-Velocity-Flat-G1-v0`, evaluated on flat and rough ground) and
Franka drawer opening (`Isaac-Open-Drawer-Franka-v0`), each trained with PPO, DR-PPO, ATLA-PPO,
RAMU-PPO and SPiDR-PPO and evaluated under the shift grid of the task.

| Path | Content |
|---|---|
| `train.py` | one trainer: `--task {g1,drawer} --method {ppo,dr,atla,ramu,spidr} --seed S` |
| `evaluate.py` | one evaluator for the checkpoints of every method: `--task --checkpoint --condition {name,all}` |
| `tasks/g1.py` | G1 training configurations, OmniH2O evaluation conditions, rough test terrain, joint-limit cost |
| `tasks/drawer.py` | Drawer training randomisation, the 16 evaluation conditions, arm joint-limit cost, evaluation rollout |
| `tasks/assets.py` | local copies of the robot and cabinet USD files (`assets.py {drawer,g1} DIR`) |
| `methods/spidr.py` | SPiDR-PPO: primal-dual PPO solver and the one-step dynamics ensemble |
| `methods/ramu.py` | RAMU-PPO: Wang-distorted value targets over sampled next observations |
| `methods/atla.py` | ATLA-PPO: alternating protagonist / observation-adversary runner |
| `configs/omnih2o.yaml`, `configs/g1_conditions.tsv` | G1 condition definitions and the 31 evaluated terrain/condition pairs |

## Requirements

Isaac Lab 3.0.0 on Isaac Sim 6.0.1 (Python 3.12, PhysX) with `rsl-rl-lib==5.0.1` (torch 2.10, tensordict 0.13).
The scripts are run with Isaac Lab's Python from the Isaac Lab directory, e.g.
`./isaaclab.sh -p <repo>/isaac/train.py ...`; they add the repository root to the import path themselves.

The Drawer task loads the Franka and cabinet files of the Isaac 5.1 asset tree from a local copy:

```bash
./isaaclab.sh -p <repo>/isaac/tasks/assets.py drawer <assets>/drawer
./isaaclab.sh -p <repo>/isaac/tasks/assets.py g1 <assets>/g1        # optional for G1
```

## Training

```bash
./isaaclab.sh -p <repo>/isaac/train.py --task g1     --method spidr --seed 0 --headless
./isaaclab.sh -p <repo>/isaac/train.py --task drawer --method ramu  --seed 0 --assets <assets>/drawer --headless
```

Checkpoints go to `--out` (default `runs/isaac/<task>_<method>/seed<S>`) as `model_<iteration>.pt` with
`params/agent.yaml`. Default budgets (iterations x environments x steps per environment), as in the paper:

| Task | PPO, DR-PPO, RAMU-PPO | ATLA-PPO (two rollouts per iteration) | SPiDR-PPO |
|---|---|---|---|
| G1 | 1500 x 4096 x 24 | 750 x 4096 x 24 | 900 x 512 x 320 (+ 8 ensemble members per environment) |
| Drawer | 3200 x 512 x 96 | 1600 x 512 x 96 | 3200 x 512 x 96 (+ 8 ensemble members per environment) |

Training environments: on G1, PPO, ATLA-PPO and RAMU-PPO train without randomisation, DR-PPO with the
task's own (observation noise, velocity pushes) and SPiDR-PPO inside the OmniH2O envelope (friction,
link mass, base CoM, actuator gains, torque noise, 1-3 step action delay, 1 m/s pushes every 5 s).
On the Drawer, PPO and RAMU-PPO train without randomisation and the other three with the Drawer
randomisation of `tasks/drawer.py`.

Learners: G1 PPO, DR-PPO, ATLA-PPO and RAMU-PPO use the task's PPO settings (256-128-128 ELU, adaptive
learning rate 1e-3); SPiDR-PPO uses the reference implementation's settings (512-256-128 swish with
observation normalisation, fixed learning rate 3e-4, discount 0.97, 4 epochs x 32 minibatches, clip 0.3,
entropy 0.01). Every Drawer method uses the primal-dual solver of `methods/spidr.py` (the cost terms are
active for SPiDR-PPO only) with the task's PPO network (256-128-64 ELU) and a fixed learning rate of
5e-4; RAMU-PPO uses its own solver with a fixed learning rate of 1e-4 on the Drawer.
SPiDR-PPO and RAMU-PPO use an episode cost budget of 20, cost discount 0.9 and multiplier step 0.01
from 0.01. ATLA perturbations are bounded by 0.05 (per-dimension scales on the Drawer).

## Evaluation

```bash
./isaaclab.sh -p <repo>/isaac/evaluate.py --task g1 --checkpoint runs/isaac/g1_spidr/seed0/model_899.pt \
    --condition all --headless
```

128 environments, a fixed evaluation seed, deterministic actions, the first episode of every
environment, one simulator process per condition. Each condition writes `<out>/<condition>.json` (default
`<checkpoint dir>/eval`) with the per-episode records and: G1 `mean_return`, `survival_rate` (episodes
reaching the 20 s limit); Drawer `success_rate` (opening > 0.30 m at the end of the episode),
`within_budget_rate` (joint-limit cost within 20 per episode) and `mean_return`.

## Tests

```bash
./isaaclab.sh -p -m pytest -q <repo>/isaac/tests        # CPU only
```
