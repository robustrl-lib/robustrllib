#!/usr/bin/env python
"""Create the LunarLanderContinuous-v3 ``medium`` dataset.

    python baselines/offline/lunarlander.py --out datasets/lunarlander_medium.npz

SAC trains online for ``--train-steps`` steps. Every ``--eval-every`` steps the
deterministic policy is scored on ``--screen-episodes`` episodes; snapshots whose mean
return lies in [``--return-min``, ``--return-max``] with a safe-landing rate of at least
``--safe-rate`` are kept, closest to ``--return-target`` first. The first of them that
passes the same test on ``--confirm-episodes`` new episodes collects ``--steps``
transitions deterministically (episode ``i`` reset with seed ``collect seed + i``); the
collected episodes must pass the test as well. Screening, confirmation and collection use
the seeds ``seed + 10000``, ``seed + 30000`` and ``seed + 50000``.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from baselines.mopo.sac import SACAgent
from baselines.offline.common import make_env

TASK = {"env_id": "LunarLanderContinuous-v3"}


def rollouts(agent, env, episodes, seed):
    """Mean return and safe-landing rate of the deterministic policy."""
    returns, safe = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done, info = 0.0, False, {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
        safe.append(bool(info.get("safe_landing", False)))
    return float(np.mean(returns)), float(np.mean(safe))


def collect(agent, env, steps, seed):
    keys = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts")
    data = {k: [] for k in keys}
    returns, safe, total, episode = [], [], 0.0, 0
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        for key, value in zip(keys, (obs, action, reward, next_obs, terminated, truncated)):
            data[key].append(np.asarray(value, dtype=np.float32))
        total += float(reward)
        if terminated or truncated:
            returns.append(total)
            safe.append(bool(info.get("safe_landing", False)))
            total, episode = 0.0, episode + 1
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs = next_obs
    return {k: np.stack(v) for k, v in data.items()}, float(np.mean(returns)), float(np.mean(safe))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="datasets/lunarlander_medium.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=1_500_000)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--screen-episodes", type=int, default=20)
    p.add_argument("--confirm-episodes", type=int, default=100)
    p.add_argument("--shortlist", type=int, default=12)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--return-min", type=float, default=100.0)
    p.add_argument("--return-max", type=float, default=200.0)
    p.add_argument("--return-target", type=float, default=150.0)
    p.add_argument("--safe-rate", type=float, default=0.4)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, eval_env = make_env(TASK, a.seed), make_env(TASK, a.seed)
    env.action_space.seed(a.seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, float(env.action_space.high[0]), target_entropy=-act_dim, device=device)
    passes = lambda ret, safe: a.return_min <= ret <= a.return_max and safe >= a.safe_rate

    size = min(a.train_steps, 1_000_000)
    buffer = [np.zeros((size, n), dtype=np.float32) for n in (obs_dim, act_dim, 1, obs_dim, 1)]
    shortlist, obs = [], env.reset(seed=a.seed)[0]
    for step in range(1, a.train_steps + 1):
        if step <= a.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
                action = agent.policy.sample(state)[0].cpu().numpy()[0]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        for array, value in zip(buffer, (obs, action, reward, next_obs, float(terminated))):
            array[(step - 1) % size] = value
        obs = env.reset()[0] if terminated or truncated else next_obs
        if step > a.warmup_steps:
            idx = rng.integers(0, min(step, size), size=256)
            agent.update(*(torch.as_tensor(array[idx], device=device) for array in buffer))
        if step % a.eval_every == 0:
            ret, safe = rollouts(agent, eval_env, a.screen_episodes, a.seed + 10_000)
            print(f"[screen] step {step} return {ret:.1f} safe {safe:.2f}", flush=True)
            if passes(ret, safe):
                shortlist.append((abs(ret - a.return_target), -safe, step, copy.deepcopy(agent.policy.state_dict())))
                shortlist = sorted(shortlist, key=lambda c: c[:3])[:a.shortlist]

    for _, _, step, weights in shortlist:
        agent.policy.load_state_dict(weights)
        ret, safe = rollouts(agent, eval_env, a.confirm_episodes, a.seed + 30_000)
        print(f"[confirm] step {step} return {ret:.1f} safe {safe:.2f}", flush=True)
        if passes(ret, safe):
            break
    else:
        raise SystemExit("no snapshot passed the confirmation test")
    data, ret, safe = collect(agent, make_env(TASK, a.seed), a.steps, a.seed + 50_000)
    print(f"[collect] {a.steps} transitions, return {ret:.1f}, safe {safe:.2f}", flush=True)
    if not passes(ret, safe):
        raise SystemExit("the collected episodes do not pass the test")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
