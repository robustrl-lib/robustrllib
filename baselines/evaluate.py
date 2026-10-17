#!/usr/bin/env python
"""Evaluate a trained policy on the shift grid of its experiment card.

    python baselines/evaluate.py --run runs/iql_hopper/seed0
    python baselines/evaluate.py --run runs/iql_hopper/seed0 --eval robustrllib/configs/eval/part2_hopper.yaml

Protocol: a checkpoint rule fixed before evaluation, a fixed evaluation seed, and
``episodes`` episodes per condition; episode ``i`` resets with ``seed + i``, so every
method meets the same initial states. Each condition is one fresh environment: the task
card's environment with the condition's shifts stacked on it.

The checkpoint rule is ``last`` (the final checkpoint, the default), ``last:K`` (the
last K checkpoints, whose episodes are pooled) or ``best`` (the run's own ``best``
checkpoint, written by methods that keep a selection rule of their own). The algorithm
card's ``checkpoint`` key sets it per method; ``--checkpoint`` overrides it per run, and
``--ckpt`` names one file explicitly.

Writes ``<run>/eval/<grid>.json``: per condition the mean, spread and lower tail of the
return, the normalized score ``100 (R - R_min) / (R_max - R_min)`` with the task card's
anchors, and the success rate where the task reports one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.common import REPO, load_run_config, read_yaml  # noqa: E402


def find_checkpoints(run: Path, which: str = "last") -> list[Path]:
    """The checkpoints a rule selects in ``<run>/ckpt/``.

    ``last``: the highest-numbered file. ``last:K``: the K highest-numbered files, oldest
    first. ``best``: the file whose name starts with ``best``. Numbers are the step or
    update count in the file name.
    """
    ckpt = run / "ckpt"
    files = sorted(p for p in ckpt.iterdir() if p.is_file()) if ckpt.is_dir() else []
    if which == "best":
        best = [p for p in files if p.stem.startswith("best")]
        if best:
            return best[:1]
        raise FileNotFoundError(f"no best checkpoint in {ckpt}")
    m = re.fullmatch(r"last(?::(\d+))?", which)
    if not m:
        raise ValueError(f"unknown checkpoint rule {which!r}; use last, last:K or best")
    k = int(m.group(1) or 1)
    if k < 1:
        raise ValueError("last:K needs K >= 1")
    numbered = sorted((int(n.group(1)), p) for p in files if (n := re.search(r"(\d+)", p.stem)))
    if numbered:
        if len(numbered) < k:
            raise FileNotFoundError(f"{ckpt} holds {len(numbered)} numbered checkpoints, fewer than {k}")
        return [p for _, p in numbered[-k:]]
    if len(files) == 1 and k == 1:
        return files
    raise FileNotFoundError(f"no checkpoint in {ckpt}")


def find_checkpoint(run: Path, which: str = "last") -> Path:
    """The single checkpoint of a rule that selects one (``last`` or ``best``)."""
    return find_checkpoints(run, which)[-1]


def run_condition(policies, task, cond, episodes, seed, cvar_alpha=0.1):
    """Score one condition. ``policies`` holds one policy per selected checkpoint; every policy
    plays the same ``episodes`` seeds, and the episodes of all of them are pooled."""
    from robustrllib.metrics import condition_metrics
    from robustrllib.shifts import bind_actor
    from robustrllib.tasks import make_env

    task = dict(task)
    if cond.get("env_id"):
        task["env_id"] = cond["env_id"]
    env = make_env(task, shifts=cond.get("shifts") or [], seed=seed, env_kwargs=cond.get("env_kwargs"))
    adversarial = any(s.get("target") == "observation" and s.get("mode") == "adversarial"
                      for s in cond.get("shifts") or [])
    if adversarial and any(getattr(p, "actor", None) is None for p in policies):
        env.close()
        return {"skipped": "the policy has no differentiable actor for the adversarial observation shift"}

    returns, successes, lengths = [], [], []
    for policy in policies:
        if adversarial:
            bind_actor(env, policy.actor)
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            if hasattr(policy, "reset"):
                policy.reset()
            total, steps, success, done = 0.0, 0, None, False
            while not done:
                action = policy.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
                flag = info.get("success", info.get("is_success"))
                if flag is not None:
                    success = bool(success) or bool(flag)
                done = terminated or truncated
            returns.append(total)
            lengths.append(steps)
            if success is not None:
                successes.append(float(success))
    env.close()

    out = condition_metrics(returns, successes=successes or None, cvar_alpha=cvar_alpha)
    score = task.get("score")
    if score:
        out["score"] = 100.0 * (out["return_mean"] - score["min"]) / (score["max"] - score["min"])
    out["episode_length_mean"] = float(np.mean(lengths))
    out["returns"] = [float(r) for r in returns]
    for key in ("axis", "quartile", "severity", "channel"):
        if key in cond:
            out[key] = cond[key]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory written by baselines/train.py")
    p.add_argument("-c", "--config", default=None, help="run configuration (default <run>/config.yaml)")
    p.add_argument("--eval", default=None, help="evaluation card (default: the card's eval section)")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint rule: last (default), last:K or best; overrides the algorithm card")
    p.add_argument("--ckpt", default=None, help="one explicit checkpoint file, for diagnostics")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import torch

    from robustrllib.algos import load_algo
    from robustrllib.tasks import make_env

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO / run
    cfg = load_run_config(args.config or run / "config.yaml")
    grid_card = read_yaml(REPO / args.eval if args.eval and not Path(args.eval).is_absolute() else args.eval) \
        if args.eval else cfg["eval"]
    grid_name = Path(args.eval).stem if args.eval else grid_card.get("name", "eval")
    episodes = int(args.episodes or grid_card.get("episodes", 20))
    seed = int(grid_card.get("seed", 10000))
    algo, task = cfg["algorithm"], cfg["task"]

    rule = "file" if args.ckpt else (args.checkpoint or algo.get("checkpoint") or "last")
    ckpts = [Path(args.ckpt)] if args.ckpt else find_checkpoints(run, rule)
    torch.manual_seed(seed)
    policies = [load_algo(algo["name"], str(c), cls=algo.get("class"), env=make_env(task, seed=seed),
                          device=args.device, config=cfg) for c in ckpts]
    shown = ", ".join(str(c.relative_to(run) if c.is_relative_to(run) else c) for c in ckpts)
    print(f"[eval] {cfg['name']}  checkpoint rule {rule}: {shown}  "
          f"{len(grid_card['grid'])} conditions x {episodes} episodes, seed {seed}", flush=True)

    results = {}
    for cond in grid_card["grid"]:
        results[cond["name"]] = r = run_condition(policies, task, cond, episodes, seed)
        if "skipped" in r:
            print(f"  {cond['name']:<22} skipped: {r['skipped']}", flush=True)
        else:
            score = f"  score {r['score']:7.1f}" if "score" in r else ""
            print(f"  {cond['name']:<22} return {r['return_mean']:10.2f} +- {r['return_std']:8.2f}{score}", flush=True)

    out = run / "eval" / f"{grid_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": cfg["name"], "checkpoint_rule": rule,
                               "checkpoint": str(ckpts[-1]) if len(ckpts) == 1 else [str(c) for c in ckpts],
                               "seed": seed, "episodes": episodes, "conditions": results}, indent=1))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
