"""Benchmark metrics.

Per-condition aggregates plus cross-condition robustness summaries.
"""
from __future__ import annotations

import numpy as np

# numpy>=2 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def cvar(returns, alpha=0.1):
    """Mean of the worst ``alpha`` fraction of episodes (lower-tail CVaR)."""
    r = np.sort(np.asarray(returns, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * len(r))))
    return float(r[:k].mean())


def normalize(value, random_score, expert_score):
    denom = expert_score - random_score
    return float((value - random_score) / denom) if denom else float("nan")


def condition_metrics(returns, costs=None, successes=None, norm=None, cvar_alpha=0.1):
    """Aggregate one eval condition's per-episode results."""
    returns = np.asarray(returns, dtype=np.float64)
    out = {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "cvar": cvar(returns, cvar_alpha),
        "n_episodes": int(len(returns)),
    }
    if norm is not None:
        out["normalized_return"] = normalize(out["return_mean"], norm["random"], norm["expert"])
    if costs is not None and len(costs):
        costs = np.asarray(costs, dtype=np.float64)
        out["cost_mean"] = float(costs.mean())
        out["constraint_violation"] = float((costs > 0).mean())
    if successes is not None and len(successes):
        out["success_rate"] = float(np.mean(successes))
    return out


def summary_metrics(per_condition: dict, severity_key=None):
    """Cross-condition robustness: worst-case, mean, and AUC over severity."""
    means = [m["return_mean"] for m in per_condition.values()]
    summary = {
        "return_mean_avg": float(np.mean(means)),
        "worst_case_return": float(np.min(means)),
    }
    if any("normalized_return" in m for m in per_condition.values()):
        norms = [m.get("normalized_return", np.nan) for m in per_condition.values()]
        summary["normalized_return_avg"] = float(np.nanmean(norms))
        summary["worst_case_normalized"] = float(np.nanmin(norms))
    # AUC over an ordered severity axis, if provided as {condition: severity}.
    if severity_key:
        pts = sorted((severity_key[c], per_condition[c]["return_mean"])
                     for c in per_condition if c in severity_key)
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            summary["auc"] = float(_trapz(ys, xs) / (xs[-1] - xs[0]))
    return summary
