"""Observation-space perturbation.

Handles both Box observations (perturb the whole array) and Dict observations
(goal-conditioned / manipulation envs). For Dict obs, by default only the
``observation`` entry is perturbed -- the goal entries (``desired_goal`` /
``achieved_goal``) describe the task and are left intact -- but the keys to
perturb can be overridden via ``params["keys"]``.

ADVERSARIAL MODE. ``ShiftSpec("observation", "adversarial", {"eps": 0.05})``
replaces the random draw by a worst-case search: at every step the wrapper
looks inside the L-inf ball of radius `eps` around the true observation for the
point that most changes the policy's deterministic action, maximizing
``||pi(o + d) - pi(o)||^2`` by `steps` iterations of projected gradient ascent
(step size ``2.5 * eps / steps``, random start). It needs only actor gradients,
so every method is attacked with identical machinery, and it needs no training,
so the condition is reproducible. This is the training-free stand-in for the
SA-MDP worst case that a learned observation adversary approximates.

The attack differentiates through the policy, so the wrapper must be given the
policy's actor: a callable from a float32 tensor of shape ``[1, obs_dim]`` to
the action tensor, with the graph intact. Pass it as ``params["actor"]`` or
attach it afterwards with :func:`bind_actor`, which is how a specification
loaded from YAML gets one. Any observation normalization the policy applies
belongs inside the callable, so that `eps` is measured in raw observation
units for every method.
"""
from __future__ import annotations

import numpy as np

from robustrllib.shifts.base import ShiftWrapper, sample_noise


def bind_actor(env, actor):
    """Attach `actor` to every adversarial observation shift in ``env``'s stack.

    Returns the number of wrappers it was attached to, so a caller can tell a
    stack that has no adversarial observation shift from one that does.
    """
    n, cur = 0, env
    while cur is not None:
        if isinstance(cur, ObservationShift) and cur._spec.mode == "adversarial":
            cur._actor = actor
            n += 1
        cur = getattr(cur, "env", None)
    return n


class ObservationShift(ShiftWrapper):
    # Modes with per-wrapper state, handled here rather than in the stateless
    # `sample_noise`. Keeping them out of that helper is deliberate: it is also
    # used by the reward shift and by the multi-agent wrappers, neither of which
    # carries the state these need, so a config that asks for them there fails
    # loudly instead of silently degrading to something else.
    _STATEFUL = ("relative", "bias")

    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        self._stateful = spec.mode in self._STATEFUL
        # Welford accumulators for `relative`, persisted ACROSS episodes: a
        # per-episode reset would leave sigma at ~0 for the opening steps of
        # every episode, which is exactly where a control policy is most
        # sensitive to sensor noise.
        # Keyed by observation key (None for a Box obs): a Dict observation has
        # several entries with different shapes AND different physical units, so
        # one shared accumulator would be both shape-invalid and meaningless.
        self._stats: dict = {}
        self._bias: dict = {}
        self._adversarial = spec.mode == "adversarial"
        self._actor = spec.params.get("actor") if self._adversarial else None
        if self._adversarial and int(spec.params.get("steps", 10)) < 1:
            raise ValueError("adversarial observation shift needs steps >= 1")

    def _attack(self, arr):
        """K-step PGD in the L-inf eps-ball, maximizing the action change."""
        if self._actor is None:
            raise ValueError(
                "adversarial observation shift has no actor to differentiate "
                "through; pass params['actor'] or call bind_actor(env, actor) "
                "with a callable mapping a [1, obs_dim] tensor to the action")
        import torch  # local: the rest of the framework does not need torch

        p = self._spec.params
        eps = float(p.get("eps", 0.05)) * self._scale
        k = int(p.get("steps", 10))
        if eps <= 0.0:
            return arr
        o = torch.as_tensor(arr, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            a0 = self._actor(o)
        # Random start drawn from the wrapper's own stream, so the attack is a
        # function of the seed like every other shift.
        d = torch.as_tensor(self._rng.uniform(-eps, eps, size=tuple(o.shape)),
                            dtype=torch.float32)
        step = 2.5 * eps / k
        for _ in range(k):
            d = d.detach().requires_grad_(True)
            loss = ((self._actor(o + d) - a0) ** 2).sum()
            loss.backward()
            with torch.no_grad():
                d = torch.clamp(d + step * d.grad.sign(), -eps, eps)
        out = (o + d.detach()).numpy().astype(np.float64).reshape(np.shape(arr))
        # The search runs in float32; project once more in float64 so the
        # result is inside the ball exactly, not up to single-precision rounding.
        return np.clip(out, arr - eps, arr + eps)

    def _running_std(self, key, arr):
        st = self._stats.get(key)
        if st is None:
            st = self._stats[key] = {"n": 0, "mean": np.zeros_like(arr),
                                     "m2": np.zeros_like(arr)}
        st["n"] += 1
        delta = arr - st["mean"]
        st["mean"] = st["mean"] + delta / st["n"]
        st["m2"] = st["m2"] + delta * (arr - st["mean"])
        if st["n"] < 2:
            return np.zeros_like(arr)
        return np.sqrt(st["m2"] / (st["n"] - 1))

    def _noise(self, arr, key=None):
        arr = np.asarray(arr, dtype=np.float64)
        if self._adversarial:
            return self._attack(arr)
        if not self._stateful:
            return arr + sample_noise(self._rng, self._spec.mode, self._spec.params,
                                      arr.shape, self._scale)
        p = self._spec.params
        if self._spec.mode == "relative":
            # sigma_i = frac * scale_i, one sigma per feature. Peng et al. 2018
            # use 5% of each feature's running standard deviation; an absolute
            # sigma cannot express this, because a 52-D arm state mixes joint
            # angles (order 1 rad) with velocities (order 10 rad/s) and one
            # sigma is either negligible for the latter or overwhelming for the
            # former. `ref_std` pins the scale to a measured constant instead,
            # which is what an eval grid should use if it wants cells that do
            # not depend on how many episodes ran before them.
            ref = p.get("ref_std")
            std = (np.asarray(ref, dtype=np.float64) if ref is not None
                   else self._running_std(key, arr))
            sigma = float(p.get("frac", 0.05)) * std * self._scale
            return arr + self._rng.normal(0.0, 1.0, size=arr.shape) * sigma
        # bias: one constant offset per episode -- a sensor zero-offset, not
        # noise. Tan et al. 2018 randomize IMU bias (+/-0.05 rad) as an axis
        # separate from IMU noise for exactly this reason: a policy can average
        # out zero-mean per-step noise but not a constant it cannot observe.
        if key not in self._bias:
            sigma = float(p.get("sigma", 0.0))
            if sigma > 0.0:
                self._bias[key] = self._rng.normal(0.0, sigma, size=arr.shape)
            else:
                lo, hi = float(p.get("low", -0.05)), float(p.get("high", 0.05))
                self._bias[key] = self._rng.uniform(lo, hi, size=arr.shape)
        return arr + self._bias[key] * self._scale

    def _perturb(self, obs):
        if isinstance(obs, dict):
            if self._adversarial:
                raise NotImplementedError(
                    "adversarial observation shift is defined for Box "
                    "observations; flatten the Dict observation first (the task "
                    "configs do this with flatten_keys) so the actor and the "
                    "attack see the same vector")
            keys = self._spec.params.get("keys")
            if keys is None:
                keys = ["observation"] if "observation" in obs else list(obs)
            return {k: (self._noise(v, k) if k in keys else v) for k, v in obs.items()}
        return self._noise(obs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # A sensor offset is drawn per power-cycle, i.e. per episode; the
        # running-std accumulators deliberately survive (see __init__).
        self._bias.clear()
        return self._perturb(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        return self._perturb(obs), reward, term, trunc, info
