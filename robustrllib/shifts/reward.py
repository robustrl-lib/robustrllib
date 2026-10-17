"""Reward (and cost) perturbation.

Two families live here. ADDITIVE CORRUPTION (gauss / uniform / shift) adds a
draw to the reward or to a named cost field.

DELAY is different in kind and only meaningful at TRAINING time. Corrupting or
delaying the reward of a *frozen* policy perturbs the measurement, not the
behaviour: the policy never reads the reward, so a delayed-reward evaluation
would report the same trajectory with its reward stream rearranged. Delay is
therefore a training-time channel -- it changes credit assignment, not the
task -- and the way to measure it is to train under it and evaluate normally.

The `interval` release accumulates reward and pays the sum out every `steps`
control steps, which is the aggregated-feedback setting of the delayed-reward
literature; `shift` pays each reward out `steps` steps late. Both FLUSH the
accumulator on termination, so the UNDISCOUNTED return of an episode is exactly
what it would have been without the wrapper -- the delay moves reward in time
and never destroys it. The discounted objective does change, necessarily: that
is what a delay is. Keeping the undiscounted return invariant is what makes a
delayed-reward training run comparable with its own control rather than with a
differently-scaled objective.

Delay is a TIMING shift, so its home is the `latency` target:
``ShiftSpec("latency", "delay", {"steps": 16})``; ``ShiftSpec("reward", "delay", ...)``
builds the same wrapper. The `cost` target has no delay.
"""
from __future__ import annotations

from collections import deque

from robustrllib.shifts.base import ShiftWrapper, sample_noise


class RewardShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        self._acc = 0.0
        self._rbuf: deque = deque()
        if spec.mode == "delay":
            if spec.target == "cost":
                raise ValueError(
                    "mode 'delay' postpones the reward and has no cost form; "
                    "use target 'latency' (or 'reward')")
            steps = int(spec.params.get("steps", 1))
            if steps < 1:
                raise ValueError(f"reward delay needs steps >= 1, got {steps}")
            release = spec.params.get("release", "interval")
            if release not in ("interval", "shift"):
                raise ValueError(
                    f"unknown reward-delay release {release!r}; expected "
                    "'interval' (pay the accumulated sum every `steps`) or "
                    "'shift' (pay each reward `steps` late)")

    def reset(self, **kwargs):
        self._acc = 0.0
        self._rbuf.clear()
        return super().reset(**kwargs)

    def _delay(self, reward, done):
        p = self._spec.params
        k = int(p.get("steps", 1))
        if p.get("release", "interval") == "interval":
            self._acc += reward
            if done or self._t % k == 0:
                out, self._acc = self._acc, 0.0
                return out
            return 0.0
        self._rbuf.append(reward)
        out = self._rbuf.popleft() if len(self._rbuf) > k else 0.0
        if done:                      # flush, so the episode return is preserved
            out += sum(self._rbuf)
            self._rbuf.clear()
        return out

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        if self._spec.mode == "delay":
            return obs, self._delay(reward, term or trunc), term, trunc, info
        noise = float(sample_noise(self._rng, self._spec.mode, self._spec.params, (), self._scale))
        key = self._spec.params.get("info_key", "cost") if self._spec.target == "cost" else None
        if key is not None:
            info[key] = info.get(key, 0.0) + noise
        else:
            reward = reward + noise
        return obs, reward, term, trunc, info
