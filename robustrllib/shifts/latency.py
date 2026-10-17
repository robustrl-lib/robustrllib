"""Control-loop latency: the delay between issuing a command and observing it.

Every other shift target perturbs a *quantity* (an observation, an action, a
model parameter). This one perturbs *timing*, which no amount of observation or
action noise can imitate: it changes the operator the policy is closing its loop
around. Two independent sim-to-real studies isolate it as the dominant axis.

  * Peng et al. 2018 (arXiv:1710.06537), Table III. LSTM policies trained with
    the full randomization set reach 0.89 +/- 0.06 success on a real Fetch arm.
    Freeze only the action timestep and that falls to 0.29 +/- 0.11 -- a larger
    drop than freezing link mass (0.64) or puck friction (0.48). Their model is
    a variable control period, ``dt ~ dt0 + Exp(lambda)`` with dt0 = 0.04 s and
    lambda ~ U[125, 1000] 1/s; lambda is fixed within an episode, dt is redrawn
    every step. See `substep`.
  * Tan et al. 2018 (arXiv:1804.10332), Sec. V-A. They name "lack of latency
    modeling" one of the two major causes of the reality gap, measure it on
    hardware (3 ms on the PD microcontroller, 15-19 ms on the TX2), and both
    model and randomize it: latency 0-40 ms, control step 3-20 ms, as two
    separate axes. Their implementation keeps a timestamped observation history
    and linearly interpolates at ``t - t_latency``. See `interp`.

MMDR (arXiv:2109.14549) later reuses exactly Tan's interpolation scheme for the
proprioceptive stream, which is why `interp` takes seconds rather than steps:
sub-step-resolution delay is the whole point, and rounding to whole control
steps would discard it.

Modes
-----
fixed    Action delay of a constant integer number of control steps. The
         deterministic severity axis -- use this in an eval grid, where a
         stochastic axis would put variance into every cell.
buffer   Action delay of an integer number of steps drawn from U[low, high].
         `resample: episode` (default) or `step`. Works on any env, including
         Box2D and the maze tasks, so it is the portable one.
substep  Peng's variable control period. MuJoCo only: rescales the env's
         `frame_skip` / `n_substeps` per step so one policy action is held for a
         randomized amount of simulated time.
interp   Tan's continuous observation delay: keep a timestamped history and
         return the linear interpolation at `t - t_latency`. MuJoCo not
         required, but the observation must be a float array (or a Dict of
         them) for interpolation to mean anything.

delay    Credit latency: the REWARD is paid out late while observation and
         action stay on time. `steps` control steps, `release: interval` (pay
         the accumulated sum every `steps`) or `shift` (pay each reward `steps`
         late); the accumulator is flushed on termination, so the undiscounted
         return is preserved. A frozen policy never reads the reward, so this is
         a TRAINING-time mode: train under it, evaluate on the nominal task.
         Built by the shift registry on top of RewardShift (see
         shifts/reward.py), not by the class below.

`fixed`/`buffer` delay the ACTION; `interp` delays the OBSERVATION; `substep`
stretches the control period itself; `delay` postpones the REWARD. On a real robot all three coexist and only
their sum is identifiable from behaviour (MMDR Sec. IV-C makes this argument),
so composing more than one is physically meaningful, not double-counting.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from robustrllib.shifts.base import ShiftWrapper


def _map_leaves(obs, fn):
    """Apply ``fn`` to a Box observation, or to each entry of a Dict one."""
    if isinstance(obs, dict):
        return {k: fn(v) for k, v in obs.items()}
    return fn(obs)


class LatencyShift(ShiftWrapper):
    def __init__(self, env, spec, seed=None):
        super().__init__(env, spec, seed)
        mode, p = spec.mode, spec.params
        if mode not in ("fixed", "buffer", "substep", "interp"):
            hint = (" ('delay' is built through make_robust / build_pipeline, "
                    "which route it to the reward-delay wrapper)"
                    if mode == "delay" else "")
            raise ValueError(
                f"latency shift does not support mode {mode!r}; "
                f"expected one of fixed, buffer, substep, interp, delay{hint}")
        self._mode = mode
        self._act_buf: deque | None = None
        self._obs_hist: list[tuple[float, object]] = []
        self._delay_steps = 0
        self._lam = None
        self._t_sim = 0.0

        if mode == "substep":
            # Resolve the substep knob once, loudly. Gymnasium MuJoCo envs call
            # it frame_skip; the Fetch/robotics envs call it n_substeps. An env
            # with neither cannot express a variable control period at all, and
            # silently falling back to an integer action delay would report a
            # Peng-faithful number for a different experiment.
            u = env.unwrapped
            self._skip_attr = next(
                (a for a in ("frame_skip", "n_substeps") if hasattr(u, a)), None)
            if self._skip_attr is None:
                raise NotImplementedError(
                    "latency mode 'substep' needs a MuJoCo env exposing "
                    f"frame_skip or n_substeps; {type(u).__name__} exposes "
                    "neither. Use mode 'buffer' for a portable integer delay.")
            self._skip_nominal = int(getattr(u, self._skip_attr))
            self._sim_dt = float(u.model.opt.timestep)
            self._dt0 = float(p.get("dt0", self._skip_nominal * self._sim_dt))

    # ---- delay draws -------------------------------------------------------

    def _draw_steps(self) -> int:
        p = self._spec.params
        if self._mode == "fixed":
            return int(p["steps"])
        lo, hi = int(p.get("low", 0)), int(p.get("high", 1))
        # `high` is inclusive: a config asking for [0, 2] means 0, 1 or 2 steps.
        return int(self._rng.integers(lo, hi + 1))

    def _draw_seconds(self) -> float:
        p = self._spec.params
        lo, hi = float(p.get("low", 0.0)), float(p.get("high", 0.0))
        return float(self._rng.uniform(lo, hi))

    def _draw_skip(self) -> int:
        """Peng's dt ~ dt0 + Exp(lambda), rounded to whole simulation steps."""
        dt = self._dt0 + float(self._rng.exponential(1.0 / self._lam))
        # The schedule multiplier scales only the EXCESS over the nominal period,
        # so scale=0 is exactly the unperturbed env -- the same convention the
        # dynamics shift uses for absolute randomization.
        dt = self._dt0 + (dt - self._dt0) * self._scale
        return max(1, int(round(dt / self._sim_dt)))

    # ---- gym API -----------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        p = self._spec.params
        self._t_sim = 0.0
        if self._mode in ("fixed", "buffer"):
            self._delay_steps = self._draw_steps()
            # Pre-fill with the pending action rather than zeros: a zero action
            # is a real command on a torque-controlled arm and would inject a
            # transient that has nothing to do with latency. Filled lazily on
            # the first step, once the action shape is known.
            self._act_buf = deque(maxlen=self._delay_steps + 1)
        elif self._mode == "substep":
            self._lam = float(self._rng.uniform(
                float(p.get("lambda_low", 125.0)), float(p.get("lambda_high", 1000.0))))
        elif self._mode == "interp":
            self._delay_s = self._draw_seconds() * self._scale
            self._obs_hist = [(0.0, _map_leaves(obs, lambda a: np.asarray(a, dtype=np.float64)))]
        return obs, info

    def _control_dt(self) -> float:
        u = self.env.unwrapped
        if hasattr(u, "dt"):
            return float(u.dt)
        return float(self._spec.params.get("control_dt", 1.0))

    def _interp_obs(self):
        """Linear interpolation of the observation history at t - delay."""
        target = self._t_sim - self._delay_s
        hist = self._obs_hist
        if target <= hist[0][0]:
            return hist[0][1]
        for i in range(len(hist) - 1, 0, -1):
            t0, o0 = hist[i - 1]
            t1, o1 = hist[i]
            if t0 <= target <= t1:
                w = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
                if isinstance(o0, dict):
                    return {k: o0[k] + w * (o1[k] - o0[k]) for k in o0}
                return o0 + w * (o1 - o0)
        return hist[-1][1]

    def step(self, action):
        p = self._spec.params

        if self._mode in ("fixed", "buffer"):
            if p.get("resample") == "step" and self._mode == "buffer":
                self._delay_steps = self._draw_steps()
                self._act_buf = deque(self._act_buf or [], maxlen=self._delay_steps + 1)
            action = np.asarray(action, dtype=np.float64)
            if not self._act_buf:
                self._act_buf.extend([action] * self._act_buf.maxlen)
            self._act_buf.append(action)
            applied = self._act_buf[0]
            obs, reward, term, trunc, info = self.env.step(applied)
            self._t += 1
            return obs, reward, term, trunc, info

        if self._mode == "substep":
            setattr(self.env.unwrapped, self._skip_attr, self._draw_skip())
            try:
                obs, reward, term, trunc, info = self.env.step(action)
            finally:
                # Restore the nominal period even on error. Envs cache `dt` for
                # velocity-based rewards, and a leaked frame_skip would silently
                # change the reward scale of every later episode in the process.
                setattr(self.env.unwrapped, self._skip_attr, self._skip_nominal)
            self._t += 1
            return obs, reward, term, trunc, info

        # interp
        if p.get("resample") == "step":
            self._delay_s = self._draw_seconds() * self._scale
        obs, reward, term, trunc, info = self.env.step(action)
        self._t += 1
        self._t_sim += self._control_dt()
        self._obs_hist.append(
            (self._t_sim, _map_leaves(obs, lambda a: np.asarray(a, dtype=np.float64))))
        # The history only needs to reach back past the largest possible delay.
        max_delay = float(p.get("high", 0.0))
        keep = int(max_delay / max(self._control_dt(), 1e-9)) + 2
        if len(self._obs_hist) > keep:
            del self._obs_hist[:-keep]
        return self._interp_obs(), reward, term, trunc, info
