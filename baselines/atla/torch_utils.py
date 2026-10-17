"""Tensor helpers, running normalisers and the trajectory container of the ATLA trainer.

Adapted from ATLA (https://github.com/huanzhang12/ATLA_robust_RL),
``src/policy_gradients/torch_utils.py``, keeping the parts ATLA-PPO and ATLA-SA-PPO use.
"""
import numpy as np
import torch as ch


class Parameters(dict):
    """A dict whose keys are read and written as case-insensitive attributes (``p.T == p['t']``)."""

    og_getattr = dict.__getitem__
    og_setattr = dict.__setitem__

    def __getattr__(self, x):
        try:
            return self.og_getattr(x.lower())
        except KeyError:
            raise AttributeError(x)

    def __setattr__(self, x, v):
        return self.og_setattr(x.lower(), v)


def unroll(*tensors):
    """Flatten the first two dimensions (actors, time) of each tensor."""
    rets = []
    for t in tensors:
        if t is None:
            rets.append(None)
        else:
            assert len(t.shape) >= 2
            new_shape = [t.shape[0] * t.shape[1]] + list(t.shape[2:])
            rets.append(t.contiguous().view(new_shape))
    return rets


def cpu_tensorize(t):
    return ch.tensor(np.asarray(t)).float()


def shape_equal_cmp(*args):
    for i in range(len(args) - 1):
        if args[i].shape != args[i + 1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True


def determinant(mat):
    """Determinant of a diagonal matrix given by its diagonal."""
    return ch.exp(ch.log(mat).sum())


def discount_path(path, h):
    """``X_i = sum_{t >= i} h^(t-i) x_t`` along one path."""
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr * h + path[-1 - i]
        rets.append(curr)
    return ch.stack(list(reversed(rets)), 0)


def get_path_indices(not_dones):
    """``(actor, start, end + 1)`` for every path in a ``(actors, T)`` not-done mask."""
    indices = []
    num_timesteps = not_dones.shape[1]
    for actor in range(not_dones.shape[0]):
        last_index = 0
        for i in range(num_timesteps):
            if not_dones[actor, i] == 0.:
                indices.append((actor, last_index, i + 1))
                last_index = i + 1
        if last_index != num_timesteps:
            indices.append((actor, last_index, num_timesteps))
    return indices


class RunningStat(object):
    """Running mean and variance of a stream (https://github.com/joschu/modular_rl)."""

    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)


class Identity:
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class RewardFilter:
    """Divide rewards by the running std of the discounted return (OpenAI baselines' scheme)."""

    def __init__(self, prev_filter, shape, gamma, clip=None, read_only=False):
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip
        self.read_only = read_only

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        if not self.read_only:
            self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class ZFilter:
    """``y = clip((x - mean) / std)`` with running estimates of mean and std."""

    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None, read_only=False):
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter
        self.read_only = read_only

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        if not self.read_only:
            self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff / (self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()


class Trajectories:
    def __init__(self, states=None, rewards=None, returns=None, not_dones=None,
                 actions=None, action_log_probs=None, advantages=None,
                 unrolled=False, values=None, action_means=None):
        self.states = states
        self.rewards = rewards
        self.returns = returns
        self.values = values
        self.not_dones = not_dones
        self.actions = actions
        self.action_log_probs = action_log_probs
        self.advantages = advantages
        self.action_means = action_means
        self.unrolled = unrolled

    def unroll(self):
        assert not self.unrolled
        tt = unroll(self.states, self.rewards, self.returns, self.not_dones)
        tt2 = unroll(self.actions, self.action_log_probs, self.advantages, self.action_means)
        values, = unroll(self.values)
        return Trajectories(states=tt[0], rewards=tt[1], returns=tt[2], not_dones=tt[3],
                            actions=tt2[0], action_log_probs=tt2[1], advantages=tt2[2],
                            action_means=tt2[3], values=values, unrolled=True)
