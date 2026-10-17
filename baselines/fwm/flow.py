"""Flow-matching world model of FWM and PLR-PVL.

The model generates a ``T``-step window of (state, reward) from the first state-action
pair and the window's discounted return-to-go. States are
normalised per dimension through the empirical CDF, rewards and return-to-go by min-max,
all to [-1, 1]. Sampling integrates the learned velocity field with classifier-free
guidance (conditional and unconditional velocities mixed by ``guidance_weight``).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FLOW_DEFAULTS = {"n_feat": 128, "lr": 1e-4, "batch_size": 256, "drop_prob": 0.2, "epochs": 5000,
                 "batches_per_epoch": 40}


# ------------------------------------------------------------------ normaliser
class FlowNormalizer:
    def __init__(self, probs, quantiles, reward_min, reward_max, rtg_min, rtg_max):
        self.probs, self.quantiles = np.asarray(probs, np.float32), np.asarray(quantiles, np.float32)
        self.reward_min, self.reward_max = float(reward_min), float(reward_max)
        self.rtg_min, self.rtg_max = float(rtg_min), float(rtg_max)
        self._tables = []
        for d in range(self.quantiles.shape[1]):
            q, i = np.unique(self.quantiles[:, d].astype(np.float64), return_index=True)
            self._tables.append((q, self.probs.astype(np.float64)[i]))

    @classmethod
    def fit(cls, episodes, window, gamma):
        obs = np.concatenate([ep.observations for ep in episodes]).astype(np.float64)
        rewards = np.concatenate([ep.rewards for ep in episodes])
        probs = np.linspace(0.0, 1.0, 10001)
        rtg = np.concatenate([window_returns(ep.rewards, window, gamma, complete=True) for ep in episodes])
        return cls(probs, np.quantile(obs, probs, axis=0), rewards.min(), rewards.max(), rtg.min(), rtg.max())

    @property
    def state_dim(self):
        return self.quantiles.shape[1]

    def states(self, x):
        x = np.asarray(x, np.float32)
        flat, out = x.reshape(-1, x.shape[-1]), np.empty(x.reshape(-1, x.shape[-1]).shape, np.float32)
        for d, (q, p) in enumerate(self._tables):
            out[:, d] = 0.0 if q.size <= 1 else 2.0 * np.interp(flat[:, d], q, p, left=p[0], right=p[-1]) - 1.0
        return out.reshape(x.shape)

    def unstates(self, x):
        x = np.asarray(x, np.float32)
        flat = x.reshape(-1, x.shape[-1])
        cdf = np.clip((flat + 1.0) * 0.5, 0.0, 1.0).astype(np.float64)
        out = np.empty(flat.shape, np.float32)
        for d, (q, p) in enumerate(self._tables):
            if q.size <= 1:
                out[:, d] = q[0] if q.size else 0.0
            else:
                out[:, d] = np.interp(cdf[:, d], p, q, left=q[0], right=q[-1])
        return out.reshape(x.shape)

    def rewards(self, r):
        return (2.0 * (np.asarray(r, np.float32) - self.reward_min) / max(self.reward_max - self.reward_min, 1e-6) - 1.0).astype(np.float32)

    def unrewards(self, r):
        r = np.clip(np.asarray(r, np.float32), -1.0, 1.0)
        return (0.5 * (r + 1.0) * (self.reward_max - self.reward_min) + self.reward_min).astype(np.float32)

    def rtg(self, g):
        return (2.0 * (np.asarray(g, np.float32) - self.rtg_min) / max(self.rtg_max - self.rtg_min, 1e-6) - 1.0).astype(np.float32)

    def state_dict(self):
        return {"probs": self.probs, "quantiles": self.quantiles, "reward_min": self.reward_min,
                "reward_max": self.reward_max, "rtg_min": self.rtg_min, "rtg_max": self.rtg_max}

    @classmethod
    def from_state_dict(cls, s):
        return cls(s["probs"], s["quantiles"], s["reward_min"], s["reward_max"], s["rtg_min"], s["rtg_max"])


def window_returns(rewards, window, gamma, complete=False):
    """Discounted ``window``-step return from every step (from complete windows only if ``complete``)."""
    discounts = gamma ** np.arange(window)
    n = len(rewards)
    starts = range(n - window + 1) if complete else range(n)
    return np.asarray([float((rewards[t:t + window] * discounts[:min(window, n - t)]).sum()) for t in starts],
                      dtype=np.float32)


# ------------------------------------------------------------------- network
class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, residual=False):
        super().__init__()
        self.residual, self.same = residual, cin == cout
        self.conv1 = nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.GELU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if not self.residual:
            return x2
        return ((x if self.same else x1) + x2) / 1.414


class _Embed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)

    def forward(self, x, size):
        x = self.up(x)
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UNet(nn.Module):
    """Velocity field over a [1, T, state_dim + 1] window, conditioned on time, the first
    (state, action) and the return-to-go; ``context_mask = 1`` drops the conditioning."""

    def __init__(self, n_feat=128):
        super().__init__()
        f = n_feat
        self.n_feat = f
        self.init = _ConvBlock(1, f, residual=True)
        self.down1 = nn.Sequential(_ConvBlock(f, f), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(_ConvBlock(f, 2 * f), nn.MaxPool2d(2))
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.GELU())
        self.time1, self.time2 = _Embed(2 * f), _Embed(f)
        self.cond1, self.cond2 = _Embed(2 * f), _Embed(f)
        self.rtg1, self.rtg2 = _Embed(2 * f), _Embed(f)
        self.up0 = nn.Sequential(nn.Conv2d(2 * f, 2 * f, 1), nn.GroupNorm(8, 2 * f), nn.ReLU())
        self.fuse1 = nn.Sequential(_ConvBlock(4 * f, 2 * f), _ConvBlock(2 * f, 2 * f))
        self.up1 = _Up(2 * f, f)
        self.fuse2 = nn.Sequential(_ConvBlock(2 * f, f), _ConvBlock(f, f))
        self.up2 = _Up(f, f)
        self.out = nn.Sequential(_ConvBlock(2 * f, f), nn.Conv2d(f, 1, 3, 1, 1))

    def forward(self, x, t, c, context_mask, r):
        b, f = x.shape[0], self.n_feat
        keep = 1.0 - context_mask[:, None].float()
        c = c.float() * keep
        r = r.reshape(b, 1).float() * keep
        x0 = self.init(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        h = F.interpolate(self.up0(self.to_vec(d2)), size=d2.shape[-2:], mode="bilinear", align_corners=False)
        t = t.reshape(b, 1).float()
        e1 = self.cond1(c).view(b, 2 * f, 1, 1) + self.rtg1(r).view(b, 2 * f, 1, 1)
        h = self.fuse1(torch.cat([e1 * h + self.time1(t).view(b, 2 * f, 1, 1), d2], 1))
        h = self.up1(h, d1.shape[-2:])
        e2 = self.cond2(c).view(b, f, 1, 1) + self.rtg2(r).view(b, f, 1, 1)
        h = self.fuse2(torch.cat([e2 * h + self.time2(t).view(b, f, 1, 1), d1], 1))
        h = self.up2(h, x0.shape[-2:])
        return self.out(torch.cat([h, x0], 1))


# --------------------------------------------------------------------- model
class FlowModel:
    def __init__(self, net: UNet, normalizer: FlowNormalizer, window: int, device):
        self.device = torch.device(device)
        self.net, self.normalizer, self.window = net.to(self.device), normalizer, int(window)

    def _condition(self, c):
        c = c.detach().cpu().numpy().copy()
        d = self.normalizer.state_dim
        c[:, :d] = self.normalizer.states(c[:, :d])
        return torch.as_tensor(c, device=self.device)

    def _velocity(self, x, t, c, r, guidance):
        b = x.shape[0]
        cond = self.net(x, t, c, torch.zeros(b, device=self.device), r)
        uncond = self.net(x, t, c, torch.ones(b, device=self.device), r)
        return guidance * cond + (1.0 - guidance) * uncond

    @torch.no_grad()
    def sample(self, state_action, rtg, n_steps=5, solver="midpoint", guidance=2.0):
        """Raw (states [B, T, d], rewards [B, T]) generated from raw first state-action pairs."""
        self.net.eval()
        c = self._condition(state_action)
        b = c.shape[0]
        x = torch.randn(b, 1, self.window, self.normalizer.state_dim + 1, device=self.device)
        ts = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=self.device)
        for i in range(int(n_steps)):
            t0, t1 = ts[i].repeat(b), ts[i + 1].repeat(b)
            dt = (t1 - t0).view(b, 1, 1, 1)
            v = self._velocity(x, t0, c, rtg, guidance)
            if solver == "midpoint":
                v = self._velocity(x + 0.5 * dt * v, t0 + 0.5 * (t1 - t0), c, rtg, guidance)
            x = x + dt * v
        window = x.squeeze(1).cpu().numpy()
        d = self.normalizer.state_dim
        states = self.normalizer.unstates(window[..., :d])
        rewards = self.normalizer.unrewards(window[..., d])
        return (torch.as_tensor(states, device=self.device), torch.as_tensor(rewards, device=self.device))

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "n_feat": self.net.n_feat, "window": self.window,
                    "normalizer": self.normalizer.state_dict()}, path)

    @classmethod
    def load(cls, path, device):
        blob = torch.load(path, map_location=device, weights_only=False)
        net = UNet(blob["n_feat"])
        net.load_state_dict(blob["net"])
        return cls(net, FlowNormalizer.from_state_dict(blob["normalizer"]), blob["window"], device)


def train_flow(episodes, seed: int, device, window=12, gamma=0.99, **overrides) -> FlowModel:
    """Fit one flow model to the ``window``-step windows of ``episodes``; keeps the epoch with
    the lowest loss."""
    hp = {**FLOW_DEFAULTS, **overrides}
    T, gamma = int(window), float(gamma)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    normalizer = FlowNormalizer.fit(episodes, T, gamma)
    xs, cs, rs = [], [], []
    for ep in episodes:
        if len(ep.observations) < T:
            continue
        s, r = normalizer.states(ep.observations), normalizer.rewards(ep.rewards)
        g = normalizer.rtg(window_returns(ep.rewards, T, gamma, complete=True))
        for start in range(len(ep.observations) - T + 1):
            xs.append(np.concatenate([s[start:start + T], r[start:start + T, None]], -1))
            cs.append(np.concatenate([s[start], ep.actions[start]]))
            rs.append(g[start])
    x = torch.as_tensor(np.asarray(xs, np.float32)).unsqueeze(1)
    c, r = torch.as_tensor(np.asarray(cs, np.float32)), torch.as_tensor(np.asarray(rs, np.float32))
    model = FlowModel(UNet(int(hp["n_feat"])), normalizer, T, device)
    opt = torch.optim.Adam(model.net.parameters(), lr=float(hp["lr"]))
    bs, per_epoch = int(hp["batch_size"]), int(hp["batches_per_epoch"])
    best, best_state, start_time = float("inf"), None, time.time()
    for epoch in range(1, int(hp["epochs"]) + 1):
        order = torch.randperm(len(x), generator=gen)[: per_epoch * bs]
        losses = []
        for idx in order.split(bs):
            if len(idx) < bs:
                break
            x1, cb, rb = x[idx].to(model.device), c[idx].to(model.device), r[idx].to(model.device)
            x0 = torch.randn_like(x1)
            t = torch.rand(bs, device=model.device)
            xt = (1.0 - t.view(bs, 1, 1, 1)) * x0 + t.view(bs, 1, 1, 1) * x1
            drop = (torch.rand(bs, device=model.device) < float(hp["drop_prob"])).float()
            loss = F.mse_loss(model.net(xt, t, cb, drop, rb), x1 - x0)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        if mean_loss < best:
            best, best_state = mean_loss, {k: v.detach().clone() for k, v in model.net.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == int(hp["epochs"]):
            print(f"[flow] seed {seed} epoch {epoch}/{hp['epochs']} loss {mean_loss:.5f} best {best:.5f} "
                  f"({(time.time() - start_time) / 60:.1f} min)", flush=True)
    model.net.load_state_dict(best_state)
    return model


def flow_models(cfg, data, out: Path, count: int, seed: int, device) -> list[FlowModel]:
    """``count`` flow models of the run: loaded from ``hparams.flow_dir`` if given, otherwise
    trained on the run's dataset (model ``k`` from seed ``(seed, k)``) into ``<run>/flow``."""
    hp = cfg["algorithm"]["hparams"]
    folder = Path(hp["flow_dir"]) if hp.get("flow_dir") else Path(out) / "flow"
    folder.mkdir(parents=True, exist_ok=True)
    models = []
    for k in range(int(count)):
        path = folder / f"model{k}.pt"
        if path.exists():
            models.append(FlowModel.load(path, device))
            continue
        member_seed = int(np.random.SeedSequence([seed, k]).generate_state(1)[0] % (2 ** 31))
        model = train_flow(data.episodes, member_seed, device, window=hp.get("window", 12),
                           gamma=hp.get("gamma", 0.99), **(hp.get("flow") or {}))
        model.save(path)
        models.append(model)
    for model in models:
        model.net.requires_grad_(False).eval()
    return models
