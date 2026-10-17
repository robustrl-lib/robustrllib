"""Elucidated diffusion over packed transitions, as in SynthER (Lu et al., 2023).

Adapted from https://github.com/conglu1997/SynthER (MIT licence, Copyright (c) 2023 Cong Lu,
Philip J. Ball; see LICENSE in this folder), whose diffusion code is adapted from
https://github.com/lucidrains/denoising-diffusion-pytorch and, for the sampler, from
https://github.com/crowsonkb/k-diffusion. The residual-MLP denoiser with random Fourier
time features, the Karras et al. (2022) preconditioning, noise schedule and stochastic
second-order sampler, and the training loop (AdamW, cosine schedule, gradient clipping,
exponential moving average of the weights) are those of SynthER's default configuration.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_pytorch import EMA

DIFFUSION_DEFAULTS = {"width": 2048, "layers": 6, "time_dim": 128, "fourier_dim": 16, "sample_steps": 128,
                      "sigma_min": 0.002, "sigma_max": 80.0, "sigma_data": 1.0, "rho": 7.0, "p_mean": -1.2,
                      "p_std": 1.2, "s_churn": 80.0, "s_tmin": 0.05, "s_tmax": 50.0, "s_noise": 1.003,
                      "lr": 3e-4, "batch_size": 1024, "small_batch_size": 256, "small_dataset": 800_000,
                      "steps": 100_000, "weight_decay": 0.0, "ema_decay": 0.995, "ema_every": 10}


class RandomFourierTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(dim // 2), requires_grad=False)

    def forward(self, t):
        freqs = t[:, None] * self.weights[None, :] * 2 * math.pi
        return torch.cat([t[:, None], freqs.sin(), freqs.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return x + self.linear(F.relu(x))


class Denoiser(nn.Module):
    def __init__(self, d_in, width=2048, layers=6, time_dim=128, fourier_dim=16):
        super().__init__()
        self.proj = nn.Linear(d_in, time_dim)
        self.time_mlp = nn.Sequential(RandomFourierTime(fourier_dim), nn.Linear(fourier_dim + 1, time_dim),
                                      nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.network = nn.Sequential(nn.Linear(time_dim, width), *[ResidualBlock(width) for _ in range(layers)])
        self.final_linear = nn.Linear(width, d_in)

    def forward(self, x, t):
        return self.final_linear(F.relu(self.network(self.proj(x) + self.time_mlp(t))))


class Normalizer(nn.Module):
    """Standardisation per input dimension; ``skip`` dimensions are left as they are."""

    def __init__(self, data, skip=()):
        super().__init__()
        mean, std = data.mean(0), data.std(0) + 1e-5
        mean[list(skip)], std[list(skip)] = 0.0, 1.0
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean


class ElucidatedDiffusion(nn.Module):
    def __init__(self, net, normalizer, dim, hp):
        super().__init__()
        self.net, self.normalizer, self.dim = net, normalizer, dim
        self.hp = {k: float(hp[k]) for k in ("sigma_min", "sigma_max", "sigma_data", "rho", "p_mean", "p_std",
                                              "s_churn", "s_tmin", "s_tmax", "s_noise")}

    def denoise(self, x, sigma):
        sd = self.hp["sigma_data"]
        if isinstance(sigma, float):
            sigma = torch.full((x.shape[0],), sigma, device=x.device)
        s = sigma.view(-1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd * (sd ** 2 + s ** 2) ** -0.5
        c_in = (s ** 2 + sd ** 2) ** -0.5
        return c_skip * x + c_out * self.net(c_in * x, torch.log(sigma.clamp(min=1e-20)) * 0.25)

    def forward(self, x):
        """Training loss on a batch of raw inputs."""
        x = self.normalizer.normalize(x)
        sigma = (self.hp["p_mean"] + self.hp["p_std"] * torch.randn(x.shape[0], device=x.device)).exp()
        noised = x + sigma.view(-1, 1) * torch.randn_like(x)
        weight = (sigma ** 2 + self.hp["sigma_data"] ** 2) * (sigma * self.hp["sigma_data"]) ** -2
        return ((self.denoise(noised, sigma) - x) ** 2).mean(-1).mul(weight).mean()

    @torch.no_grad()
    def sample(self, n, steps):
        hp, device = self.hp, next(self.net.parameters()).device
        inv_rho = 1 / hp["rho"]
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / (steps - 1)
        sigmas = (hp["sigma_max"] ** inv_rho + ramp * (hp["sigma_min"] ** inv_rho - hp["sigma_max"] ** inv_rho)) ** hp["rho"]
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        gamma = min(hp["s_churn"] / steps, math.sqrt(2) - 1)
        x = sigmas[0] * torch.randn(n, self.dim, device=device)
        for sigma, sigma_next in zip(sigmas[:-1].tolist(), sigmas[1:].tolist()):
            g = gamma if hp["s_tmin"] <= sigma <= hp["s_tmax"] else 0.0
            sigma_hat = sigma + g * sigma
            x_hat = x + math.sqrt(sigma_hat ** 2 - sigma ** 2) * hp["s_noise"] * torch.randn_like(x)
            d = (x_hat - self.denoise(x_hat, sigma_hat)) / sigma_hat
            x = x_hat + (sigma_next - sigma_hat) * d
            if sigma_next != 0:
                d_next = (x - self.denoise(x, sigma_next)) / sigma_next
                x = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_next)
        return self.normalizer.unnormalize(x)


def train_diffusion(inputs: np.ndarray, skip, seed: int, device, **overrides) -> ElucidatedDiffusion:
    """Fit the diffusion model to the rows of ``inputs``; returns the averaged-weights model."""
    hp = {**DIFFUSION_DEFAULTS, **overrides}
    torch.manual_seed(seed)
    data = torch.as_tensor(inputs, dtype=torch.float32)
    model = ElucidatedDiffusion(Denoiser(data.shape[1], int(hp["width"]), int(hp["layers"]), int(hp["time_dim"]),
                                         int(hp["fourier_dim"])), Normalizer(data, skip), data.shape[1], hp)
    ema = EMA(model, beta=float(hp["ema_decay"]), update_every=int(hp["ema_every"]))
    model.to(device)
    ema.to(device)
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not ("bias" in n or "norm" in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(hp["weight_decay"])},
                             {"params": no_decay, "weight_decay": 0.0}], lr=float(hp["lr"]), betas=(0.9, 0.99))
    steps = int(hp["steps"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    batch = int(hp["small_batch_size"] if len(data) < int(hp["small_dataset"]) else hp["batch_size"])
    gen = torch.Generator().manual_seed(seed)
    order, cursor = torch.randperm(len(data), generator=gen), 0
    for step in range(1, steps + 1):
        if cursor + batch > len(data):
            order, cursor = torch.randperm(len(data), generator=gen), 0
        x = data[order[cursor:cursor + batch]].to(device)
        cursor += batch
        loss = model(x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); schedule.step(); ema.update()
        if step % 1000 == 0 or step == steps:
            print(f"[diffusion] step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return ema.ema_model.eval()
