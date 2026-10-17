"""Causal mask of RSC-IQL: which observation dimensions the learned dynamics depend on.

A per-dimension encode-mask-decode dynamics model predicts (next state, reward) from
(state, action) through a learnable sparse mask over input -> output edges, sampled
with a straight-through Gumbel-Bernoulli. ``fit_mask`` trains it on the run's dataset
and writes, per observation dimension, the normalised importance (summed edge
probability), ``prob_d = 1 - importance`` and ``sigma_d = sigma_max * (1 - importance)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_DEFAULTS = {"epochs": 50, "lr": 1e-4, "batch_size": 512, "causal_dim": 32, "hidden_dim": 256,
                 "num_hidden": 3, "sparse_weight": 0.5, "sparse_norm": 1.0, "sigma_max": 0.1}


def _mlp(in_dim, out_dim, hidden, depth):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(depth):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(hidden, out_dim))


class CausalDynamics(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256, causal_dim=32, emb_dim=10,
                 num_hidden=3, sparse_weight=0.5, sparse_norm=1.0, tau=1.0):
        super().__init__()
        self.state_dim, self.input_dim, self.output_dim = state_dim, state_dim + act_dim, state_dim + 1
        self.sparse_weight, self.sparse_norm, self.tau = sparse_weight, sparse_norm, tau
        self.encoder = _mlp(emb_dim + 1, causal_dim, hidden_dim, num_hidden)
        self.encoder_emb = nn.Embedding(self.input_dim, emb_dim)
        self.decoder = _mlp(causal_dim + emb_dim, 1, hidden_dim, num_hidden)
        self.decoder_emb = nn.Embedding(self.output_dim, emb_dim)
        self.mask_logits = nn.Parameter(3.0 * torch.ones(self.input_dim, self.output_dim))

    def sample_mask(self):
        prob = torch.sigmoid(self.mask_logits)
        logits = torch.stack([torch.log(1 - prob + 1e-8), torch.log(prob + 1e-8)], dim=-1)
        return F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)[..., 1]

    def forward(self, s, a):
        b = s.size(0)
        x = torch.cat([s, a], dim=-1).unsqueeze(-1)
        enc = self.encoder_emb(torch.arange(self.input_dim, device=s.device)).expand(b, -1, -1)
        z = torch.einsum("bic,io->boc", self.encoder(torch.cat([x, enc], dim=-1)), self.sample_mask())
        dec = self.decoder_emb(torch.arange(self.output_dim, device=s.device)).expand(b, -1, -1)
        out = self.decoder(torch.cat([z, dec], dim=-1)).squeeze(-1)
        return out[:, :self.state_dim], out[:, self.state_dim]

    def loss(self, s, a, s_next, r):
        s_pred, r_pred = self.forward(s, a)
        sparse = self.sparse_weight * (torch.sigmoid(self.mask_logits) ** self.sparse_norm).mean()
        return F.mse_loss(s_pred, s_next) + F.mse_loss(r_pred, r) + sparse

    @torch.no_grad()
    def state_importance(self):
        return torch.sigmoid(self.mask_logits[:self.state_dim]).sum(dim=1)


def fit_mask(data, path, seed: int, device, **overrides) -> Path:
    """Fit the causal mask on ``data`` (raw observations) and save it to ``path``."""
    hp = {**MASK_DEFAULTS, **overrides}
    torch.manual_seed(seed)
    tensors = [torch.as_tensor(x) for x in (data.states, data.actions, data.next_states, data.rewards.reshape(-1))]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=int(hp["batch_size"]),
                                         shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    model = CausalDynamics(data.state_dim, data.action_dim, int(hp["hidden_dim"]), int(hp["causal_dim"]),
                           num_hidden=int(hp["num_hidden"]), sparse_weight=float(hp["sparse_weight"]),
                           sparse_norm=float(hp["sparse_norm"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    for epoch in range(1, int(hp["epochs"]) + 1):
        for s, a, s2, r in loader:
            loss = model.loss(s.to(device), a.to(device), s2.to(device), r.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[mask] epoch {epoch}/{hp['epochs']} loss {loss.item():.5f}", flush=True)
    importance = model.state_importance().cpu().numpy()
    importance = (importance / importance.max()).astype(np.float32)
    path = Path(path)
    np.savez(path, importance=importance, prob_d=(1.0 - importance).astype(np.float32),
             sigma_d=(float(hp["sigma_max"]) * (1.0 - importance)).astype(np.float32),
             fit=json.dumps(hp))
    print(f"[mask] importance {np.round(importance, 3).tolist()} -> {path}", flush=True)
    return path
