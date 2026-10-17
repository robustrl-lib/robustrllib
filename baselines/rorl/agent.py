"""RORL (Yang et al., 2022) through its released implementation (``lifelong_rl``, MIT licence,
see LICENSE in this folder; from https://github.com/YangRui2015/RORL at commit
c6040438b037180047e68d1f243d8f172e610e00).

The released SAC trainer carries RORL's losses: an ensemble of ``num_qs`` critics with
smoothness and out-of-distribution penalties on perturbed observations and policy
smoothing. Two additions to the released objective, both on observations normalised with
the dataset statistics:

* for the first ``policy_eval_start`` updates the actor is trained by behaviour cloning
  (the release's warm start);
* from update ``bc_regularization_start`` on, ``bc_regularization_weight`` times the
  negative log-likelihood of the dataset actions is added to the actor loss.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

TRAINER_KEYS = ("discount", "reward_scale", "use_automatic_entropy_tuning", "target_entropy", "policy_lr",
                "qf_lr", "soft_target_tau", "target_update_period", "max_q_backup", "deterministic_backup",
                "policy_eval_start", "eta", "num_qs", "num_samples", "policy_smooth_eps", "policy_smooth_reg",
                "q_smooth_eps", "q_smooth_reg", "q_smooth_tau", "q_ood_eps", "q_ood_reg",
                "q_ood_uncertainty_reg", "q_ood_uncertainty_reg_min", "q_ood_uncertainty_decay")


def _stub(name, **attrs):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        sys.modules[name] = types.SimpleNamespace(**attrs)


def lifelong_rl(device):
    """Import the released modules (stubbing the timing and logging packages they import
    but RORL's updates never use) and point them at ``device``."""
    folder = str(Path(__file__).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    _stub("gtimer", stamp=lambda *a, **k: None, reset=lambda *a, **k: None, start=lambda *a, **k: None,
          timed_for=lambda it, **k: it, get_times=lambda: None)
    _stub("tensorboardX", SummaryWriter=None)
    import lifelong_rl.torch.pytorch_util as ptu
    from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
    from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
    from lifelong_rl.trainers.q_learning.sac import SACTrainer

    device = torch.device(device)
    ptu.set_gpu_mode(device.type == "cuda", gpu_id=device.index or 0)
    return ParallelizedEnsembleFlattenMLP, TanhGaussianPolicy, SACTrainer


def make_policy(state_dim, action_dim, mean, std, hp, device):
    _, TanhGaussianPolicy, _ = lifelong_rl(device)
    policy = TanhGaussianPolicy(obs_dim=state_dim, action_dim=action_dim,
                                hidden_sizes=[int(hp["hidden_size"])] * int(hp["num_policy_layers"]),
                                layer_norm=None, norm_input=True, obs_norm_mean=mean, obs_norm_std=std)
    release_log_probs = type(policy).get_log_probs

    def get_log_probs(obs, actions):
        # the released log-likelihood helper skips the input normalisation of forward()
        return release_log_probs(policy, (obs - policy.obs_norm_mean) / policy.obs_norm_std, actions)

    policy.get_log_probs = get_log_probs
    return policy.to(device)


class RORLAgent:
    def __init__(self, env, mean, std, hp, device):
        Ensemble, _, SACTrainer = lifelong_rl(device)
        self.device = torch.device(device)
        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.mean, self.std, self.hp = np.asarray(mean, np.float32), np.asarray(std, np.float32), hp
        q_kwargs = dict(ensemble_size=int(hp["num_qs"]), hidden_sizes=[int(hp["hidden_size"])] * int(hp["num_q_layers"]),
                        input_size=self.state_dim + self.action_dim, output_size=1, layer_norm=None,
                        norm_input=True, obs_norm_mean=self.mean, obs_norm_std=self.std)
        self.policy = make_policy(self.state_dim, self.action_dim, self.mean, self.std, hp, device)
        self.trainer = SACTrainer(env=env, policy=self.policy, qfs=Ensemble(**q_kwargs).to(self.device),
                                  target_qfs=Ensemble(**q_kwargs).to(self.device), norm_input=True,
                                  obs_std=self.std, **{k: hp[k] for k in TRAINER_KEYS if k in hp})
        self.bc_weight = float(hp.get("bc_regularization_weight", 0.0))
        self.bc_start = int(hp.get("bc_regularization_start", 0))

    def update(self, batch) -> dict:
        tensors = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
        step = self.trainer._num_train_steps + 1
        hook = None
        if self.bc_weight > 0 and step >= self.bc_start:
            # add bc_weight * grad(-log pi(a_D | s)) to the release's single actor step
            params = [p for p in self.policy.parameters() if p.requires_grad]
            bc_loss = -self.policy.get_log_probs(tensors["observations"], tensors["actions"]).mean()
            grads = torch.autograd.grad(bc_loss, params, allow_unused=True)

            def add_bc(optimizer, args, kwargs):
                for p, g in zip(params, grads):
                    if g is not None:
                        p.grad = g * self.bc_weight if p.grad is None else p.grad.add_(g, alpha=self.bc_weight)

            hook = self.trainer.policy_optimizer.register_step_pre_hook(add_bc)
        self.trainer._num_train_steps = step
        try:
            self.trainer.train_from_torch(tensors, None)
        finally:
            if hook is not None:
                hook.remove()
        stats = {}
        for key, value in self.trainer.get_diagnostics().items():
            value = np.asarray(value)
            if value.size == 1 and np.issubdtype(value.dtype, np.number):
                stats[key.lower().replace(" ", "_")] = float(value.reshape(-1)[0])
        return stats

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        self.policy.eval()
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, -1)
        action = self.policy(obs, deterministic=True)[0][0].cpu().numpy()
        self.policy.train()
        return np.clip(action, -1.0, 1.0)

    def save(self, path) -> None:
        torch.save({"hparams": dict(self.hp), "dims": [self.state_dim, self.action_dim],
                    "obs_mean": self.mean, "obs_std": self.std, "policy": self.policy.state_dict()}, path)
