"""CPU tests of the SPiDR, RAMU and ATLA components (run inside the Isaac Lab Python environment)."""
import copy

import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaac.methods import ramu, spidr
from isaac.methods.atla import ATLARunner


# --------------------------------------------------------------------------- SPiDR solver
def test_gae_reduces_to_known_returns():
    rng = np.random.default_rng(0)
    rewards, values, bootstrap = (torch.tensor(rng.normal(size=s)) for s in ((5, 3), (5, 3), (3,)))
    zero = torch.zeros(5, 3)
    targets, _ = spidr.gae(zero, zero, rewards, values, bootstrap, 0.9, 1.0)
    expected = bootstrap.clone()
    for t in reversed(range(5)):
        expected = rewards[t] + 0.9 * expected
        assert torch.allclose(targets[t], expected)
    targets, _ = spidr.gae(zero, zero, rewards, values, bootstrap, 0.9, 0.0)
    next_values = torch.cat((values[1:], bootstrap[None]))
    assert torch.allclose(targets, rewards + 0.9 * next_values)
    truncation = zero.clone()
    truncation[2] = 1.0
    _, advantages = spidr.gae(truncation, zero, rewards, values, bootstrap, 0.9, 0.95)
    assert torch.equal(advantages[2], torch.zeros(3, dtype=advantages.dtype))


def test_budget_and_multiplier_step():
    assert spidr.discounted_budget(20, 1000, 0.9) == pytest.approx(0.2)
    multiplier = torch.tensor(0.01)
    for _ in range(128):
        spidr.multiplier_step(multiplier, torch.tensor(0.3), 0.2, 0.01)
    assert multiplier.item() == pytest.approx(0.138, abs=1e-6)
    spidr.multiplier_step(multiplier, torch.tensor(-1e6), 0.2, 0.01)
    assert multiplier.item() == 0.0


def test_policy_loss_adds_the_weighted_cost_surrogate():
    ratio = torch.tensor([0.5, 0.9, 1.1, 1.5], requires_grad=True)
    multiplier = torch.tensor(2.0, requires_grad=True)
    total, reward_loss, cost_loss = spidr.policy_loss(ratio, torch.tensor([-1.0, 1, -1, 1]),
                                                      torch.tensor([1.0, -1, -1, 1]), multiplier, 0.3)
    assert torch.allclose(total, reward_loss + 2 * cost_loss)
    total.backward()
    assert multiplier.grad is None


def _models(obs, normalise):
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 2, hidden_dims=[8], activation="elu",
                     obs_normalization=normalise, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0})
    critic = MLPModel(obs, {"critic": ["policy"]}, "critic", 1, hidden_dims=[8], activation="elu",
                      obs_normalization=normalise)
    return actor, critic


def _rollout(alg, steps, extras_fn):
    obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
    with torch.inference_mode():
        for t in range(steps):
            alg.act(obs)
            obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
            dones = torch.tensor([t == 13, t == 24, False, False]).long()
            alg.process_env_step(obs, torch.randn(4), dones, extras_fn(t, obs))
        alg.compute_returns(obs)
    return alg.update()


@pytest.mark.parametrize("constrained", [False, True])
def test_primal_dual_ppo_update(constrained):
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
    actor, critic = _models(obs, normalise=True)
    storage = RolloutStorage("rl", 4, 40, obs, [2])
    alg = spidr.PrimalDualPPO(actor, critic, storage, constrained=constrained, episode_length=1000,
                              schedule="fixed", learning_rate=3e-4, gamma=0.97, num_learning_epochs=2,
                              num_mini_batches=2)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    cost_before = [p.detach().clone() for p in alg.cost_critic.parameters()]
    extras = lambda t, obs: dict(cost=torch.full((4,), 0.1), time_outs=torch.tensor([t == 13, False, False, False]))
    result = _rollout(alg, 40, extras)
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(cost_before, alg.cost_critic.parameters())) == constrained
    assert alg.multiplier.item() != 0.01 if constrained else alg.multiplier.item() == 0.0
    assert storage.step == 0


def test_ensemble_replication_helpers():
    tensor = torch.arange(18).reshape(9, 2)
    spidr._copy_to_members(tensor, behavior_envs=3, ensemble_size=2)
    assert torch.equal(tensor[3:6], tensor[:3]) and torch.equal(tensor[6:9], tensor[:3])
    state = {"articulation": {"robot": {"q": torch.arange(12).reshape(4, 3)}}}
    copied = spidr._repeat_scene_state(state, behavior_envs=2, ensemble_size=2)["articulation"]["robot"]["q"]
    assert torch.equal(copied, state["articulation"]["robot"]["q"][:2].repeat(2, 1))


# --------------------------------------------------------------------------- RAMU
def test_wang_distortion_is_pessimistic():
    weights = ramu.wang_weights(5, 0.75)
    assert weights.sum().item() == pytest.approx(1.0) and torch.all(weights[:-1] > weights[1:])
    samples = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert ramu.wang_value(samples, 0.75).item() < 3.0 < ramu.wang_value(samples, 0.75, cost=True).item()
    assert ramu.wang_value(samples, 0.0).item() == pytest.approx(3.0)


def test_sampled_next_observations_perturb_only_masked_entries():
    current, following = torch.zeros(4, 3), torch.ones(4, 3)
    mask = torch.tensor([True, False, True])
    sampled = ramu.sample_next(current, following, mask, 5, 0.1, torch.Generator().manual_seed(0))
    assert sampled.shape == (5, 4, 3) and torch.equal(sampled[..., 1], torch.ones(5, 4))
    assert ((sampled[..., 0] - 1).abs() <= 0.2 + 1e-6).all()


def test_residual_trace():
    targets, values = torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0], [1], [0]])
    assert torch.equal(ramu.residual_trace(targets, values, dones, 0.9, 0.0), targets - values)
    trace = ramu.residual_trace(targets, values, dones, 0.9, 1.0)
    assert trace[1].item() == pytest.approx(1.5) and trace[0].item() == pytest.approx(0.5 + 0.9 * 1.5)


def test_guarded_step_matches_adam_and_rolls_back():
    a, b = torch.nn.Parameter(torch.tensor([1.0, 2.0])), torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    x, y = torch.optim.Adam([a], lr=0.01), torch.optim.Adam([b], lr=0.01)
    a.grad, b.grad = torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2])
    x.step()
    assert ramu.guarded_step([y], [b], lambda: True, 8) and torch.equal(a, b)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.Adam([p], lr=0.1)
    p.grad = torch.ones_like(p)
    assert ramu.guarded_step([opt], [p], lambda: p.item() >= 0.97, 4)
    assert abs(p.item() - 0.975) < 1e-6 and opt.state[p]["step"].item() == 1 and opt.param_groups[0]["lr"] == 0.1
    before, state = p.detach().clone(), copy.deepcopy(opt.state_dict())
    p.grad = torch.ones_like(p)
    assert not ramu.guarded_step([opt], [p], lambda: False, 2)
    assert torch.equal(p, before) and opt.state[p]["step"].item() == state["state"][0]["step"].item()
    p.grad = torch.tensor([float("nan")])
    assert not ramu.guarded_step([opt], [p], lambda: True, 2)


def test_ramu_ppo_update():
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
    actor, critic = _models(obs, normalise=False)
    storage = RolloutStorage("rl", 4, 24, obs, [2])
    alg = ramu.RAMUPPO(actor, critic, storage, feature_masks={"policy": [True] * 3 + [False] * 3}, sampling_seed=1,
                       episode_length=1000, schedule="adaptive", learning_rate=1e-3, desired_kl=0.01,
                       num_learning_epochs=2, num_mini_batches=2)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    extras = lambda t, obs: dict(next_observation=obs, terminal=torch.zeros(4, dtype=torch.bool),
                                 raw_cost=torch.full((4,), 0.1), time_outs=torch.zeros(4, dtype=torch.bool))
    result = _rollout(alg, 24, extras)
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))


# --------------------------------------------------------------------------- ATLA
def test_atla_perturbation_is_bounded_per_dimension():
    runner = object.__new__(ATLARunner)
    runner.epsilon = 0.05
    runner.attack_scale = torch.tensor([1.0, 5.0, 0.4])
    obs = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    attacked = runner._attacked(obs, torch.full((2, 3), 100.0))
    assert torch.allclose(attacked["policy"], 0.05 * runner.attack_scale.expand(2, -1))
    assert torch.equal(obs["policy"], torch.zeros(2, 3))
