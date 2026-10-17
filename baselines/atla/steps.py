"""PPO, value and SA-PPO update steps of the ATLA trainer.

Adapted from ATLA (https://github.com/huanzhang12/ATLA_robust_RL),
``src/policy_gradients/steps.py``: clipped PPO for the MLP and LSTM policies,
the GAE value regression, and the state-adversarial (SA) regulariser of SA-PPO,
whose inner maximisation is SGLD over an L-infinity ball of observations.
"""
import functools
import math

import numpy as np
import torch as ch
from torch.nn.utils.rnn import pad_sequence

from .torch_utils import shape_equal_cmp


class LinearEpsSchedule:
    """Radius of the SA regulariser: zero until epoch ``start``, then linear to ``max_eps``
    over ``length`` epochs, interpolated over the ``epoch_length`` batches of an epoch
    (the LinearScheduler of auto_LiRPA that SA-PPO uses). ``opts`` reads "start=1,length=1831".
    """

    def __init__(self, max_eps, opts):
        params = dict(o.split("=") for o in opts.split(",") if o.strip())
        self.start = int(params["start"])
        self.length = int(params["length"])
        self.max_eps = max_eps
        self.epoch = self.batch = 0
        self.eps = self.epoch_start_eps = self.epoch_end_eps = 0.0
        self.epoch_length = float("nan")

    def set_epoch_length(self, epoch_length):
        self.epoch_length = epoch_length

    def step_epoch(self):
        self.epoch += 1
        self.batch = 0
        if self.epoch < self.start:
            self.epoch_start_eps = self.epoch_end_eps = 0.0
        elif self.length == 0:
            self.epoch_start_eps = self.epoch_end_eps = self.max_eps
        else:
            k = self.epoch - self.start
            step = self.max_eps / self.length
            self.epoch_start_eps = min(k * step, self.max_eps)
            self.epoch_end_eps = min((k + 1) * step, self.max_eps)
        self.eps = self.epoch_start_eps

    def step_batch(self):
        self.batch += 1
        step = (self.epoch_end_eps - self.epoch_start_eps) / self.epoch_length
        self.eps = self.epoch_start_eps + step * (self.batch - 1)
        if self.batch > self.epoch_length:
            self.eps = self.epoch_end_eps

    def get_eps(self):
        return self.eps


def adv_normalize(adv, mask=None):
    if mask is None:
        if adv.nelement() == 1:
            return adv
        std = adv.std()
        mean = adv.mean()
    else:
        masked_adv = adv[mask]
        if masked_adv.nelement() == 1:
            return adv
        std = masked_adv.std()
        mean = masked_adv.mean()
    assert std != 0. and not ch.isnan(std), 'Need nonzero std'
    return (adv - mean) / (std + 1e-8)


def surrogate_reward(adv, *, new, old, clip_eps=None, mask=None, normalize=True):
    """``r_t A_t`` with ``r_t = exp(log p_new - log p_old)``, clamped to ``1 +- clip_eps`` if given."""
    n_advs = adv_normalize(adv, mask) if normalize else adv
    assert shape_equal_cmp(new, old, n_advs)
    ratio_new_old = ch.exp(new - old)
    if clip_eps is not None:
        ratio_new_old = ch.clamp(ratio_new_old, 1 - clip_eps, 1 + clip_eps)
    return ratio_new_old * n_advs


def value_loss_gae(vs, advantages, not_dones, params, old_vs, mask=None, reduction='mean'):
    """Regression of ``v`` on ``(v_old + A)``, optionally clipped around ``v_old``."""
    val_targ = (old_vs + advantages).detach()
    assert shape_equal_cmp(val_targ, vs, not_dones, old_vs, advantages)
    vs_clipped = old_vs + ch.clamp(vs - old_vs, -params.CLIP_VAL_EPS, params.CLIP_VAL_EPS)
    # no loss on the last step of an episode (no return to regress on)
    sel = ch.logical_and(not_dones.bool(), mask)
    val_loss_mat_unclipped = (vs - val_targ)[sel].pow(2)
    val_loss_mat_clipped = (vs_clipped - val_targ)[sel].pow(2)
    if params.VALUE_CLIPPING:
        val_loss_mat = ch.max(val_loss_mat_unclipped, val_loss_mat_clipped)
    else:
        val_loss_mat = val_loss_mat_unclipped
    if reduction == 'mean':
        return val_loss_mat.mean()
    return val_loss_mat.sum()


def value_step(all_states, returns, advantages, not_dones, net, val_opt, params, old_vs):
    """``VAL_EPOCHS`` epochs of value regression: minibatches for the MLP, whole episodes for the LSTM."""
    recurrent = params.HISTORY_LENGTH > 0
    if recurrent:
        batches, alive_masks, time_masks, lengths = pack_history(
            [all_states, returns, not_dones, advantages, old_vs], not_dones,
            max_length=params.HISTORY_LENGTH)
    for _ in range(params.VAL_EPOCHS):
        if recurrent:
            hidden = None
            val_opt.zero_grad()
            val_loss = 0.0
            for i, batch in enumerate(batches):
                batch_states, batch_returns, batch_not_dones, batch_advs, batch_old_vs = batch
                if hidden is not None:
                    hidden = [h[:, alive_masks[i], :].detach() for h in hidden]
                vs, hidden = net.multi_forward(batch_states, hidden=hidden)
                vs = vs.squeeze(-1)
                val_loss += value_loss_gae(vs, batch_advs, batch_not_dones, params, batch_old_vs,
                                           mask=time_masks[i], reduction='sum')
            val_loss = val_loss / all_states.size(0)
            val_loss.backward()
            val_opt.step()
        else:
            state_indices = np.arange(returns.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, params.NUM_MINIBATCHES)
            for selected in splits:
                val_opt.zero_grad()
                vs = net(all_states[selected]).squeeze(-1)
                val_loss = value_loss_gae(vs, advantages[selected], not_dones[selected], params,
                                          old_vs[selected], mask=ch.tensor(True))
                val_loss.backward()
                val_opt.step()
    return val_loss


def pack_history(features, not_dones, max_length):
    """Cut a rollout into episodes, pad them, and slice them into chunks of ``max_length`` steps.

    ``features`` is a list of tensors of shape (N, ...) holding several episodes,
    separated where ``not_dones`` is 0. Returns the chunks (one list of tensors per
    chunk, holding only the episodes still alive), the mask of episodes alive into
    each chunk, the time mask of each chunk and the episode lengths; an LSTM carries
    its hidden state from one chunk to the next.
    """
    nnz = ch.nonzero(1.0 - not_dones, as_tuple=False).view(-1).cpu().numpy()
    size = features[0].size(0)
    for t in features:
        assert size == t.size(0)
    all_pieces = [[] for _ in range(len(features))]
    lengths = []
    start = end = 0
    for i in nnz:
        end = i + 1
        for (a, b) in zip(all_pieces, features):
            a.append(b[start:end])
        lengths.append(end - start)
        start = end
    # the unfinished episode at the end of the rollout
    if end != size:
        for (a, b) in zip(all_pieces, features):
            a.append(b[end:])
        lengths.append(size - end)
    padded_features = [pad_sequence(a, batch_first=True) for a in all_pieces]
    longest = padded_features[0].size(1)
    extra = int(math.ceil(longest / max_length) * max_length - longest)
    new_padded_features = []
    for t in padded_features:
        if t.ndim == 3:
            new_tensor = ch.zeros(t.size(0), extra, t.size(2))
        else:
            new_tensor = ch.zeros(t.size(0), extra)
        new_padded_features.append(ch.cat([t, new_tensor], dim=1))
    nbatches = new_padded_features[0].size(1) // max_length
    alive_masks = []
    time_masks = []
    batches = [[] for _ in range(nbatches)]
    alive = ch.tensor(lengths)
    alive_iter = ch.tensor(lengths)
    for i in range(nbatches):
        full_mask = alive > 0
        iter_mask = alive_iter > 0
        for t in new_padded_features:
            batches[i].append(t[full_mask, i * max_length: i * max_length + max_length])
        alive_iter = alive_iter[iter_mask]
        time_mask = alive_iter.view(-1, 1) > ch.arange(max_length).view(1, -1)
        alive -= max_length
        alive_iter -= max_length
        alive_masks.append(iter_mask)
        time_masks.append(time_mask)
    return batches, alive_masks, time_masks, lengths


def _lstm_surrogate(net, batches, alive_masks, time_masks, clip_eps, n_states):
    """Clipped PPO surrogate of an LSTM policy over the chunks of ``pack_history``."""
    hidden = None
    surrogate = 0.0
    for i, batch in enumerate(batches):
        batch_states, batch_actions, batch_old_log_ps, batch_advs = batch
        mask = time_masks[i]
        if hidden is not None:
            hidden = [h[:, alive_masks[i], :].detach() for h in hidden]
        mean, std, hidden = net.multi_forward(batch_states, hidden=hidden)
        dist = mean, std
        new_log_ps = net.get_loglikelihood(dist, batch_actions)
        # advantages are normalised once over the whole rollout
        unclp_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps, mask=mask,
                                     normalize=False)
        clp_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps,
                                   clip_eps=clip_eps, mask=mask, normalize=False)
        surrogate = surrogate + (-ch.min(unclp_rew, clp_rew) * mask).sum()
    return surrogate / n_states, dist


def ppo_step(all_states, actions, old_log_ps, rewards, returns, not_dones, advs, net, params):
    """``PPO_EPOCHS`` epochs of clipped PPO (https://arxiv.org/abs/1707.06347) on ``net``."""
    recurrent = params.HISTORY_LENGTH > 0
    if recurrent:
        batches, alive_masks, time_masks, _ = pack_history(
            [all_states, actions, old_log_ps, adv_normalize(advs)], not_dones,
            max_length=params.HISTORY_LENGTH)
    for _ in range(params.PPO_EPOCHS):
        if recurrent:
            params.POLICY_ADAM.zero_grad()
            surrogate, dist = _lstm_surrogate(net, batches, alive_masks, time_masks,
                                              params.CLIP_EPS, all_states.size(0))
            entropy_bonus = net.entropies(dist)
            entropy = -params.ENTROPY_COEFF * entropy_bonus
            loss = surrogate + entropy
            loss.backward()
            params.POLICY_ADAM.step()
        else:
            state_indices = np.arange(all_states.shape[0])
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, params.NUM_MINIBATCHES)
            for selected in splits:
                batch_states, batch_actions, batch_old_log_ps, batch_advs = (
                    v[selected] for v in (all_states, actions, old_log_ps, advs))
                dist = net(batch_states)
                new_log_ps = net.get_loglikelihood(dist, batch_actions)
                unclp_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps)
                clp_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps,
                                           clip_eps=params.CLIP_EPS)
                entropy_bonus = net.entropies(dist).mean()
                surrogate = (-ch.min(unclp_rew, clp_rew)).mean()
                entropy = -params.ENTROPY_COEFF * entropy_bonus
                loss = surrogate + entropy
                params.POLICY_ADAM.zero_grad()
                loss.backward()
                params.POLICY_ADAM.step()
    return loss.item(), surrogate.item(), entropy.item()


def get_state_kl_bound_sgld(net, batch_states, batch_action_means, eps, steps, stdev, not_dones=None):
    """Upper estimate of ``max_{|s'-s|_inf <= eps} ||(mu(s') - mu(s)) / sigma||^2`` by ``steps`` SGLD steps."""
    wrapped_net = functools.partial(net, not_dones=not_dones) if not_dones is not None else net
    if batch_action_means is None:
        with ch.no_grad():
            batch_action_means, _ = wrapped_net(batch_states)
    else:
        batch_action_means = batch_action_means.detach()
    states_ub = batch_states + eps
    states_lb = batch_states - eps
    step_eps = eps / steps
    # SGLD noise with inverse temperature 1e-5
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = ch.randn_like(batch_states) * noise_factor
    var_states = (batch_states.clone() + noise.sign() * step_eps).detach().requires_grad_()
    for i in range(steps):
        diff = (wrapped_net(var_states)[0] - batch_action_means) / stdev.detach()
        kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
        kl.backward()
        noise_factor = np.sqrt(2 * step_eps * beta) / (i + 2)
        update = (var_states.grad + noise_factor * ch.randn_like(var_states)).sign() * step_eps
        var_states.data += update
        var_states = ch.max(var_states, states_lb)
        var_states = ch.min(var_states, states_ub)
        var_states = var_states.detach().requires_grad_()
    net.zero_grad()
    diff = (wrapped_net(var_states.requires_grad_(False))[0] - batch_action_means) / stdev
    return (diff * diff).sum(axis=-1, keepdim=True)


def robust_ppo_step(all_states, actions, old_log_ps, rewards, returns, not_dones, advs, net, params,
                    eps_scheduler):
    """PPO on the LSTM policy plus ``ROBUST_PPO_REG`` times the SA regulariser (SA-PPO)."""
    assert params.HISTORY_LENGTH > 0, "the SA regulariser is implemented for the LSTM policy"
    # all PPO epochs of one iteration count as one scheduler epoch
    eps_scheduler.set_epoch_length(params.PPO_EPOCHS * params.NUM_MINIBATCHES)
    eps_scheduler.step_epoch()
    batches, alive_masks, time_masks, _ = pack_history(
        [all_states, actions, old_log_ps, adv_normalize(advs)], not_dones,
        max_length=params.HISTORY_LENGTH)
    for _ in range(params.PPO_EPOCHS):
        params.POLICY_ADAM.zero_grad()
        surrogate, dist = _lstm_surrogate(net, batches, alive_masks, time_masks,
                                          params.CLIP_EPS, all_states.size(0))
        entropy_bonus = net.entropies(dist)
        eps_scheduler.step_batch()
        kl_upper_bound = get_state_kl_bound_sgld(
            net, all_states, None, eps=eps_scheduler.get_eps(), steps=params.ROBUST_PPO_PGD_STEPS,
            stdev=ch.exp(net.log_stdev), not_dones=not_dones).mean()
        entropy = -params.ENTROPY_COEFF * entropy_bonus
        loss = surrogate + entropy + params.ROBUST_PPO_REG * kl_upper_bound
        loss.backward()
        params.POLICY_ADAM.step()
    return loss.item(), surrogate.item(), entropy_bonus.item()
