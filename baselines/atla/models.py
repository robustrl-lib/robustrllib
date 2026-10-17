"""Policy and value networks of the ATLA trainer.

Adapted from ATLA (https://github.com/huanzhang12/ATLA_robust_RL),
``src/policy_gradients/models.py``: the Gaussian MLP policy (ATLA-PPO and the
feed-forward adversary), the Gaussian LSTM policy (ATLA-SA-PPO and its adversary),
and the MLP and LSTM value networks. Two hidden layers of 64 units, tanh,
orthogonal initialisation.
"""
import math

import numpy as np
import torch as ch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

from .torch_utils import determinant

HIDDEN_SIZES = (64, 64)
STD = 2 ** 0.5


def orthogonal_init(tensor, gain=1):
    """Orthogonal initialisation as in OpenAI baselines."""
    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)
    if rows < cols:
        flattened.t_()
    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=STD):
    """Orthogonal weights with gain ``scale``, zero biases."""
    for p in mod.parameters():
        if len(p.data.shape) >= 2:
            orthogonal_init(p.data, gain=scale)
        else:
            p.data.zero_()


def pack_history(features, not_dones):
    """Split a ``(N, dim)`` sequence into episodes at ``not_dones == 0`` and pack them for an LSTM."""
    nnz = ch.nonzero(1.0 - not_dones, as_tuple=False).view(-1).cpu().numpy()
    all_pieces = []
    lengths = []
    start = end = 0
    for i in nnz:
        end = i + 1
        all_pieces.append(features[start:end, :])
        lengths.append(end - start)
        start = end
    # the unfinished episode at the end of the sequence
    if end != features.size(0):
        all_pieces.append(features[end:, :])
        lengths.append(features.size(0) - end)
    padded = pad_sequence(all_pieces, batch_first=True)
    return pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)


def unpack_history(padded_pieces, lengths):
    """Inverse of ``pack_history`` on the LSTM output: back to ``(N, hidden)``."""
    all_pieces = []
    for i, l in enumerate(lengths.cpu().numpy()):
        all_pieces.append(padded_pieces[i, 0:l, :])
    return ch.cat(all_pieces, dim=0)


class ValueDenseNet(nn.Module):
    """MLP state-value network."""

    def __init__(self, state_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.activation = nn.Tanh()
        self.affine_layers = nn.ModuleList()
        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            initialize_weights(l)
            self.affine_layers.append(l)
            prev = h
        self.final = nn.Linear(prev, 1)
        initialize_weights(self.final, scale=1.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return self.final(x)


class ValueLSTMNet(nn.Module):
    """LSTM state-value network over the episodes of a rollout."""

    def __init__(self, state_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.embedding_layer = nn.Linear(state_dim, self.hidden_sizes[0])
        initialize_weights(self.embedding_layer, scale=0.01)
        self.lstm = nn.LSTM(input_size=self.hidden_sizes[0], hidden_size=self.hidden_sizes[1],
                            num_layers=1, batch_first=True)
        self.final = nn.Linear(self.hidden_sizes[-1], 1)
        initialize_weights(self.final, scale=1.0)

    def forward(self, states, not_dones):
        """``states`` (1, N, dim) split into episodes by ``not_dones`` (1, N); returns (1, N, 1)."""
        assert states.size(0) == 1 and states.ndim == 3
        features = self.embedding_layer(states.squeeze(0))
        packed_features = pack_history(features, not_dones.squeeze(0))
        outputs, _ = self.lstm(packed_features)
        padded, lengths = pad_packed_sequence(outputs, batch_first=True)
        hidden = unpack_history(padded, lengths)
        return self.final(hidden).unsqueeze(0)

    def multi_forward(self, x, hidden):
        embeddings = self.embedding_layer(x)
        outputs, hidden = self.lstm(embeddings, hidden)
        return self.final(outputs), hidden


class CtsPolicy(nn.Module):
    """Gaussian MLP policy: state-dependent mean, state-independent log standard deviation."""

    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.activation = nn.Tanh()
        self.action_dim = action_dim
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i, bias=True)
            initialize_weights(lin)
            self.affine_layers.append(lin)
            prev_size = i
        self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        initialize_weights(self.final_mean, scale=0.01)
        self.log_stdev = ch.nn.Parameter(ch.zeros(action_dim))

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return self.final_mean(x), ch.exp(self.log_stdev)

    def sample(self, p):
        means, std = p
        return (means + ch.randn_like(means) * std).detach()

    def get_loglikelihood(self, p, actions):
        mean, std = p
        nll = 0.5 * ((actions - mean) / std).pow(2).sum(-1) \
            + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
            + self.log_stdev.sum(-1)
        return -nll

    def entropies(self, p):
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        return ch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))

    def reset(self):
        return


class CtsLSTMPolicy(CtsPolicy):
    """Gaussian LSTM policy.

    One observation of shape (1, dim) at a time advances the stored hidden state
    (acting); a whole rollout (N, dim) with its ``not_dones`` is split into
    episodes (training, SA regulariser); ``multi_forward`` runs a padded batch of
    episode chunks from a given hidden state.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__(state_dim, action_dim, hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.embedding_layer = nn.Linear(state_dim, self.hidden_sizes[0])
        initialize_weights(self.embedding_layer, scale=0.01)
        self.lstm = nn.LSTM(input_size=self.hidden_sizes[0], hidden_size=self.hidden_sizes[1],
                            num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_sizes[-1], action_dim)
        initialize_weights(self.output_layer, scale=1.0)
        self.log_stdev = ch.nn.Parameter(ch.zeros(action_dim))
        self.reset()

    def forward(self, x, not_dones=None):
        if x.size(0) != 1:
            assert not_dones is not None
            features = self.embedding_layer(x)
            packed_features = pack_history(features, not_dones)
            outputs, _ = self.lstm(packed_features)
            padded, lengths = pad_packed_sequence(outputs, batch_first=True)
            hidden = unpack_history(padded, lengths)
            return self.output_layer(hidden), ch.exp(self.log_stdev)
        assert x.ndim == 2 and not_dones is None
        embedding = self.embedding_layer(x).unsqueeze(0)
        _, hidden = self.lstm(embedding, self.hidden)
        output = self.output_layer(hidden[0])
        self.hidden = [hidden[0], hidden[1]]
        return output.squeeze(0), ch.exp(self.log_stdev)

    def multi_forward(self, x, hidden=None):
        embeddings = self.embedding_layer(x)
        outputs, hidden = self.lstm(embeddings, hidden)
        return self.output_layer(outputs), ch.exp(self.log_stdev), hidden

    def reset(self):
        self.hidden = [ch.zeros(1, 1, self.hidden_sizes[1]), ch.zeros(1, 1, self.hidden_sizes[1])]
