import numpy as np
import torch
from torch import nn as nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class RiskZofMuMlpModel(torch.nn.Module):
    """Z portion of the model for DSAC, an MLP."""

    def __init__(
        self,
        observation_shape,
        hidden_sizes,
        action_size,
        embedding_size=64,
        layer_norm=True,
    ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)

        self.layer_norm = layer_norm
        self.embedding_size = embedding_size

        # hidden_sizes[:-1] MLP base
        # hidden_sizes[-1] before output

        self.phi_fc = []
        last_size = int(np.prod(observation_shape)) + action_size
        for next_size in hidden_sizes[:-1]:
            self.phi_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.phi_fc = nn.Sequential(*self.phi_fc)
        self.psi_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.z_head = nn.Linear(hidden_sizes[-1], 1)
        self.h_head = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = nn.Parameter(torch.arange(self.embedding_size), requires_grad=False)

    def forward(self, observation, prev_action, prev_reward, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        z_input = torch.cat([observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        hidden = self.phi_fc(z_input)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.psi_fc(x)  # (N, T, C)

        hidden = torch.mul(x + 1., hidden.unsqueeze(-2))  # (N, T, C)
        hidden = self.merge_fc(hidden)  # (N, T, C)
        z = self.z_head(hidden).squeeze(-1)  # (N, T)
        h = self.h_head(hidden).squeeze(-1)  # (N, T)
        z, h = restore_leading_dims((z, h), lead_dim, T, B)
        return z, h
