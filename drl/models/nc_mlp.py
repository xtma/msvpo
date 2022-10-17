import numpy as np
import torch
from torch import nn as nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class NonCrossZofMuMlpModel(torch.nn.Module):
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

        layers = []
        last_size = int(np.prod(observation_shape)) + action_size
        for next_size in hidden_sizes:
            layers += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.mlp = nn.Sequential(*layers)
        self.interval_head = nn.Sequential(nn.Linear(last_size, embedding_size), nn.Softmax(-1))
        self.scale_head = nn.Sequential(nn.Linear(last_size, 1), nn.Softplus())
        self.position_head = nn.Linear(last_size, 1)

    def forward(self, observation, prev_action, prev_reward, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, M)
        """
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        z_input = torch.cat([observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        hidden = self.mlp(z_input)

        position = self.position_head(hidden)  # (N, 1)
        scale = self.scale_head(hidden)  # (N, 1)
        intervals = self.interval_head(hidden)  # (N, E)
        values = torch.cat([position, intervals.cumsum(-1) * scale + position], -1)  # (N, E+1)
        i_l = torch.floor(tau * self.embedding_size).clamp(min=0, max=self.embedding_size)  # (N, M)
        i_r = torch.ceil(tau * self.embedding_size).clamp(min=0, max=self.embedding_size)  # (N, M)
        z_l = values.gather(-1, i_l.long())  # (N, T)
        z_r = values.gather(-1, i_r.long())  # (N, T)
        # if i_r > i_l: i_r = i_l + 1, z_r > z_l
        # else: i_r = i_l, z_r = z_l
        z = z_l + (tau * self.embedding_size - i_l) * (z_r - z_l)
        z = restore_leading_dims(z, lead_dim, T, B)
        return z
