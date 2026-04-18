from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = obs
        for _ in range(self.num_hidden_layers):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, log_std


class Critic(nn.Module):
    hidden_dim: int = 128
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for _ in range(self.num_hidden_layers):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x).squeeze(-1)
