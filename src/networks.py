"""Q-network definition. See docs/ARCHITECTURE.md section 2."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src.exceptions import ConfigurationError


@dataclass(frozen=True)
class NetworkConfig:
    """Layer sizes of the Q-network."""

    observation_size: int = 128
    hidden_size: int = 256
    num_actions: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_size", self.observation_size),
            ("hidden_size", self.hidden_size),
            ("num_actions", self.num_actions),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")


def normalize_observation(observation: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Scale RAM bytes to ``float32`` values in [0.0, 1.0]."""
    return observation.astype(np.float32) / 255.0


class QNetwork(nn.Module):
    """MLP mapping a normalized RAM state to one action-value per action.

    Args:
        config: Layer sizes. The defaults give 99,844 trainable parameters.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config: NetworkConfig = config if config is not None else NetworkConfig()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(self.config.observation_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return action-values shaped ``(num_actions,)`` or ``(batch, num_actions)``.

        Raises:
            ValueError: If ``states`` is not ``float32`` with shape
                ``(observation_size,)`` or ``(batch, observation_size)``.
        """
        if states.dtype != torch.float32:
            raise ValueError(f"states must be float32, got {states.dtype}")
        if states.ndim not in (1, 2) or states.shape[-1] != self.config.observation_size:
            raise ValueError(
                f"states must have shape ({self.config.observation_size},) or "
                f"(batch, {self.config.observation_size}), got {tuple(states.shape)}"
            )
        output: torch.Tensor = self.layers(states)
        return output
