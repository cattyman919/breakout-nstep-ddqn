"""DQN and Double DQN agent. See docs/ARCHITECTURE.md section 3.

n-step Double DQN is this same agent fed by an ``NStepWindow`` with ``n > 1``: the span
stored with each transition sets the bootstrap discount.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from src.buffers import TransitionBatch
from src.exceptions import ConfigurationError, TrainingError
from src.networks import NetworkConfig, QNetwork, normalize_observation


@dataclass(frozen=True)
class AgentConfig:
    """Learning settings that define an agent."""

    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 1e-4
    huber_delta: float = 1.0
    double_q: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigurationError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 < self.tau <= 1.0:
            raise ConfigurationError(f"tau must be in (0, 1], got {self.tau}")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.huber_delta <= 0.0:
            raise ConfigurationError(f"huber_delta must be positive, got {self.huber_delta}")


@dataclass(frozen=True)
class UpdateResult:
    """Metrics from one gradient update."""

    td_loss: float
    mean_q: float


class DQNAgent:
    """Online and target Q-networks with epsilon-greedy action selection.

    Args:
        config: Learning settings. ``double_q`` selects the DQN or Double DQN target.
        device: Device holding both networks.
        network_config: Layer sizes of both networks.
    """

    def __init__(
        self,
        config: AgentConfig,
        device: torch.device,
        network_config: NetworkConfig | None = None,
    ) -> None:
        self.config: AgentConfig = config
        self.device: torch.device = device
        self.online: QNetwork = QNetwork(network_config).to(device)
        self.target: QNetwork = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )

    def act(self, observation: NDArray[np.uint8], epsilon: float, rng: np.random.Generator) -> int:
        """Choose an epsilon-greedy action, skipping the network when exploring."""
        if rng.random() < epsilon:
            return self._random_action(rng)
        return int(self._action_values(observation).argmax().item())

    def act_with_value(
        self, observation: NDArray[np.uint8], epsilon: float, rng: np.random.Generator
    ) -> tuple[int, float]:
        """Choose an epsilon-greedy action and return it with ``max_a Q(s, a)``."""
        values = self._action_values(observation)
        action = self._random_action(rng) if rng.random() < epsilon else int(values.argmax().item())
        return action, float(values.max().item())

    def compute_targets(self, batch: TransitionBatch) -> torch.Tensor:
        """Return ``G + (1 - d) * gamma^m * Q_hat(s')``, which carries no gradient."""
        with torch.no_grad():
            next_values = self.target(batch.next_states)
            if self.config.double_q:
                next_actions = self.online(batch.next_states).argmax(dim=1, keepdim=True)
                bootstrap = next_values.gather(1, next_actions).squeeze(1)
            else:
                bootstrap = next_values.max(dim=1).values
            discounts = torch.pow(self.config.gamma, batch.spans)
            targets: torch.Tensor = (
                batch.n_step_returns + (1.0 - batch.dones) * discounts * bootstrap
            )
        return targets

    def update(self, batch: TransitionBatch, step: int) -> UpdateResult:
        """Take one gradient step, then move the target network toward the online network.

        Raises:
            TrainingError: If the loss or a gradient is NaN or infinite. Weights are left
                unchanged, because the check runs before the optimizer step.
        """
        targets = self.compute_targets(batch)
        all_values = self.online(batch.states)
        with torch.no_grad():
            mean_q = float(all_values.max(dim=1).values.mean().item())
        chosen_values = all_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        loss = F.huber_loss(chosen_values, targets, delta=self.config.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        self._check_finite(loss, step)
        self.optimizer.step()
        self._soft_update_target()
        return UpdateResult(td_loss=float(loss.item()), mean_q=mean_q)

    def _action_values(self, observation: NDArray[np.uint8]) -> torch.Tensor:
        state = torch.from_numpy(normalize_observation(observation)).to(self.device)
        with torch.no_grad():
            values: torch.Tensor = self.online(state)
        return values

    def _random_action(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.online.config.num_actions))

    def _check_finite(self, loss: torch.Tensor, step: int) -> None:
        if not torch.isfinite(loss).item():
            raise TrainingError(f"non-finite loss {loss.item()} at step {step}")
        for name, parameter in self.online.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise TrainingError(f"non-finite gradient in {name} at step {step}")

    def _soft_update_target(self) -> None:
        """Apply ``phi' <- tau * phi + (1 - tau) * phi'``."""
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(), self.online.parameters(), strict=True
            ):
                target.lerp_(online, self.config.tau)
