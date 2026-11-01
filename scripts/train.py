"""Training entry point. The configuration and CLI exist; the training loop does not yet.

Run from the project root::

    uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device mps
"""

from dataclasses import dataclass
from typing import Literal

import torch
import tyro


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one run. See docs/EXPERIMENTS.md for each value's source."""

    variant: Literal["dqn_n1", "ddqn_n1", "ddqn_n3", "ddqn_n5"] = "ddqn_n3"
    """Ablation configuration."""
    seed: int = 1
    """Seed for torch, numpy and gymnasium."""
    device: Literal["mps", "cpu"] = "mps"
    """Compute device. A request for mps falls back to cpu if it is unavailable."""
    wandb_mode: Literal["online", "offline"] = "online"
    """Weights & Biases sync mode."""

    total_timesteps: int = 500_000
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_capacity: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.1

    learning_starts: int = 10_000
    train_frequency: int = 4
    huber_delta: float = 1.0
    clip_rewards: bool = True
    """Use the sign of each reward in training targets. Logged returns stay raw."""
    terminal_on_life_loss: bool = True
    """End training targets at a lost life, without resetting the environment."""
    auto_fire: bool = True
    """Press FIRE once after each reset and lost life."""

    @property
    def run_name(self) -> str:
        """Run identifier; scripts/plot_results.py parses this format."""
        return f"{self.variant}_seed{self.seed}"


def resolve_device(requested: Literal["mps", "cpu"]) -> torch.device:
    """Return the requested device, falling back from mps to cpu if unavailable."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("warning: mps is unavailable, using cpu")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    """Parse the CLI and report that the training loop is missing."""
    config = tyro.cli(TrainConfig)
    device = resolve_device(config.device)
    raise NotImplementedError(
        f"Training loop not implemented (run {config.run_name} on {device}). "
        "See docs/ARCHITECTURE.md."
    )


if __name__ == "__main__":
    main()
