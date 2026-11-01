"""Measure the random-policy return, the baseline for the pilot in docs/EXPERIMENTS.md section 5.

Run from the project root::

    uv run python -m scripts.measure_random
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import tyro

from src.environment import BreakoutGame, make_breakout_env
from src.exceptions import ConfigurationError, GameEnvironmentError


@dataclass(frozen=True)
class MeasureConfig:
    """Command-line options for the baseline."""

    games: int = 100
    """Complete games to play."""
    seed: int = 0
    """Seed for the environment and the action generator. Seed 0 is reserved for pilots."""

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ConfigurationError(f"games must be positive, got {self.games}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must not be negative, got {self.seed}")


def measure_random_returns(
    config: MeasureConfig,
    env_factory: Callable[[], gym.Env[Any, Any]] = make_breakout_env,
) -> list[float]:
    """Play complete games with a uniform random policy and automatic FIRE.

    Returns:
        The raw undiscounted return of each game, including any press reward.
    """
    rng = np.random.default_rng(config.seed)
    returns: list[float] = []
    env = env_factory()
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise GameEnvironmentError(f"expected discrete actions, got {action_space}")
    num_actions = int(action_space.n)
    with BreakoutGame(env, config.seed) as game:
        for _ in range(config.games):
            game.reset()
            total = 0.0
            game_over = False
            while not game_over:
                if game.needs_fire:
                    result = game.press_fire()
                else:
                    result = game.step(int(rng.integers(num_actions)))
                total += result.raw_reward
                game_over = result.game_over
            returns.append(total)
    return returns


def main() -> None:
    """Print the mean return and the pilot threshold."""
    config = tyro.cli(MeasureConfig)
    returns = measure_random_returns(config)
    mean = float(np.mean(returns))
    print(f"games: {len(returns)}")
    print(f"mean return: {mean:.3f}")
    print(f"standard deviation: {float(np.std(returns)):.3f}")
    print(f"pilot threshold (2 x mean): {2.0 * mean:.3f}")


if __name__ == "__main__":
    main()
