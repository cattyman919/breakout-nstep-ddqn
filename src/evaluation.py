"""Periodic evaluation: predicted values against returns earned. See ARCHITECTURE.md section 6."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.agents import DQNAgent
from src.environment import BreakoutGame


@dataclass(frozen=True)
class EvalResult:
    """Means over every state with a known return, across all games of one evaluation."""

    predicted_q: float
    actual_return: float
    num_states: int


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    """Return ``R_i`` for each position of one life's clipped rewards."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def evaluate(
    agent: DQNAgent,
    game: BreakoutGame,
    *,
    num_games: int,
    epsilon: float,
    gamma: float,
    rng: np.random.Generator,
) -> EvalResult | None:
    """Play complete games and compare predicted values with discounted clipped returns.

    Performs no gradient updates and stores nothing. Returns None if no state has a known
    return, which happens when every life was cut off by truncation.
    """
    predicted: list[float] = []
    actual: list[float] = []

    for _ in range(num_games):
        observation = game.reset()
        life_values: list[float] = []
        life_rewards: list[float] = []
        game_over = False
        while not game_over:
            if game.needs_fire:
                # A press is not an agent decision: no state, and its reward is in no R_i.
                result = game.press_fire()
            else:
                action, value = agent.act_with_value(observation, epsilon, rng)
                result = game.step(action)
                life_values.append(value)
                life_rewards.append(result.clipped_reward)
            observation = result.observation
            game_over = result.game_over
            if result.life_lost or result.terminated:
                predicted.extend(life_values)
                actual.extend(discounted_returns(life_rewards, gamma))
                life_values, life_rewards = [], []

    if not predicted:
        return None
    return EvalResult(
        predicted_q=float(np.mean(predicted)),
        actual_return=float(np.mean(actual)),
        num_states=len(predicted),
    )
