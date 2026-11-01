"""Shared test doubles."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScriptedStep:
    """What the scripted environment does on one step."""

    reward: float = 0.0
    lives: int = 5
    terminated: bool = False
    truncated: bool = False


class ScriptedEnv(gym.Env[NDArray[np.uint8], int]):
    """A Breakout stand-in that replays a fixed script.

    The first byte of each observation is a running counter, so tests can tell observations
    apart. After the script runs out, each game repeats it from the start.
    """

    def __init__(self, script: list[ScriptedStep], lives: int = 5) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (128,), np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.script = script
        self.start_lives = lives
        self.actions: list[int] = []
        self.reset_seeds: list[int | None] = []
        self.closed = False
        self._position = 0
        self._counter = 0

    def _observation(self) -> NDArray[np.uint8]:
        observation = np.zeros(128, dtype=np.uint8)
        observation[0] = self._counter % 256
        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        self.reset_seeds.append(seed)
        self._position = 0
        self._counter += 1
        return self._observation(), {"lives": self.start_lives}

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        step = self.script[self._position % len(self.script)]
        self._position += 1
        self._counter += 1
        return (
            self._observation(),
            step.reward,
            step.terminated,
            step.truncated,
            {"lives": step.lives},
        )

    def close(self) -> None:
        self.closed = True
