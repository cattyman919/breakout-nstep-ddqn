"""Breakout environment access: lives, automatic FIRE and clipped rewards.

See docs/ARCHITECTURE.md section 1. ``BreakoutGame`` is shared by training, evaluation and
the random baseline, so all three see the same episode rules.
"""

from dataclasses import dataclass
from types import TracebackType
from typing import Any, ClassVar, Self

import ale_py
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.exceptions import GameEnvironmentError


def make_breakout_env(render_mode: str | None = None) -> gym.Env[Any, Any]:
    """Create the RAM Breakout environment.

    Args:
        render_mode: Passed to Gymnasium. ``"rgb_array"`` makes ``render`` return frames.

    Raises:
        GameEnvironmentError: If the environment cannot be created.
    """
    gym.register_envs(ale_py)
    try:
        return gym.make("ALE/Breakout-v5", obs_type="ram", render_mode=render_mode)
    except gym.error.Error as err:
        raise GameEnvironmentError(f"could not create the Breakout environment: {err}") from err


def clip_reward(raw_reward: float) -> float:
    """Return the sign of a reward: -1.0, 0.0 or 1.0."""
    return float(np.sign(raw_reward))


@dataclass(frozen=True)
class StepResult:
    """The outcome of one environment step."""

    observation: NDArray[np.uint8]
    raw_reward: float
    clipped_reward: float
    terminated: bool
    truncated: bool
    life_lost: bool

    @property
    def game_over(self) -> bool:
        """Whether the environment must be reset before the next step."""
        return self.terminated or self.truncated


class BreakoutGame:
    """A game session that tracks lives and says when FIRE must be pressed.

    The caller performs the press through ``step``, so that it counts as an ordinary
    environment step. The session owns the environment and closes it on exit.

    Args:
        env: The wrapped environment, with ``info["lives"]`` and 128-byte observations.
        seed: Seeds the environment on the first reset only, and its action space.
        auto_fire: Request one FIRE press after each reset and lost life.
    """

    FIRE_ACTION: ClassVar[int] = 1
    OBSERVATION_SHAPE: ClassVar[tuple[int]] = (128,)

    def __init__(self, env: gym.Env[Any, Any], seed: int, *, auto_fire: bool = True) -> None:
        self._env: gym.Env[Any, Any] = env
        self._seed: int | None = seed
        self._auto_fire: bool = auto_fire
        self._lives: int = 0
        self.needs_fire: bool = False
        self._env.action_space.seed(seed)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def reset(self) -> NDArray[np.uint8]:
        """Start a new game and return its first observation."""
        observation, info = self._env.reset(seed=self._seed)
        self._seed = None
        self._lives = self._read_lives(info)
        self.needs_fire = self._auto_fire
        return self._check_observation(observation)

    def step(self, action: int) -> StepResult:
        """Apply one action. ``needs_fire`` is updated from what happened."""
        observation, reward, terminated, truncated, info = self._env.step(action)
        lives = self._read_lives(info)
        life_lost = lives < self._lives
        self._lives = lives
        if action == self.FIRE_ACTION:
            self.needs_fire = False
        if life_lost:
            self.needs_fire = self._auto_fire
        raw_reward = float(reward)
        return StepResult(
            observation=self._check_observation(observation),
            raw_reward=raw_reward,
            clipped_reward=clip_reward(raw_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            life_lost=life_lost,
        )

    def press_fire(self) -> StepResult:
        """Perform the automatic press."""
        return self.step(self.FIRE_ACTION)

    def _read_lives(self, info: dict[str, Any]) -> int:
        try:
            return int(info["lives"])
        except KeyError as err:
            raise GameEnvironmentError(
                f"environment info has no 'lives' entry: {sorted(info)}"
            ) from err

    def _check_observation(self, observation: Any) -> NDArray[np.uint8]:
        """Validate an observation and return a private copy of it.

        The n-step window holds references to earlier observations, so none may share
        memory with an emulator buffer.
        """
        array = np.array(observation, copy=True)
        if array.dtype != np.uint8 or array.shape != self.OBSERVATION_SHAPE:
            raise GameEnvironmentError(
                f"expected a uint8 observation with shape {self.OBSERVATION_SHAPE}, "
                f"got {array.dtype} with shape {array.shape}"
            )
        return array
