"""N-step transition assembly and uniform replay. See docs/ARCHITECTURE.md section 4."""

from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from src.exceptions import ConfigurationError
from src.networks import normalize_observation


@dataclass(frozen=True)
class Transition:
    """One stored transition: ``(s_t, a_t, G_t^(m), s_{t+m}, d_t, m)``."""

    state: NDArray[np.uint8]
    action: int
    n_step_return: float
    next_state: NDArray[np.uint8]
    done: bool
    span: int


@dataclass(frozen=True)
class TransitionBatch:
    """A sampled mini-batch on the target device, with states normalized to [0, 1]."""

    states: torch.Tensor
    actions: torch.Tensor
    n_step_returns: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    spans: torch.Tensor


class NStepWindow:
    """Sliding window in front of the replay buffer.

    Args:
        n: Largest number of steps a transition spans.
        gamma: Discount used inside each return.

    Raises:
        ConfigurationError: If ``n`` or ``gamma`` is out of range.
    """

    def __init__(self, n: int, gamma: float) -> None:
        if n < 1:
            raise ConfigurationError(f"n must be at least 1, got {n}")
        if not 0.0 < gamma <= 1.0:
            raise ConfigurationError(f"gamma must be in (0, 1], got {gamma}")
        self._n: int = n
        self._gamma: float = gamma
        self._window: deque[tuple[NDArray[np.uint8], int, float]] = deque(maxlen=n)

    def __len__(self) -> int:
        return len(self._window)

    def push(
        self,
        state: NDArray[np.uint8],
        action: int,
        clipped_reward: float,
        next_state: NDArray[np.uint8],
        *,
        terminated: bool,
        truncated: bool,
        life_lost: bool,
    ) -> list[Transition]:
        """Add one environment step and return the transitions it completes.

        ``next_state`` must be the real observation returned by this step. A truncated
        window stays bootstrappable, so ``truncated`` flushes but never sets ``done``.
        """
        self._window.append((state, action, clipped_reward))
        done = terminated or life_lost
        committed: list[Transition] = []

        if len(self._window) == self._n:
            committed.append(self._commit_oldest(next_state, done))
        if done or truncated:
            while self._window:
                committed.append(self._commit_oldest(next_state, done))
        return committed

    def _commit_oldest(self, next_state: NDArray[np.uint8], done: bool) -> Transition:
        """Build a transition for the oldest entry from the rewards left, then drop it."""
        n_step_return = sum(
            self._gamma**k * reward for k, (_, _, reward) in enumerate(self._window)
        )
        span = len(self._window)
        state, action, _ = self._window.popleft()
        return Transition(state, action, float(n_step_return), next_state, done, span)


class ReplayBuffer:
    """Fixed-size ring buffer with uniform sampling, with replacement.

    Filling up is normal: the oldest transition is then overwritten. States are stored as
    ``uint8``, which takes a quarter of the memory of ``float32``.

    Args:
        capacity: Largest number of stored transitions.
        observation_size: Length of each RAM observation.

    Raises:
        ConfigurationError: If ``capacity`` or ``observation_size`` is not positive.
    """

    STATE_DTYPE: ClassVar[type[np.uint8]] = np.uint8

    def __init__(self, capacity: int, observation_size: int = 128) -> None:
        if capacity <= 0:
            raise ConfigurationError(f"capacity must be positive, got {capacity}")
        if observation_size <= 0:
            raise ConfigurationError(f"observation_size must be positive, got {observation_size}")
        self.capacity: int = capacity
        self._observation_shape: tuple[int] = (observation_size,)
        self._states: NDArray[np.uint8] = np.zeros(
            (capacity, observation_size), dtype=self.STATE_DTYPE
        )
        self._next_states: NDArray[np.uint8] = np.zeros(
            (capacity, observation_size), dtype=self.STATE_DTYPE
        )
        self._actions: NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self._returns: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._dones: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._spans: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._cursor: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        """Store one transition, overwriting the oldest once at capacity.

        Raises:
            ValueError: If a state has the wrong shape or is not ``uint8``.
        """
        self._check_observation(transition.state, "state")
        self._check_observation(transition.next_state, "next_state")
        index = self._cursor
        self._states[index] = transition.state
        self._next_states[index] = transition.next_state
        self._actions[index] = transition.action
        self._returns[index] = transition.n_step_return
        self._dones[index] = float(transition.done)
        self._spans[index] = transition.span
        self._cursor = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: torch.device
    ) -> TransitionBatch:
        """Draw ``batch_size`` transitions uniformly, with replacement.

        Raises:
            ValueError: If the buffer is empty or ``batch_size`` is not positive.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        indices = rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            states=self._to_device(normalize_observation(self._states[indices]), device),
            actions=self._to_device(self._actions[indices], device),
            n_step_returns=self._to_device(self._returns[indices], device),
            next_states=self._to_device(normalize_observation(self._next_states[indices]), device),
            dones=self._to_device(self._dones[indices], device),
            spans=self._to_device(self._spans[indices], device),
        )

    def _check_observation(self, observation: NDArray[np.uint8], name: str) -> None:
        if observation.dtype != self.STATE_DTYPE or observation.shape != self._observation_shape:
            raise ValueError(
                f"{name} must be uint8 with shape {self._observation_shape}, "
                f"got {observation.dtype} with shape {observation.shape}"
            )

    @staticmethod
    def _to_device(values: NDArray[np.generic], device: torch.device) -> torch.Tensor:
        return torch.from_numpy(values).to(device)
