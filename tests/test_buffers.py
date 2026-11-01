"""N-step assembly and replay. Expected values are worked out by hand from ARCHITECTURE.md."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.buffers import NStepWindow, ReplayBuffer, Transition
from src.exceptions import ConfigurationError


def obs(value: int) -> NDArray[np.uint8]:
    """Return an observation identified by its first byte."""
    state = np.zeros(128, dtype=np.uint8)
    state[0] = value
    return state


def play(
    window: NStepWindow,
    rewards: list[float],
    *,
    terminated: bool = False,
    truncated: bool = False,
    life_lost: bool = False,
) -> list[Transition]:
    """Feed one segment; states are 0, 1, ... and only the last step has the boundary flags."""
    committed: list[Transition] = []
    for t, reward in enumerate(rewards):
        last = t == len(rewards) - 1
        committed += window.push(
            obs(t),
            t,
            reward,
            obs(t + 1),
            terminated=terminated and last,
            truncated=truncated and last,
            life_lost=life_lost and last,
        )
    return committed


def summary(transitions: list[Transition]) -> list[tuple[int, int, int, bool]]:
    """Reduce transitions to (first state, next state, span, done)."""
    return [(int(t.state[0]), int(t.next_state[0]), t.span, t.done) for t in transitions]


def test_full_window_commits_entire_return_and_oldest_action() -> None:
    window = NStepWindow(n=3, gamma=0.5)
    committed = play(window, [1.0, 2.0, 4.0])

    assert len(committed) == 1
    transition = committed[0]
    assert transition.action == 0
    assert transition.n_step_return == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert summary(committed) == [(0, 3, 3, False)]
    assert len(window) == 2


def test_windows_slide_one_step_at_a_time() -> None:
    assert summary(play(NStepWindow(n=3, gamma=0.5), [0.0] * 5)) == [
        (0, 3, 3, False),
        (1, 4, 3, False),
        (2, 5, 3, False),
    ]


def test_termination_flushes_short_spans_with_their_own_returns() -> None:
    window = NStepWindow(n=3, gamma=0.5)
    committed = play(window, [1.0, 1.0, 1.0, 1.0], terminated=True)

    assert summary(committed) == [
        (0, 3, 3, False),
        (1, 4, 3, True),
        (2, 4, 2, True),
        (3, 4, 1, True),
    ]
    assert [t.n_step_return for t in committed] == pytest.approx([1.75, 1.75, 1.5, 1.0])
    assert len(window) == 0


def test_only_windows_ending_on_the_boundary_are_done() -> None:
    committed = play(NStepWindow(n=2, gamma=1.0), [0.0, 0.0, 0.0], life_lost=True)
    assert summary(committed) == [(0, 2, 2, False), (1, 3, 2, True), (2, 3, 1, True)]


def test_truncation_flushes_without_setting_done() -> None:
    window = NStepWindow(n=3, gamma=0.5)
    committed = play(window, [1.0, 1.0], truncated=True)
    assert summary(committed) == [(0, 2, 2, False), (1, 2, 1, False)]
    assert len(window) == 0


def test_lost_life_on_a_truncated_step_is_still_done() -> None:
    committed = play(NStepWindow(n=3, gamma=0.5), [1.0], truncated=True, life_lost=True)
    assert summary(committed) == [(0, 1, 1, True)]


def test_window_does_not_cross_a_boundary() -> None:
    window = NStepWindow(n=3, gamma=1.0)
    play(window, [1.0, 1.0], life_lost=True)
    committed = play(window, [10.0, 10.0, 10.0])
    assert [t.n_step_return for t in committed] == [30.0]


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 11])
@pytest.mark.parametrize("boundary", ["terminated", "truncated", "life_lost"])
def test_no_duplicate_or_dropped_transitions(n: int, length: int, boundary: str) -> None:
    window = NStepWindow(n=n, gamma=0.9)
    committed = play(window, [1.0] * length, **{boundary: True})

    assert [t.action for t in committed] == list(range(length))
    assert len(window) == 0
    for transition in committed:
        assert transition.span == min(n, length - transition.action)
        assert int(transition.next_state[0]) == transition.action + transition.span
        ends_on_boundary = transition.action + transition.span == length
        assert transition.done == (ends_on_boundary and boundary != "truncated")
        expected = sum(0.9**k for k in range(transition.span))
        assert transition.n_step_return == pytest.approx(expected)


def test_n_equal_one_commits_immediately() -> None:
    assert summary(play(NStepWindow(n=1, gamma=0.9), [1.0, 1.0])) == [
        (0, 1, 1, False),
        (1, 2, 1, False),
    ]


@pytest.mark.parametrize(("n", "gamma"), [(0, 0.9), (1, 0.0), (1, 1.5)])
def test_window_rejects_invalid_arguments(n: int, gamma: float) -> None:
    with pytest.raises(ConfigurationError):
        NStepWindow(n=n, gamma=gamma)


def transition(value: int) -> Transition:
    return Transition(obs(value), 2, float(value), obs(value + 1), value % 2 == 0, 3)


def test_replay_overwrites_oldest_at_capacity() -> None:
    buffer = ReplayBuffer(capacity=3)
    for value in range(5):
        buffer.add(transition(value))

    assert len(buffer) == 3
    batch = buffer.sample(200, np.random.default_rng(0), torch.device("cpu"))
    assert set(batch.n_step_returns.tolist()) == {2.0, 3.0, 4.0}


def test_replay_sample_normalizes_and_keeps_fields_aligned() -> None:
    buffer = ReplayBuffer(capacity=10)
    buffer.add(transition(51))
    batch = buffer.sample(4, np.random.default_rng(0), torch.device("cpu"))

    assert batch.states.dtype == torch.float32
    assert batch.states.shape == (4, 128)
    assert batch.states[0, 0].item() == pytest.approx(0.2)
    assert batch.next_states[0, 0].item() == pytest.approx(52 / 255)
    assert batch.actions.dtype == torch.int64
    assert batch.actions.tolist() == [2] * 4
    assert batch.n_step_returns.tolist() == [51.0] * 4
    assert batch.dones.tolist() == [0.0] * 4
    assert batch.spans.tolist() == [3.0] * 4


def test_replay_sampling_is_reproducible_from_the_generator() -> None:
    buffer = ReplayBuffer(capacity=50)
    for value in range(50):
        buffer.add(transition(value))
    device = torch.device("cpu")
    first = buffer.sample(16, np.random.default_rng(7), device)
    second = buffer.sample(16, np.random.default_rng(7), device)
    assert torch.equal(first.n_step_returns, second.n_step_returns)


def test_replay_samples_with_replacement() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.add(transition(0))
    assert len(buffer.sample(8, np.random.default_rng(0), torch.device("cpu")).actions) == 8


@pytest.mark.parametrize(
    "bad",
    [np.zeros(127, dtype=np.uint8), np.zeros((1, 128), dtype=np.uint8), np.zeros(128)],
)
def test_replay_rejects_invalid_observations(bad: NDArray[np.uint8]) -> None:
    buffer = ReplayBuffer(capacity=2)
    with pytest.raises(ValueError, match="uint8 with shape"):
        buffer.add(Transition(bad, 0, 0.0, obs(0), False, 1))
    with pytest.raises(ValueError, match="uint8 with shape"):
        buffer.add(Transition(obs(0), 0, 0.0, bad, False, 1))
    assert len(buffer) == 0


def test_replay_rejects_sampling_when_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        ReplayBuffer(capacity=2).sample(1, np.random.default_rng(0), torch.device("cpu"))


def test_replay_rejects_invalid_capacity() -> None:
    with pytest.raises(ConfigurationError, match="capacity"):
        ReplayBuffer(capacity=0)
