"""Evaluation returns: R_i stops at a lost life and truncated final lives are excluded."""

import numpy as np
import pytest
import torch
from conftest import ScriptedEnv, ScriptedStep

from src.agents import AgentConfig, DQNAgent
from src.environment import BreakoutGame
from src.evaluation import EvalResult, discounted_returns, evaluate


def run_eval(script: list[ScriptedStep], *, games: int = 1) -> tuple[EvalResult | None, DQNAgent]:
    torch.manual_seed(0)
    agent = DQNAgent(AgentConfig(), torch.device("cpu"))
    # Every state predicts 3.0, so predicted means are known exactly.
    agent.online.forward = lambda states: torch.tensor([3.0, 0.0, 0.0, 0.0])  # type: ignore[method-assign]
    with BreakoutGame(ScriptedEnv(script), seed=0) as game:
        result = evaluate(
            agent, game, num_games=games, epsilon=0.0, gamma=0.5, rng=np.random.default_rng(0)
        )
    return result, agent


def test_discounted_returns_by_hand() -> None:
    assert discounted_returns([1.0, 0.0, 1.0], 0.5) == pytest.approx([1.25, 0.5, 1.0])
    assert discounted_returns([], 0.5) == []


def test_return_stops_at_a_lost_life() -> None:
    script = [
        ScriptedStep(),  # press
        ScriptedStep(reward=0.0),
        ScriptedStep(reward=1.0, lives=4),  # R = [0.5, 1.0]
        ScriptedStep(),  # press
        ScriptedStep(reward=1.0, lives=0, terminated=True),  # R = [1.0]; earlier reward is cut off
    ]
    result, _ = run_eval(script)
    assert result is not None
    assert result.num_states == 3
    assert result.actual_return == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert result.predicted_q == pytest.approx(3.0)


def test_returns_use_clipped_rewards() -> None:
    script = [ScriptedStep(), ScriptedStep(reward=7.0, lives=0, terminated=True)]
    result, _ = run_eval(script)
    assert result is not None
    assert result.actual_return == pytest.approx(1.0)


def test_press_adds_no_state_and_its_reward_counts_for_nothing() -> None:
    script = [ScriptedStep(reward=1.0), ScriptedStep(reward=0.0, lives=0, terminated=True)]
    result, _ = run_eval(script)
    assert result is not None
    assert result.num_states == 1
    assert result.actual_return == 0.0


def test_truncated_final_life_is_excluded() -> None:
    script = [
        ScriptedStep(),  # press
        ScriptedStep(reward=1.0, lives=4),  # a complete life, R = [1.0]
        ScriptedStep(),  # press
        ScriptedStep(reward=1.0),
        ScriptedStep(reward=1.0, truncated=True),  # unknown returns
    ]
    result, _ = run_eval(script)
    assert result is not None
    assert result.num_states == 1
    assert result.actual_return == pytest.approx(1.0)


def test_life_lost_on_the_truncating_step_is_kept() -> None:
    script = [ScriptedStep(), ScriptedStep(reward=1.0, lives=4, truncated=True)]
    result, _ = run_eval(script)
    assert result is not None
    assert result.num_states == 1


def test_all_lives_truncated_gives_no_result() -> None:
    result, _ = run_eval([ScriptedStep(), ScriptedStep(truncated=True)])
    assert result is None


def test_means_span_all_games() -> None:
    script = [ScriptedStep(), ScriptedStep(reward=1.0, lives=0, terminated=True)]
    result, _ = run_eval(script, games=3)
    assert result is not None
    assert result.num_states == 3


def test_evaluation_changes_no_weights() -> None:
    script = [ScriptedStep(), ScriptedStep(lives=0, terminated=True)]
    torch.manual_seed(0)
    agent = DQNAgent(AgentConfig(), torch.device("cpu"))
    before = [parameter.clone() for parameter in agent.online.parameters()]
    with BreakoutGame(ScriptedEnv(script), seed=0) as game:
        evaluate(agent, game, num_games=1, epsilon=0.0, gamma=0.5, rng=np.random.default_rng(0))
    for old, new in zip(before, agent.online.parameters(), strict=True):
        assert torch.equal(old, new)
