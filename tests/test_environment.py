"""Automatic FIRE, life tracking, clipping and environment validation."""

from typing import Any

import numpy as np
import pytest
from conftest import ScriptedEnv, ScriptedStep

from src.environment import BreakoutGame, clip_reward, make_breakout_env
from src.exceptions import GameEnvironmentError


def game_for(script: list[ScriptedStep], *, auto_fire: bool = True) -> BreakoutGame:
    return BreakoutGame(ScriptedEnv(script), seed=3, auto_fire=auto_fire)


def test_reset_requests_fire_and_press_clears_it() -> None:
    game = game_for([ScriptedStep()])
    game.reset()
    assert game.needs_fire
    game.press_fire()
    assert not game.needs_fire


def test_auto_fire_off_never_requests_fire() -> None:
    game = game_for([ScriptedStep(lives=4)], auto_fire=False)
    game.reset()
    assert not game.needs_fire
    assert game.step(0).life_lost
    assert not game.needs_fire


def test_lost_life_requests_fire_without_ending_the_game() -> None:
    game = game_for([ScriptedStep(), ScriptedStep(), ScriptedStep(lives=4)])
    game.reset()
    game.press_fire()
    result = game.step(2)
    assert not result.life_lost
    assert not game.needs_fire

    result = game.step(2)
    assert result.life_lost
    assert not result.game_over
    assert game.needs_fire


def test_life_is_lost_only_when_lives_drop() -> None:
    game = game_for([ScriptedStep(lives=4), ScriptedStep(lives=4), ScriptedStep(lives=3)])
    game.reset()
    assert [game.step(0).life_lost for _ in range(3)] == [True, False, True]


def test_press_that_loses_a_life_requests_another_press() -> None:
    game = game_for([ScriptedStep(lives=4)])
    game.reset()
    assert game.press_fire().life_lost
    assert game.needs_fire


def test_agent_fire_action_counts_as_the_press() -> None:
    game = game_for([ScriptedStep()])
    game.reset()
    game.step(BreakoutGame.FIRE_ACTION)
    assert not game.needs_fire


def test_non_fire_action_leaves_the_press_pending() -> None:
    game = game_for([ScriptedStep()])
    game.reset()
    game.step(0)
    assert game.needs_fire


def test_rewards_keep_raw_and_clipped_values_separate() -> None:
    game = game_for([ScriptedStep(reward=7.0)])
    game.reset()
    result = game.step(0)
    assert result.raw_reward == 7.0
    assert result.clipped_reward == 1.0


@pytest.mark.parametrize(("raw", "clipped"), [(0.0, 0.0), (1.0, 1.0), (7.0, 1.0), (-3.0, -1.0)])
def test_clip_reward_is_the_sign(raw: float, clipped: float) -> None:
    assert clip_reward(raw) == clipped


def test_game_over_covers_termination_and_truncation() -> None:
    game = game_for([ScriptedStep(terminated=True, lives=0)])
    game.reset()
    assert game.step(0).game_over
    game = game_for([ScriptedStep(truncated=True)])
    game.reset()
    assert game.step(0).game_over


def test_only_the_first_reset_is_seeded() -> None:
    env = ScriptedEnv([ScriptedStep()])
    game = BreakoutGame(env, seed=9)
    game.reset()
    game.reset()
    assert env.reset_seeds == [9, None]


def test_observations_do_not_share_memory_with_the_source() -> None:
    source = np.zeros(128, dtype=np.uint8)

    class SharedBufferEnv(ScriptedEnv):
        def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            _, reward, terminated, truncated, info = super().step(action)
            source[0] += 1
            return source, reward, terminated, truncated, info

    game = BreakoutGame(SharedBufferEnv([ScriptedStep()]), seed=0)
    game.reset()
    first = game.step(0).observation
    game.step(0)
    assert first[0] == 1


def test_context_manager_closes_the_env_on_failure() -> None:
    env = ScriptedEnv([ScriptedStep()])
    with pytest.raises(RuntimeError, match="boom"), BreakoutGame(env, seed=0):
        raise RuntimeError("boom")
    assert env.closed


def test_wrong_observation_shape_raises() -> None:
    class BadEnv(ScriptedEnv):
        def _observation(self) -> Any:
            return np.zeros(64, dtype=np.uint8)

    with pytest.raises(GameEnvironmentError, match="shape"):
        BreakoutGame(BadEnv([ScriptedStep()]), seed=0).reset()


def test_missing_lives_raises() -> None:
    class NoLivesEnv(ScriptedEnv):
        def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, Any]:
            observation, _ = super().reset(seed=seed, options=options)
            return observation, {}

    with pytest.raises(GameEnvironmentError, match="lives"):
        BreakoutGame(NoLivesEnv([ScriptedStep()]), seed=0).reset()


def test_real_environment_matches_the_documented_spaces() -> None:
    with BreakoutGame(make_breakout_env(), seed=0) as game:
        observation = game.reset()
        assert observation.shape == (128,)
        assert observation.dtype == np.uint8
        assert game.needs_fire
        result = game.press_fire()
        assert not result.life_lost
        assert not game.needs_fire
