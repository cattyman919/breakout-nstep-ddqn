"""The random-policy baseline, with a scripted environment."""

import pytest
from conftest import ScriptedEnv, ScriptedStep

from scripts.measure_random import MeasureConfig, measure_random_returns
from src.exceptions import ConfigurationError


def scripted_game() -> ScriptedEnv:
    return ScriptedEnv(
        [
            ScriptedStep(reward=1.0),  # press: its raw reward counts
            ScriptedStep(reward=7.0, lives=4),
            ScriptedStep(),  # press
            ScriptedStep(reward=4.0, lives=0, terminated=True),
        ]
    )


def test_returns_are_raw_and_span_the_full_game() -> None:
    returns = measure_random_returns(MeasureConfig(games=3), env_factory=scripted_game)
    assert returns == [12.0, 12.0, 12.0]


def test_presses_are_made_after_reset_and_lost_life() -> None:
    envs: list[ScriptedEnv] = []

    def make() -> ScriptedEnv:
        envs.append(scripted_game())
        return envs[-1]

    measure_random_returns(MeasureConfig(games=1), env_factory=make)
    assert envs[0].actions[0] == 1
    assert envs[0].actions[2] == 1
    assert envs[0].closed


def test_truncated_game_ends_the_measurement_of_that_game() -> None:
    def make() -> ScriptedEnv:
        return ScriptedEnv([ScriptedStep(), ScriptedStep(reward=1.0, truncated=True)])

    assert measure_random_returns(MeasureConfig(games=1), env_factory=make) == [1.0]


@pytest.mark.parametrize("overrides", [{"games": 0}, {"seed": -1}])
def test_invalid_config_raises(overrides: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError):
        MeasureConfig(**overrides)
