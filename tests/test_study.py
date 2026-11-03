"""Planning, parallel execution and option passing for the study command."""

import threading
import time
from pathlib import Path

import pytest
import tyro

from scripts.study import (
    StudyConfig,
    default_train_config,
    parse_args,
    plan_runs,
    run_study,
    to_cli_args,
)
from scripts.train import TrainConfig
from src.exceptions import ConfigurationError


def template_in(tmp_path: Path) -> TrainConfig:
    return TrainConfig(device="cpu", wandb_mode="offline", log_dir=tmp_path)


def finish(config: TrainConfig) -> None:
    path = config.log_dir / "checkpoints" / f"{config.run_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")


def test_cli_args_recreate_the_config_exactly(tmp_path: Path) -> None:
    config = TrainConfig(
        variant="dqn_n1",
        seed=7,
        device="cpu",
        wandb_mode="offline",
        log_dir=tmp_path / "x",
        total_timesteps=1234,
        learning_rate=3e-5,
        clip_rewards=False,
        auto_fire=False,
        terminal_on_life_loss=False,
        snapshot_seeds=(),
    )
    assert tyro.cli(TrainConfig, args=to_cli_args(config)) == config
    defaults = default_train_config()
    assert tyro.cli(TrainConfig, args=to_cli_args(defaults)) == defaults


def test_plan_skips_runs_with_a_final_model(tmp_path: Path) -> None:
    template = template_in(tmp_path)
    finish(TrainConfig(variant="dqn_n1", seed=2, log_dir=tmp_path))
    study = StudyConfig(variants=("dqn_n1", "ddqn_n3"), seeds=(1, 2))

    finished, pending = plan_runs(study, template)

    assert finished == ["dqn_n1_seed2"]
    assert [run.run_name for run in pending] == ["dqn_n1_seed1", "ddqn_n3_seed1", "ddqn_n3_seed2"]
    assert all(run.log_dir == tmp_path and run.device == "cpu" for run in pending)


def test_runs_execute_up_to_jobs_at_a_time(tmp_path: Path) -> None:
    running = 0
    most = 0
    lock = threading.Lock()

    def fake_train(config: TrainConfig) -> int:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return 0

    study = StudyConfig(seeds=(1, 2, 3), jobs=3)
    result = run_study(study, template_in(tmp_path), fake_train, report=lambda _: None)

    assert len(result.finished) == 12
    assert most == 3


def test_one_job_runs_one_at_a_time(tmp_path: Path) -> None:
    running = 0
    most = 0

    def fake_train(config: TrainConfig) -> int:
        nonlocal running, most
        running += 1
        most = max(most, running)
        time.sleep(0.01)
        running -= 1
        return 0

    run_study(StudyConfig(seeds=(1, 2)), template_in(tmp_path), fake_train, report=lambda _: None)
    assert most == 1


def test_failures_are_collected_and_others_continue(tmp_path: Path) -> None:
    messages: list[str] = []

    def fake_train(config: TrainConfig) -> int:
        return 1 if config.run_name == "ddqn_n1_seed1" else 0

    study = StudyConfig(variants=("dqn_n1", "ddqn_n1"), seeds=(1, 2), jobs=2)
    result = run_study(study, template_in(tmp_path), fake_train, report=messages.append)

    assert result.failed == ["ddqn_n1_seed1"]
    assert result.finished == ["ddqn_n1_seed2", "dqn_n1_seed1", "dqn_n1_seed2"]
    failure = next(message for message in messages if "FAILED" in message)
    assert str(tmp_path / "run_logs" / "ddqn_n1_seed1.log") in failure


def test_finished_runs_are_reported_as_skipped(tmp_path: Path) -> None:
    finish(TrainConfig(variant="dqn_n1", seed=1, log_dir=tmp_path))
    messages: list[str] = []
    result = run_study(
        StudyConfig(variants=("dqn_n1",), seeds=(1,)),
        template_in(tmp_path),
        lambda config: pytest.fail("a finished run was trained again"),
        report=messages.append,
    )
    assert result.skipped == ["dqn_n1_seed1"]
    assert "Skipping dqn_n1_seed1: already finished" in messages


def test_training_options_pass_through_to_every_run() -> None:
    study, template = parse_args(
        ["--jobs", "4", "--seeds", "1", "2", "--total-timesteps", "2000000", "--tau", "0.01"]
    )
    assert study.jobs == 4
    assert study.seeds == (1, 2)
    assert template.total_timesteps == 2_000_000
    assert template.tau == 0.01
    assert template.device == "cpu"
    assert template.wandb_mode == "offline"


@pytest.mark.parametrize("blocked", [["--variant", "dqn_n1"], ["--seed", "3"], ["--seed=3"]])
def test_single_run_options_are_rejected(blocked: list[str]) -> None:
    with pytest.raises(ConfigurationError, match="to choose runs"):
        parse_args(blocked)


def test_invalid_training_options_are_rejected_before_any_run() -> None:
    with pytest.raises(ConfigurationError, match="tau"):
        parse_args(["--tau", "2"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"jobs": 0},
        {"variants": ()},
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"variants": ("dqn_n1",) * 2},
    ],
)
def test_invalid_study_config_raises(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        StudyConfig(**overrides)  # type: ignore[arg-type]


def test_real_runs_train_in_parallel_with_their_own_logs(tmp_path: Path) -> None:
    template = TrainConfig(
        device="cpu",
        wandb_mode="offline",
        log_dir=tmp_path,
        total_timesteps=200,
        learning_starts=100,
        eval_interval=100,
        eval_games=1,
        snapshot_seeds=(),
    )
    result = run_study(
        StudyConfig(variants=("dqn_n1", "ddqn_n3"), seeds=(1,), jobs=2),
        template,
        report=lambda _: None,
    )

    assert result.finished == ["ddqn_n3_seed1", "dqn_n1_seed1"]
    for run in ("dqn_n1_seed1", "ddqn_n3_seed1"):
        assert (tmp_path / "checkpoints" / f"{run}.pt").exists()
        assert '"run_finished"' in (tmp_path / "run_logs" / f"{run}.log").read_text()
