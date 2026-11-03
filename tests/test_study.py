"""Planning, parallel execution and option passing for the study command."""

import csv
import threading
import time
from pathlib import Path

import pytest
import tyro

from scripts.study import (
    RunTiming,
    StudyConfig,
    StudyResult,
    default_train_config,
    format_duration,
    parse_args,
    plan_runs,
    run_study,
    timing_summary,
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


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (42.4, "42s"), (185, "3m 05s"), (3599.6, "1h 00m"), (4440, "1h 14m")],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


def test_timing_summary_totals_per_variant_and_overall() -> None:
    result = StudyResult(
        skipped=["dqn_n1_seed3"],
        finished=[],
        failed=[],
        timings=[
            RunTiming("dqn_n1", 1, 200.0, finished=True),
            RunTiming("dqn_n1", 2, 220.0, finished=True),
            RunTiming("ddqn_n3", 1, 180.0, finished=True),
            RunTiming("ddqn_n3", 2, 999.0, finished=False),
        ],
        wall_seconds=300.0,
    )

    lines = timing_summary(result, ["dqn_n1", "ddqn_n1", "ddqn_n3"])

    assert lines[2].split() == ["dqn_n1", "2", "7m", "00s", "3m", "30s"]
    assert lines[3].split() == ["ddqn_n3", "1", "3m", "00s", "3m", "00s"]
    assert lines[4].split() == ["all", "3", "10m", "00s", "3m", "20s"]
    assert lines[5] == "Wall-clock time: 5m 00s, for 10m 00s of training"
    assert "skipped" in lines[6]
    assert not any("ddqn_n1 " in line for line in lines)


def test_timing_summary_is_empty_without_finished_runs() -> None:
    result = StudyResult([], [], ["a"], [RunTiming("dqn_n1", 1, 5.0, False)], 5.0)
    assert timing_summary(result, ["dqn_n1"]) == []


def test_run_study_measures_each_run_and_records_it(tmp_path: Path) -> None:
    ticks = iter(range(0, 1000, 10))

    def fake_train(config: TrainConfig) -> int:
        return 1 if config.seed == 2 else 0

    messages: list[str] = []
    result = run_study(
        StudyConfig(variants=("dqn_n1",), seeds=(1, 2)),
        template_in(tmp_path),
        fake_train,
        report=messages.append,
        clock=lambda: float(next(ticks)),
    )

    assert sorted(timing.seconds for timing in result.timings) == [10.0, 10.0]
    assert result.wall_seconds == 50.0
    assert any(message.endswith("Finished dqn_n1_seed1 in 10s") for message in messages)
    assert any("FAILED   dqn_n1_seed2 after 10s" in message for message in messages)
    with (tmp_path / "run_logs" / "timings.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["run_name"], row["seconds"], row["result"]) for row in rows] == [
        ("dqn_n1_seed1", "10.0", "finished"),
        ("dqn_n1_seed2", "10.0", "failed"),
    ]


def test_timings_file_accumulates_across_invocations(tmp_path: Path) -> None:
    study = StudyConfig(variants=("dqn_n1",), seeds=(1,))
    run_study(study, template_in(tmp_path), lambda config: 1, report=lambda _: None)
    run_study(study, template_in(tmp_path), lambda config: 0, report=lambda _: None)
    lines = (tmp_path / "run_logs" / "timings.csv").read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("ended_at,")
