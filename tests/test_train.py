"""Training loop behavior, with a scripted environment and W&B offline mode."""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import ScriptedEnv, ScriptedStep

import scripts.train as train_module
from scripts.train import TrainConfig, resolve_device, run, save_checkpoint, train
from src.buffers import ReplayBuffer, Transition
from src.exceptions import ConfigurationError, TrainingError
from src.utils.logger import DualLogger, make_structured_logger


def short_config(tmp_path: Path, **overrides: Any) -> TrainConfig:
    settings: dict[str, Any] = {
        "variant": "ddqn_n3",
        "seed": 1,
        "device": "cpu",
        "wandb_mode": "offline",
        "log_dir": tmp_path,
        "total_timesteps": 120,
        "learning_starts": 20,
        "batch_size": 8,
        "buffer_capacity": 200,
        "log_interval": 5,
        "eval_interval": 50,
        "eval_games": 1,
    }
    return TrainConfig(**{**settings, **overrides})


def game_script() -> list[ScriptedStep]:
    """A press, then a life lost after 3 decisions, then a press and game over after 3 more."""
    return [
        ScriptedStep(),
        ScriptedStep(),
        ScriptedStep(reward=7.0),
        ScriptedStep(lives=4),
        ScriptedStep(),
        ScriptedStep(),
        ScriptedStep(reward=4.0),
        ScriptedStep(lives=0, terminated=True),
    ]


def factory(envs: list[ScriptedEnv]) -> Callable[[], ScriptedEnv]:
    """Build scripted environments, remembering each: training first, then evaluation."""

    def make() -> ScriptedEnv:
        env = ScriptedEnv(game_script())
        envs.append(env)
        return env

    return make


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_scripted(config: TrainConfig, envs: list[ScriptedEnv]) -> list[dict[str, str]]:
    log = make_structured_logger(config.run_name, config.seed)
    with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=config.log_dir) as metrics:
        train(config, torch.device("cpu"), metrics, log, env_factory=factory(envs))
        path = metrics.csv_path
    return read_rows(path)


@pytest.fixture(name="stored")
def fixture_stored(monkeypatch: pytest.MonkeyPatch) -> list[Transition]:
    """Record every transition added to replay."""
    stored: list[Transition] = []
    original = ReplayBuffer.add

    def add(self: ReplayBuffer, transition: Transition) -> None:
        stored.append(transition)
        original(self, transition)

    monkeypatch.setattr(ReplayBuffer, "add", add)
    return stored


def test_update_count_is_exact_with_presses_counted(tmp_path: Path) -> None:
    config = short_config(tmp_path, total_timesteps=1020, learning_starts=20, log_interval=25)
    rows = run_scripted(config, [])
    # (1020 - 20) / 4 = 250 updates, in groups of 25, each logged at the group's last update.
    train_rows = [row for row in rows if row["td_loss"]]
    assert len(train_rows) == 10
    assert [int(row["step"]) for row in train_rows] == [20 + 100 * i for i in range(1, 11)]


def test_partial_metric_group_is_dropped(tmp_path: Path) -> None:
    config = short_config(tmp_path, total_timesteps=60, learning_starts=20, log_interval=4)
    rows = run_scripted(config, [])
    # 10 updates give 2 full groups of 4; the last 2 updates are not logged.
    assert len([row for row in rows if row["td_loss"]]) == 2


def test_evaluation_fires_on_exact_multiples_even_on_a_press(tmp_path: Path) -> None:
    # Step 1 of every 8-step game is a press, and step 41 is the first press of game 6.
    config = short_config(tmp_path, total_timesteps=123, eval_interval=41)
    envs: list[ScriptedEnv] = []
    rows = run_scripted(config, envs)
    assert [int(row["step"]) for row in rows if row["eval_predicted_q"]] == [41, 82, 123]
    assert envs[0].actions[40] == 1


def test_presses_are_never_stored_and_no_step_is_lost(
    tmp_path: Path, stored: list[Transition]
) -> None:
    # Two 8-step games: each has 2 presses and 6 decisions.
    config = short_config(
        tmp_path,
        variant="ddqn_n1",
        total_timesteps=16,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    envs: list[ScriptedEnv] = []
    run_scripted(config, envs)

    assert len(stored) == 12
    # Observations count up from 1 at reset. Presses take steps 1 and 5 of each game, so
    # those steps never appear as a transition's first state.
    assert [int(t.state[0]) for t in stored[:6]] == [2, 3, 4, 6, 7, 8]
    assert envs[0].actions[0] == 1
    assert envs[0].actions[4] == 1


def test_training_targets_clip_while_logged_returns_stay_raw(
    tmp_path: Path, stored: list[Transition]
) -> None:
    config = short_config(
        tmp_path,
        variant="ddqn_n1",
        total_timesteps=8,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    rows = run_scripted(config, [])

    assert sorted({t.n_step_return for t in stored}) == [0.0, 1.0]
    episode = next(row for row in rows if row["episodic_return"])
    assert float(episode["episodic_return"]) == 11.0
    assert int(episode["episode_length"]) == 8


def test_life_loss_ends_the_target_without_reset(tmp_path: Path, stored: list[Transition]) -> None:
    config = short_config(
        tmp_path,
        variant="ddqn_n1",
        total_timesteps=8,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    envs: list[ScriptedEnv] = []
    run_scripted(config, envs)

    assert [t.done for t in stored] == [False, False, True, False, False, True]
    assert len(envs[0].reset_seeds) == 2  # the first game plus the reset after game over


def test_truncation_flushes_with_the_real_final_observation(
    tmp_path: Path, stored: list[Transition]
) -> None:
    script = [ScriptedStep(), ScriptedStep(), ScriptedStep(truncated=True)]
    config = short_config(
        tmp_path,
        variant="ddqn_n3",
        total_timesteps=3,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    log = make_structured_logger(config.run_name, config.seed)
    with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=tmp_path) as metrics:
        train(
            config,
            torch.device("cpu"),
            metrics,
            log,
            env_factory=lambda: ScriptedEnv(script),
        )
    assert [(t.span, t.done) for t in stored] == [(2, False), (1, False)]
    assert {int(t.next_state[0]) for t in stored} == {4}


def test_n_step_variant_stores_n_step_spans(tmp_path: Path, stored: list[Transition]) -> None:
    config = short_config(
        tmp_path,
        variant="ddqn_n3",
        total_timesteps=8,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    run_scripted(config, [])
    assert [t.span for t in stored] == [3, 2, 1, 3, 2, 1]


def test_evaluation_does_not_change_the_course_of_training(tmp_path: Path) -> None:
    def final_weights(directory: Path, eval_interval: int) -> list[torch.Tensor]:
        config = short_config(directory, eval_interval=eval_interval)
        log = make_structured_logger(config.run_name, config.seed)
        with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=directory) as metrics:
            agent = train(config, torch.device("cpu"), metrics, log, env_factory=factory([]))
        return [parameter.detach().clone() for parameter in agent.online.parameters()]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_eval = final_weights(tmp_path / "a", 10)
    without_eval = final_weights(tmp_path / "b", 10_000)
    for first, second in zip(with_eval, without_eval, strict=True):
        assert torch.equal(first, second)


def test_runs_are_reproducible_on_cpu(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = run_scripted(short_config(tmp_path / "a"), [])
    second = run_scripted(short_config(tmp_path / "b"), [])
    assert first == second


def test_environments_are_seeded_separately_on_first_reset_only(tmp_path: Path) -> None:
    envs: list[ScriptedEnv] = []
    run_scripted(short_config(tmp_path, seed=3, eval_interval=8), envs)
    training, evaluation = envs
    assert training.reset_seeds[0] == 3
    assert evaluation.reset_seeds[0] == 10_003
    assert set(training.reset_seeds[1:]) == {None}
    assert set(evaluation.reset_seeds[1:]) == {None}


def test_failure_closes_envs_and_logger_and_writes_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(self: Any, batch: Any, step: int) -> None:
        raise TrainingError(f"non-finite loss nan at step {step}")

    monkeypatch.setattr(train_module.DQNAgent, "update", explode)
    exit_codes: list[int] = []
    original_close = DualLogger.close

    def close(self: DualLogger, exit_code: int = 0) -> None:
        exit_codes.append(exit_code)
        original_close(self, exit_code)

    monkeypatch.setattr(DualLogger, "close", close)

    envs: list[ScriptedEnv] = []
    config = short_config(tmp_path)
    with pytest.raises(TrainingError, match="step 24"):
        run(config, env_factory=factory(envs))

    assert len(envs) == 2
    assert all(env.closed for env in envs)
    assert exit_codes[0] == 1
    assert (tmp_path / "csv" / f"{config.run_name}_metrics.csv").exists()
    assert not (tmp_path / "checkpoints" / f"{config.run_name}.pt").exists()
    # W&B prints its own plain text to stderr too, so pick out the diagnostic entry.
    lines = capsys.readouterr().err.splitlines()
    entry = json.loads(next(line for line in lines if '"run_diverged"' in line))
    assert entry["step"] == 24
    assert "Traceback" in entry["exception"]


def test_press_that_ends_the_game_is_a_normal_episode_end(
    tmp_path: Path, stored: list[Transition]
) -> None:
    config = short_config(
        tmp_path,
        variant="ddqn_n1",
        total_timesteps=3,
        learning_starts=8,
        batch_size=4,
        eval_interval=1000,
    )
    script = [ScriptedStep(reward=2.0, lives=0, terminated=True)]
    log = make_structured_logger(config.run_name, config.seed)
    with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=tmp_path) as metrics:
        train(config, torch.device("cpu"), metrics, log, env_factory=lambda: ScriptedEnv(script))
        path = metrics.csv_path

    assert stored == []
    episodes = [row for row in read_rows(path) if row["episodic_return"]]
    assert [float(row["episodic_return"]) for row in episodes] == [2.0, 2.0, 2.0]
    assert [int(row["episode_length"]) for row in episodes] == [1, 1, 1]


def test_snapshots_cover_the_start_and_each_interval(tmp_path: Path) -> None:
    config = short_config(tmp_path, snapshot_interval=50)
    run_scripted(config, [])

    directory = tmp_path / "checkpoints" / config.run_name
    assert sorted(path.name for path in directory.iterdir()) == [
        "step_000000.pt",
        "step_000050.pt",
        "step_000100.pt",
    ]
    snapshot = torch.load(directory / "step_000050.pt")
    assert snapshot["step"] == 50
    assert "layers.0.weight" in snapshot["online_state_dict"]


def test_snapshots_hold_the_weights_of_their_own_step(tmp_path: Path) -> None:
    config = short_config(tmp_path, snapshot_interval=50)
    run_scripted(config, [])

    directory = tmp_path / "checkpoints" / config.run_name
    start = torch.load(directory / "step_000000.pt")["online_state_dict"]
    later = torch.load(directory / "step_000100.pt")["online_state_dict"]
    assert not torch.equal(start["layers.0.weight"], later["layers.0.weight"])


def test_only_selected_seeds_save_snapshots(tmp_path: Path) -> None:
    config = short_config(tmp_path, seed=2, snapshot_seeds=(1,))
    run_scripted(config, [])
    assert not (tmp_path / "checkpoints").exists()


def test_snapshots_do_not_change_the_course_of_training(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_snapshots = short_config(tmp_path / "a", snapshot_seeds=(1,), snapshot_interval=10)
    without_snapshots = short_config(tmp_path / "b", snapshot_seeds=())
    assert run_scripted(with_snapshots, []) == run_scripted(without_snapshots, [])


def test_integration_short_run_on_the_real_game(tmp_path: Path) -> None:
    config = short_config(tmp_path, total_timesteps=600, learning_starts=100, log_interval=25)
    run(config)

    rows = read_rows(tmp_path / "csv" / f"{config.run_name}_metrics.csv")
    assert len([row for row in rows if row["td_loss"]]) == 5
    assert any(row["episodic_return"] for row in rows)
    assert (tmp_path / "tensorboard" / config.run_name).exists()
    checkpoint = torch.load(tmp_path / "checkpoints" / f"{config.run_name}.pt")
    assert checkpoint["config"]["variant"] == "ddqn_n3"
    assert "layers.0.weight" in checkpoint["online_state_dict"]


def test_rerun_refuses_to_overwrite_results(tmp_path: Path) -> None:
    config = short_config(tmp_path, total_timesteps=10, learning_starts=8, eval_interval=1000)
    run(config)
    with pytest.raises(FileExistsError):
        run(config)


def test_checkpoint_path_uses_the_run_name(tmp_path: Path) -> None:
    config = short_config(tmp_path)
    log = make_structured_logger(config.run_name, config.seed)
    with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=tmp_path) as metrics:
        agent = train(config, torch.device("cpu"), metrics, log, env_factory=factory([]))
    assert save_checkpoint(agent, config) == tmp_path / "checkpoints" / "ddqn_n3_seed1.pt"


@pytest.mark.parametrize(
    ("variant", "double_q", "n_step"),
    [("dqn_n1", False, 1), ("ddqn_n1", True, 1), ("ddqn_n3", True, 3), ("ddqn_n5", True, 5)],
)
def test_variant_sets_the_algorithm_and_n(variant: str, double_q: bool, n_step: int) -> None:
    config = TrainConfig(variant=variant)  # type: ignore[arg-type]
    assert config.double_q is double_q
    assert config.n_step == n_step
    assert config.agent_config.double_q is double_q


def test_epsilon_schedule_matches_the_protocol() -> None:
    config = TrainConfig()
    assert config.epsilon_at(0) == pytest.approx(1.0)
    assert config.epsilon_at(10_000) == pytest.approx(0.81)
    assert config.epsilon_at(50_000) == pytest.approx(0.05)
    assert config.epsilon_at(400_000) == pytest.approx(0.05)


def test_default_settings_give_the_documented_schedule_counts() -> None:
    config = TrainConfig()
    steps = np.arange(1, config.total_timesteps + 1)
    updates = (steps > config.learning_starts) & (steps % config.train_frequency == 0)
    assert updates.sum() == 122_500
    assert updates.sum() // config.log_interval == 1_225
    assert (steps % config.eval_interval == 0).sum() == 20


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_timesteps": 0},
        {"seed": -1},
        {"gamma": 0.0},
        {"tau": 1.5},
        {"batch_size": 0},
        {"epsilon_end": 1.5},
        {"epsilon_start": 0.01},
        {"epsilon_decay_fraction": 0.0},
        {"learning_starts": 1},
        {"eval_games": 0},
        {"log_interval": 0},
        {"snapshot_interval": 0},
        {"terminal_on_life_loss": False},
    ],
)
def test_invalid_config_raises(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        TrainConfig(**overrides)


def test_resolve_device_returns_cpu_when_requested() -> None:
    log = make_structured_logger("run", 0)
    assert resolve_device("cpu", log) == torch.device("cpu")


def test_resolve_device_falls_back_with_a_structured_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("mps", make_structured_logger("run", 0)) == torch.device("cpu")
    assert "mps_unavailable" in capsys.readouterr().err


def test_each_run_length_defaults_to_its_own_folder(tmp_path: Path) -> None:
    assert TrainConfig().output_dir == Path("logs/steps_500000")
    assert TrainConfig(total_timesteps=12_000_000).output_dir == Path("logs/steps_12000000")
    assert TrainConfig(log_dir=tmp_path).output_dir == tmp_path


def test_checkpoints_and_hyperparams_use_the_resolved_folder(tmp_path: Path) -> None:
    config = short_config(tmp_path)
    log = make_structured_logger(config.run_name, config.seed)
    with DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=tmp_path) as metrics:
        agent = train(config, torch.device("cpu"), metrics, log, env_factory=factory([]))
    path = save_checkpoint(agent, config)
    assert torch.load(path)["config"]["log_dir"] == str(tmp_path)


def test_progress_file_tracks_steps_and_ends_at_the_total(tmp_path: Path) -> None:
    from scripts.train import RunProgress

    path = tmp_path / "run.progress"
    progress = RunProgress(25_000, "run", path, show_bar=False)
    assert path.read_text() == "0\n"
    for step in range(1, 25_001):
        progress.update(step, None, 1.0)
        if step == 10_000:
            assert path.read_text() == "10000\n"
        if step == 19_999:
            assert path.read_text() == "10000\n"
    assert path.read_text() == "25000\n"
    assert not path.with_suffix(".progress.tmp").exists()


def test_progress_bar_reaches_the_total_with_return_and_epsilon(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.train import RunProgress

    with RunProgress(2_500, "dqn_n1_seed1", None, show_bar=True) as progress:
        for step in range(1, 2_501):
            progress.update(step, 12.345 if step > 1_000 else None, 0.05)
    err = capsys.readouterr().err
    assert "dqn_n1_seed1" in err
    assert "2.50k/2.50k" in err
    assert "return 12.3, eps 0.05" in err


def test_training_reports_progress_without_changing_results(tmp_path: Path) -> None:
    from scripts.train import RunProgress

    config = short_config(tmp_path / "a")
    path = tmp_path / "a.progress"
    log = make_structured_logger(config.run_name, config.seed)
    with (
        DualLogger(config.run_name, {}, wandb_mode="offline", log_dir=tmp_path / "a") as metrics,
        RunProgress(config.total_timesteps, "run", path, show_bar=False) as progress,
    ):
        train(config, torch.device("cpu"), metrics, log, factory([]), progress)
        with_progress = metrics.csv_path
    assert path.read_text() == f"{config.total_timesteps}\n"
    without_progress = run_scripted(short_config(tmp_path / "b"), [])
    assert read_rows(with_progress) == without_progress
