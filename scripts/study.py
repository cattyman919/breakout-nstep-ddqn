"""Train the study's runs, several at a time if asked.

Run from the project root::

    uv run python -m scripts.study --jobs 4

Each run is a separate ``scripts.train`` process using one CPU core, so a run's results are the
same whether it trains alone or beside others; only the wall-clock time changes. Runs whose final
model already exists are skipped, so the command can be rerun after an interruption. Each run's
output goes to ``{log_dir}/run_logs/{run_name}.log``.

Besides the options below, the command accepts every ``scripts.train`` option, such as
``--total-timesteps 2000000``, and applies it to every run. ``--variant`` and ``--seed`` are
chosen with ``--variants`` and ``--seeds`` instead.
"""

import dataclasses
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from scripts.train import TrainConfig
from src.exceptions import ConfigurationError
from src.utils.logger import make_structured_logger


@dataclass(frozen=True)
class StudyConfig:
    """Which runs to train and how many at once. Training options are passed through."""

    variants: tuple[Literal["dqn_n1", "ddqn_n1", "ddqn_n3", "ddqn_n5"], ...] = (
        "dqn_n1",
        "ddqn_n1",
        "ddqn_n3",
        "ddqn_n5",
    )
    """Variants to train."""
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    """Seeds to train for each variant."""
    jobs: int = 1
    """Runs to train at the same time. Each uses one CPU core."""

    def __post_init__(self) -> None:
        if self.jobs < 1:
            raise ConfigurationError(f"jobs must be at least 1, got {self.jobs}")
        if not self.variants:
            raise ConfigurationError("variants must name at least one variant")
        if not self.seeds:
            raise ConfigurationError("seeds must name at least one seed")
        if len(set(self.variants)) != len(self.variants):
            raise ConfigurationError(f"variants contains duplicates: {self.variants}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError(f"seeds contains duplicates: {self.seeds}")


@dataclass(frozen=True)
class StudyResult:
    """What happened to each run."""

    skipped: list[str]
    finished: list[str]
    failed: list[str]


def default_train_config() -> TrainConfig:
    """Training settings used unless overridden: the study's, on CPU and offline."""
    return TrainConfig(device="cpu", wandb_mode="offline")


def to_cli_args(config: TrainConfig) -> list[str]:
    """Turn a training config into the ``scripts.train`` options that recreate it exactly."""
    args: list[str] = []
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + field.name.replace("_", "-"))
        elif isinstance(value, tuple):
            args += [flag, *map(str, value)]
        else:
            args += [flag, str(value)]
    return args


def plan_runs(study: StudyConfig, template: TrainConfig) -> tuple[list[str], list[TrainConfig]]:
    """Split the study's runs into finished ones, by name, and configs still to train."""
    finished: list[str] = []
    pending: list[TrainConfig] = []
    for variant in study.variants:
        for seed in study.seeds:
            run = dataclasses.replace(template, variant=variant, seed=seed)
            if final_model_path(run).exists():
                finished.append(run.run_name)
            else:
                pending.append(run)
    return finished, pending


def final_model_path(config: TrainConfig) -> Path:
    """Where a run's final model is saved; it exists only if the run finished."""
    return config.log_dir / "checkpoints" / f"{config.run_name}.pt"


def run_log_path(config: TrainConfig) -> Path:
    """Where a run's console output is written."""
    return config.log_dir / "run_logs" / f"{config.run_name}.log"


def train_in_subprocess(config: TrainConfig) -> int:
    """Train one run in its own process and return its exit code."""
    log_path = run_log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.train", *to_cli_args(config)]
    with log_path.open("w", encoding="utf-8") as output:
        return subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=False
        ).returncode


def run_study(
    study: StudyConfig,
    template: TrainConfig,
    train_one: Callable[[TrainConfig], int] = train_in_subprocess,
    report: Callable[[str], None] = print,
) -> StudyResult:
    """Train every unfinished run, ``study.jobs`` at a time, and report each start and end."""
    skipped, pending = plan_runs(study, template)
    for name in skipped:
        report(f"Skipping {name}: already finished")
    report(f"Training {len(pending)} runs, {study.jobs} at a time.")

    lock = threading.Lock()

    def say(message: str) -> None:
        with lock:
            report(f"{datetime.now():%H:%M:%S} {message}")

    def train_and_report(config: TrainConfig) -> tuple[str, int]:
        say(f"Started  {config.run_name}")
        code = train_one(config)
        if code == 0:
            say(f"Finished {config.run_name}")
        else:
            say(f"FAILED   {config.run_name}: see {run_log_path(config)}")
        return config.run_name, code

    finished: list[str] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=study.jobs) as executor:
        futures = [executor.submit(train_and_report, config) for config in pending]
        try:
            for future in as_completed(futures):
                name, code = future.result()
                (finished if code == 0 else failed).append(name)
        except KeyboardInterrupt:
            # The terminal also interrupts the training processes; don't start new ones.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    return StudyResult(skipped, sorted(finished), sorted(failed))


def parse_args(argv: Sequence[str]) -> tuple[StudyConfig, TrainConfig]:
    """Read the study's own options, and pass the rest to the training config.

    Raises:
        ConfigurationError: If ``--variant`` or ``--seed`` is passed through.
    """
    study, train_args = tyro.cli(StudyConfig, args=list(argv), return_unknown_args=True)
    for blocked in ("--variant", "--seed"):
        if any(arg == blocked or arg.startswith(blocked + "=") for arg in train_args):
            raise ConfigurationError(f"use --{blocked[2:]}s to choose runs, not {blocked}")
    template = tyro.cli(TrainConfig, args=train_args, default=default_train_config())
    return study, template


def main() -> int:
    """Train the study and return a process exit code."""
    study, template = parse_args(sys.argv[1:])
    log = make_structured_logger("study", None)
    cores = os.cpu_count() or 1
    if study.jobs > cores:
        log.warning("more_jobs_than_cores", jobs=study.jobs, cores=cores)
    try:
        result = run_study(study, template)
    except KeyboardInterrupt:
        print("Interrupted. Runs that didn't finish need `make clean-run` before retrying.")
        return 130

    total = len(result.skipped) + len(result.finished) + len(result.failed)
    print(f"{len(result.skipped) + len(result.finished)} of {total} runs finished.")
    if result.failed:
        log.error("runs_failed", runs=result.failed)
        print("These runs did not finish: " + ", ".join(result.failed))
        print("Check their logs in " + str(template.log_dir / "run_logs") + "/.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
