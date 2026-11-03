"""Train the study's runs, several at a time if asked.

Run from the project root::

    uv run python -m scripts.study --jobs 4

Each run is a separate ``scripts.train`` process using one CPU core, so a run's results are the
same whether it trains alone or beside others; only the wall-clock time changes. Runs whose final
model already exists are skipped, so the command can be rerun after an interruption. Each run's
output goes to ``{log_dir}/run_logs/{run_name}.log``, where ``log_dir`` defaults to
``logs/steps_<total_timesteps>``.

Each finished line shows how long the run took. At the end, a table gives the training time per
variant and in total, and each run's time is appended to ``{log_dir}/run_logs/timings.csv`` as soon
as it ends, so the record survives an interruption.

Besides the options below, the command accepts every ``scripts.train`` option, such as
``--total-timesteps 2000000``, and applies it to every run. ``--variant`` and ``--seed`` are
chosen with ``--variants`` and ``--seeds`` instead.
"""

import csv
import dataclasses
import os
import subprocess
import sys
import threading
import time
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
class RunTiming:
    """How long one run took to train, and whether it finished."""

    variant: str
    seed: int
    seconds: float
    finished: bool

    @property
    def run_name(self) -> str:
        """The run's name, as used in file names."""
        return f"{self.variant}_seed{self.seed}"


@dataclass(frozen=True)
class StudyResult:
    """What happened to each run, and how long the whole study took."""

    skipped: list[str]
    finished: list[str]
    failed: list[str]
    timings: list[RunTiming]
    wall_seconds: float


def format_duration(seconds: float) -> str:
    """Format a duration as ``42s``, ``3m 05s`` or ``1h 14m``."""
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def timing_summary(result: StudyResult, variants: Sequence[str]) -> list[str]:
    """Return a table of training time per variant and in total, for finished runs.

    Run time is each run's own time from start to end. Wall-clock time is how long the whole
    command took, which is shorter when runs train in parallel.
    """
    finished = [timing for timing in result.timings if timing.finished]
    if not finished:
        return []
    lines = [
        "Training time for the runs trained now:",
        f"  {'variant':<10}{'runs':>5}{'total':>11}{'per run':>11}",
    ]
    for variant in variants:
        runs = [timing.seconds for timing in finished if timing.variant == variant]
        if runs:
            lines.append(
                f"  {variant:<10}{len(runs):>5}{format_duration(sum(runs)):>11}"
                f"{format_duration(sum(runs) / len(runs)):>11}"
            )
    run_total = sum(timing.seconds for timing in finished)
    lines.append(
        f"  {'all':<10}{len(finished):>5}{format_duration(run_total):>11}"
        f"{format_duration(run_total / len(finished)):>11}"
    )
    lines.append(
        f"Wall-clock time: {format_duration(result.wall_seconds)}, "
        f"for {format_duration(run_total)} of training"
    )
    if result.skipped:
        lines.append(f"Runs skipped as already finished aren't included: {len(result.skipped)}.")
    return lines


def append_timing(path: Path, timing: RunTiming) -> None:
    """Add one run's timing to the CSV record, writing the header for a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["ended_at", "run_name", "variant", "seed", "seconds", "result"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                timing.run_name,
                timing.variant,
                timing.seed,
                f"{timing.seconds:.1f}",
                "finished" if timing.finished else "failed",
            ]
        )


def default_train_config() -> TrainConfig:
    """Training settings used unless overridden: the study's, on CPU and offline."""
    return TrainConfig(device="cpu", wandb_mode="offline")


def to_cli_args(config: TrainConfig) -> list[str]:
    """Turn a training config into the ``scripts.train`` options that recreate it exactly."""
    args: list[str] = []
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        flag = "--" + field.name.replace("_", "-")
        if value is None:
            continue
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
    return config.output_dir / "checkpoints" / f"{config.run_name}.pt"


def run_log_path(config: TrainConfig) -> Path:
    """Where a run's console output is written."""
    return config.output_dir / "run_logs" / f"{config.run_name}.log"


def timings_path(config: TrainConfig) -> Path:
    """Where each run's training time is recorded."""
    return config.output_dir / "run_logs" / "timings.csv"


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
    clock: Callable[[], float] = time.monotonic,
) -> StudyResult:
    """Train every unfinished run, ``study.jobs`` at a time, and report each start and end."""
    study_start = clock()
    skipped, pending = plan_runs(study, template)
    for name in skipped:
        report(f"Skipping {name}: already finished")
    report(f"Training {len(pending)} runs, {study.jobs} at a time.")

    lock = threading.Lock()

    def say(message: str) -> None:
        with lock:
            report(f"{datetime.now():%H:%M:%S} {message}")

    def train_and_report(config: TrainConfig) -> RunTiming:
        say(f"Started  {config.run_name}")
        start = clock()
        code = train_one(config)
        timing = RunTiming(config.variant, config.seed, clock() - start, finished=code == 0)
        took = format_duration(timing.seconds)
        if timing.finished:
            say(f"Finished {config.run_name} in {took}")
        else:
            say(f"FAILED   {config.run_name} after {took}: see {run_log_path(config)}")
        return timing

    timings: list[RunTiming] = []
    with ThreadPoolExecutor(max_workers=study.jobs) as executor:
        futures = [executor.submit(train_and_report, config) for config in pending]
        try:
            for future in as_completed(futures):
                timing = future.result()
                timings.append(timing)
                append_timing(timings_path(template), timing)
        except KeyboardInterrupt:
            # The terminal also interrupts the training processes; don't start new ones.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    return StudyResult(
        skipped=skipped,
        finished=sorted(timing.run_name for timing in timings if timing.finished),
        failed=sorted(timing.run_name for timing in timings if not timing.finished),
        timings=timings,
        wall_seconds=clock() - study_start,
    )


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
    summary = timing_summary(result, study.variants)
    if summary:
        print()
        print("\n".join(summary))
        print(f"Each run's time is recorded in {timings_path(template)}.")
        print()
    log.info(
        "study_finished",
        trained=len(result.timings),
        wall_seconds=round(result.wall_seconds, 1),
        run_seconds={timing.run_name: round(timing.seconds, 1) for timing in result.timings},
    )
    print(f"{len(result.skipped) + len(result.finished)} of {total} runs finished.")
    if result.failed:
        log.error("runs_failed", runs=result.failed)
        print("These runs did not finish: " + ", ".join(result.failed))
        print("Check their logs in " + str(template.output_dir / "run_logs") + "/.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
