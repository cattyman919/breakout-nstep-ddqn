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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import tyro
from tqdm import tqdm

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


def progress_path(config: TrainConfig) -> Path:
    """Where a run keeps its step count while it trains."""
    return config.output_dir / "run_logs" / f"{config.run_name}.progress"


def read_progress(config: TrainConfig) -> int:
    """Return how many steps a running run has reported, or 0 before its first report."""
    try:
        return int(progress_path(config).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return 0


@dataclass
class _Tally:
    """Steps and runs counted toward one progress bar."""

    total_steps: int = 0
    completed_steps: int = 0
    shown_steps: int = 0
    runs: int = 0
    finished_runs: int = 0


class StudyProgress:
    """Progress bars for the runs the study trains: one per variant, plus one for the whole study.

    Each running run's progress file is read every ``POLL_SECONDS``. A variant's bar covers all of
    its runs being trained, and the study bar covers every run, so each time estimate is for the
    whole set. Without ``show_bar`` it does nothing.
    """

    POLL_SECONDS: ClassVar[float] = 2.0
    STUDY_LABEL: ClassVar[str] = "study"

    def __init__(self, runs: Sequence[TrainConfig], *, show_bar: bool) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, TrainConfig] = {}
        self._tallies: dict[str, _Tally] = {}
        for run in runs:
            for key in (run.variant, self.STUDY_LABEL):
                tally = self._tallies.setdefault(key, _Tally())
                tally.total_steps += run.total_timesteps
                tally.runs += 1
        # The study's own tally goes last, so its bar sits at the bottom.
        study = self._tallies.pop(self.STUDY_LABEL, None)
        if study is not None:
            self._tallies[self.STUDY_LABEL] = study
        self._bars: dict[str, tqdm[Any]] = {}
        if show_bar and runs:
            width = max(len(key) for key in self._tallies)
            for position, (key, tally) in enumerate(self._tallies.items()):
                self._bars[key] = tqdm(
                    total=tally.total_steps,
                    desc=key.ljust(width),
                    unit="step",
                    unit_scale=True,
                    dynamic_ncols=True,
                    smoothing=0.05,
                    position=position,
                    leave=True,
                )

    @property
    def active(self) -> bool:
        """Whether bars are on screen, so messages must be printed above them."""
        return bool(self._bars)

    def started(self, config: TrainConfig) -> None:
        """Note that a run started."""
        with self._lock:
            self._running[config.run_name] = config

    def ended(self, config: TrainConfig, *, finished: bool) -> None:
        """Note that a run ended; a failed run's remaining steps leave the totals."""
        steps = read_progress(config)
        with self._lock:
            self._running.pop(config.run_name, None)
            for key in (config.variant, self.STUDY_LABEL):
                tally = self._tallies[key]
                if finished:
                    tally.completed_steps += config.total_timesteps
                    tally.finished_runs += 1
                else:
                    tally.completed_steps += steps
                    tally.total_steps -= config.total_timesteps - steps
                    bar = self._bars.get(key)
                    if bar is not None:
                        bar.total = tally.total_steps
        progress_path(config).unlink(missing_ok=True)

    def refresh(self) -> None:
        """Read each running run's progress file and redraw every bar."""
        if not self._bars:
            return
        with self._lock:
            running = {name: (run, read_progress(run)) for name, run in self._running.items()}
            for key, bar in self._bars.items():
                mine = [
                    (run, steps)
                    for run, steps in running.values()
                    if key in (run.variant, self.STUDY_LABEL)
                ]
                tally = self._tallies[key]
                done = tally.completed_steps + sum(steps for _, steps in mine)
                bar.update(max(done - tally.shown_steps, 0))
                tally.shown_steps = max(done, tally.shown_steps)
                bar.set_postfix_str(self._label(key, tally, mine), refresh=False)
                bar.refresh()

    def close(self) -> None:
        """Close every bar, leaving their last state on screen."""
        if self._bars:
            self.refresh()
            for bar in self._bars.values():
                bar.close()
            self._bars = {}

    def _label(self, key: str, tally: _Tally, running: Sequence[tuple[TrainConfig, int]]) -> str:
        """Say how many runs are done, and each running run's percentage."""
        done = f"{tally.finished_runs}/{tally.runs} done"
        if key == self.STUDY_LABEL:
            return f"{done}, {len(running)} running"
        seeds = " ".join(
            f"seed {run.seed}: {100 * steps // run.total_timesteps}%" for run, steps in running
        )
        return f"{done}, {seeds}" if seeds else done


def train_in_subprocess(config: TrainConfig) -> int:
    """Train one run in its own process and return its exit code."""
    log_path = run_log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(config, progress_file=progress_path(config))
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
    show_progress: bool = False,
) -> StudyResult:
    """Train every unfinished run, ``study.jobs`` at a time, and report each start and end.

    With ``show_progress``, a bar covering every run stays at the bottom of the terminal, and
    messages print above it.
    """
    study_start = clock()
    skipped, pending = plan_runs(study, template)
    progress = StudyProgress(pending, show_bar=show_progress)
    if progress.active:
        report = tqdm.write
    for name in skipped:
        report(f"Skipping {name}: already finished")
    report(f"Training {len(pending)} runs, {study.jobs} at a time.")

    lock = threading.Lock()

    def say(message: str) -> None:
        with lock:
            report(f"{datetime.now():%H:%M:%S} {message}")

    def train_and_report(config: TrainConfig) -> RunTiming:
        say(f"Started  {config.run_name}")
        progress.started(config)
        start = clock()
        code = train_one(config)
        timing = RunTiming(config.variant, config.seed, clock() - start, finished=code == 0)
        progress.ended(config, finished=timing.finished)
        took = format_duration(timing.seconds)
        if timing.finished:
            say(f"Finished {config.run_name} in {took}")
        else:
            say(f"FAILED   {config.run_name} after {took}: see {run_log_path(config)}")
        return timing

    timings: list[RunTiming] = []
    with ThreadPoolExecutor(max_workers=study.jobs) as executor:
        remaining: set[Future[RunTiming]] = {
            executor.submit(train_and_report, config) for config in pending
        }
        try:
            while remaining:
                done, remaining = wait(
                    remaining, timeout=StudyProgress.POLL_SECONDS, return_when=FIRST_COMPLETED
                )
                for future in done:
                    timing = future.result()
                    timings.append(timing)
                    append_timing(timings_path(template), timing)
                progress.refresh()
        except KeyboardInterrupt:
            # The terminal also interrupts the training processes; don't start new ones.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            progress.close()
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
        result = run_study(study, template, show_progress=sys.stderr.isatty())
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
