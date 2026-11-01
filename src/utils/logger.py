"""Dual-telemetry logging: TensorBoard, Weights & Biases and a raw CSV file.

Every metric goes to all three sinks. The CSV holds two kinds of row, told apart
by which columns are filled:

* episode rows: ``step, episode, episodic_return, episode_length, epsilon``
* train rows: ``step, mean_q_value, td_loss``

Columns that do not apply to a row are left empty.
"""

import csv
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, Self

import wandb
from torch.utils.tensorboard import SummaryWriter


class DualLogger:
    """Dispatch training metrics to TensorBoard, W&B and CSV at the same time.

    Args:
        run_name: Unique run identifier, used as a single path component.
        hyperparams: Run configuration stored in W&B and as TensorBoard text.
        wandb_mode: ``"online"`` syncs live; ``"offline"`` stores the run locally
            for a later ``wandb sync``.
        log_dir: Root directory for all logs.
        overwrite: Replace an existing CSV for ``run_name``. By default an existing
            file raises ``FileExistsError`` so a rerun cannot destroy results.
    """

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "step",
        "episode",
        "episodic_return",
        "episode_length",
        "mean_q_value",
        "td_loss",
        "epsilon",
    )
    ROLLING_WINDOW: ClassVar[int] = 100
    WANDB_PROJECT: ClassVar[str] = "breakout-nstep-ddqn"

    def __init__(
        self,
        run_name: str,
        hyperparams: Mapping[str, Any],
        *,
        wandb_mode: Literal["online", "offline"] = "online",
        log_dir: Path = Path("logs"),
        overwrite: bool = False,
    ) -> None:
        if not run_name or Path(run_name).name != run_name:
            raise ValueError(f"run_name must be a single path component, got {run_name!r}")

        csv_dir = log_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_dir / f"{run_name}_metrics.csv"
        self._csv_file = self.csv_path.open("w" if overwrite else "x", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS, restval="")
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._recent_returns: deque[float] = deque(maxlen=self.ROLLING_WINDOW)
        self._closed = False

        self._writer: SummaryWriter | None = None
        try:
            self._writer = SummaryWriter(log_dir=str(log_dir / "tensorboard" / run_name))
            self._writer.add_text("hyperparams", _markdown_table(hyperparams))
            self._run = wandb.init(
                project=self.WANDB_PROJECT,
                name=run_name,
                config=dict(hyperparams),
                mode=wandb_mode,
                dir=str(log_dir),
            )
        except BaseException:
            if self._writer is not None:
                self._writer.close()
            self._csv_file.close()
            raise

        # An explicit x-axis metric keeps W&B from dropping points that arrive with
        # a step lower than one already logged.
        self._run.define_metric("env_step")
        self._run.define_metric("*", step_metric="env_step")

    def log_episode(
        self,
        step: int,
        episode: int,
        return_val: float,
        length: int,
        epsilon: float,
    ) -> None:
        """Record one finished episode at environment step ``step``."""
        return_val = float(return_val)
        self._recent_returns.append(return_val)
        rolling_mean = sum(self._recent_returns) / len(self._recent_returns)

        self._dispatch(
            int(step),
            {
                "episode/return": return_val,
                "episode/return_rolling_100": rolling_mean,
                "episode/length": int(length),
                "episode/epsilon": float(epsilon),
            },
            {
                "step": int(step),
                "episode": int(episode),
                "episodic_return": return_val,
                "episode_length": int(length),
                "epsilon": float(epsilon),
            },
        )
        self._csv_file.flush()

    def log_train_step(self, step: int, td_loss: float, mean_q: float) -> None:
        """Record one gradient update at environment step ``step``."""
        self._dispatch(
            int(step),
            {"train/td_loss": float(td_loss), "train/mean_q_value": float(mean_q)},
            {"step": int(step), "td_loss": float(td_loss), "mean_q_value": float(mean_q)},
        )

    def close(self, exit_code: int = 0) -> None:
        """Flush and close every sink. Safe to call more than once.

        Args:
            exit_code: Passed to W&B; a nonzero value marks the run as failed.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._csv_file.close()
        finally:
            try:
                if self._writer is not None:
                    self._writer.flush()
                    self._writer.close()
            finally:
                self._run.finish(exit_code=exit_code)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exit_code=0 if exc_type is None else 1)

    def _dispatch(
        self,
        step: int,
        scalars: Mapping[str, float],
        csv_row: Mapping[str, float],
    ) -> None:
        if self._closed or self._writer is None:
            raise RuntimeError("DualLogger is closed")
        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, global_step=step)
        self._run.log({"env_step": step, **scalars})
        self._csv_writer.writerow(csv_row)


def _markdown_table(hyperparams: Mapping[str, Any]) -> str:
    """Render hyperparameters as a Markdown table for TensorBoard's text tab."""
    rows = "\n".join(f"| {key} | {value} |" for key, value in hyperparams.items())
    return f"| parameter | value |\n|---|---|\n{rows}"
