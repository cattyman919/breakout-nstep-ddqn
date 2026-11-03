"""Generate the ablation and overestimation figures from the CSV logs.

Run from the project root::

    uv run python -m scripts.plot_results

The ablation figure has two panels. Panel A shows the 100-episode rolling mean of episodic return with a band of one
standard deviation. Panel B shows the mean predicted Q-value. Both use environment
steps on the x-axis.

Run files are named ``{variant}_seed{N}_metrics.csv``; other files are skipped with a
warning. Runs of one variant are averaged on a shared step grid limited to the steps every
run reached. The band is the pooled standard deviation::

    sqrt(mean(rolling variance per run) + variance(rolling mean across runs))

so it combines episode-to-episode and seed-to-seed spread, and reduces to the plain
rolling standard deviation for a single run. Windows are partial for the first
episodes of a run; a point needs at least ``min_periods`` episodes.

The overestimation figure has one panel per variant, comparing the predicted value with the
discounted return earned at each evaluation point. Seeds are averaged at the evaluation steps
every run reached.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tyro
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from scripts.train import TrainConfig, steps_log_dir
from src.exceptions import ConfigurationError
from src.utils.logger import make_structured_logger


@dataclass(frozen=True)
class PlotConfig:
    """Command-line options for the figure."""

    steps: int = TrainConfig.total_timesteps
    """Run length to plot. Sets the default folders, logs/steps_<steps>/csv and figures/."""
    csv_dir: Path | None = None
    """Directory holding ``*_metrics.csv`` files. Defaults to logs/steps_<steps>/csv."""
    output: Path | None = None
    """Ablation figure path. Defaults to logs/steps_<steps>/figures/ablation_comparison.png."""
    overestimation_output: Path | None = None
    """Overestimation figure path. Defaults to the same folder as the ablation figure."""
    window: int = 100
    """Rolling window, in episodes, for Panel A."""
    min_periods: int = 10
    """Fewest episodes in a window for a point to be drawn."""
    grid_points: int = 500
    """Points per curve; also the number of step bins for Q-values."""
    dpi: int = 300
    """Output resolution."""

    def __post_init__(self) -> None:
        if not 2 <= self.min_periods <= self.window:
            raise ConfigurationError(
                f"min_periods must be between 2 and window ({self.window}), got {self.min_periods}"
            )
        if self.grid_points < 2:
            raise ConfigurationError(f"grid_points must be at least 2, got {self.grid_points}")
        if self.dpi <= 0:
            raise ConfigurationError(f"dpi must be positive, got {self.dpi}")

    @property
    def csv_directory(self) -> Path:
        """The folder of run CSVs to read."""
        return self.csv_dir if self.csv_dir is not None else steps_log_dir(self.steps) / "csv"

    @property
    def comparison_path(self) -> Path:
        """Where the ablation figure is written."""
        if self.output is not None:
            return self.output
        return steps_log_dir(self.steps) / "figures" / "ablation_comparison.png"

    @property
    def overestimation_path(self) -> Path:
        """Where the overestimation figure is written."""
        if self.overestimation_output is not None:
            return self.overestimation_output
        return steps_log_dir(self.steps) / "figures" / "overestimation_evaluation.png"


@dataclass(frozen=True)
class SeriesStyle:
    """Fixed visual identity of one variant, independent of which runs exist."""

    color: str
    linestyle: str | tuple[int, tuple[float, ...]]
    legend: str


@dataclass(frozen=True)
class Curve:
    """One variant's line, with an optional band half-width."""

    steps: NDArray[np.float64]
    values: NDArray[np.float64]
    spread: NDArray[np.float64] | None
    num_runs: int


@dataclass(frozen=True)
class EvalCurve:
    """One variant's predicted values and earned returns at each evaluation point."""

    steps: NDArray[np.float64]
    predicted: NDArray[np.float64]
    actual: NDArray[np.float64]
    num_runs: int


@dataclass(frozen=True)
class Chrome:
    """Ink colors for everything that is not data."""

    ink: str = "#0b0b0b"
    secondary: str = "#52514e"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    surface: str = "#fcfcfb"


def variant_styles() -> dict[str, SeriesStyle]:
    """Return variants in plotting order, each with its color and line style.

    Line style is a second channel beside color so variants stay separable in
    grayscale print and for color-blind readers.
    """
    return {
        "dqn_n1": SeriesStyle("#2a78d6", "-", "dqn_n1: DQN (n = 1)"),
        "ddqn_n1": SeriesStyle("#eb6834", "-", "ddqn_n1: Double DQN (n = 1)"),
        "ddqn_n3": SeriesStyle("#1baf7a", (0, (5, 2)), "ddqn_n3: Double DQN (n = 3)"),
        "ddqn_n5": SeriesStyle("#eda100", (0, (1, 1.5)), "ddqn_n5: Double DQN (n = 5)"),
    }


def discover_runs(csv_dir: Path, variants: Sequence[str], log: Any) -> dict[str, list[Path]]:
    """Group CSV paths by variant, warning about files that match no variant."""
    pattern = re.compile(
        rf"^(?P<variant>{'|'.join(map(re.escape, variants))})_seed\d+_metrics\.csv$"
    )
    runs: dict[str, list[Path]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match is None:
            log.warning("skipping_unrecognized_run", file=path.name)
            continue
        runs.setdefault(match["variant"], []).append(path)
    return runs


def load_metric(path: Path, column: str) -> pd.DataFrame:
    """Read ``step`` and one metric from a run CSV, keeping rows where it is set."""
    frame = pd.read_csv(path)
    missing = {"step", column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    metric = pd.DataFrame(
        {
            "step": pd.to_numeric(frame["step"], errors="coerce"),
            "value": pd.to_numeric(frame[column], errors="coerce"),
        }
    )
    return metric.dropna().sort_values("step", kind="stable").reset_index(drop=True)


def _shared_grid(
    runs: Sequence[tuple[NDArray[np.float64], ...]], points: int
) -> NDArray[np.float64] | None:
    """Return a step grid covering only the range every run reached."""
    start = max(float(run[0][0]) for run in runs)
    end = min(float(run[0][-1]) for run in runs)
    return np.linspace(start, end, points) if end > start else None


def return_curve(paths: Sequence[Path], config: PlotConfig) -> Curve | None:
    """Build the rolling-return curve for one variant, or None without enough data."""
    runs: list[tuple[NDArray[np.float64], ...]] = []
    for path in paths:
        episodes = load_metric(path, "episodic_return")
        rolling = episodes["value"].rolling(config.window, min_periods=config.min_periods)
        summary = pd.DataFrame(
            {"step": episodes["step"], "mean": rolling.mean(), "std": rolling.std()}
        ).dropna()
        if len(summary) >= 2:
            runs.append(
                (
                    summary["step"].to_numpy(dtype=np.float64),
                    summary["mean"].to_numpy(dtype=np.float64),
                    summary["std"].to_numpy(dtype=np.float64),
                )
            )
    if not runs:
        return None
    grid = _shared_grid(runs, config.grid_points)
    if grid is None:
        return None

    means = np.stack([np.interp(grid, steps, mean) for steps, mean, _ in runs])
    variances = np.stack([np.interp(grid, steps, std**2) for steps, _, std in runs])
    pooled_variance = variances.mean(axis=0) + means.var(axis=0)
    return Curve(grid, means.mean(axis=0), np.sqrt(pooled_variance), len(runs))


def q_curve(paths: Sequence[Path], config: PlotConfig) -> Curve | None:
    """Build the binned mean-Q curve for one variant, or None without enough data."""
    runs: list[tuple[NDArray[np.float64], ...]] = []
    for path in paths:
        updates = load_metric(path, "mean_q_value")
        if updates.empty:
            continue
        low, high = float(updates["step"].iloc[0]), float(updates["step"].iloc[-1])
        width = max((high - low) / config.grid_points, 1.0)
        bins = ((updates["step"] - low) // width).astype(int).clip(upper=config.grid_points - 1)
        binned = updates.groupby(bins).mean()
        if len(binned) >= 2:
            runs.append(
                (
                    binned["step"].to_numpy(dtype=np.float64),
                    binned["value"].to_numpy(dtype=np.float64),
                )
            )
    if not runs:
        return None
    grid = _shared_grid(runs, config.grid_points)
    if grid is None:
        return None

    values = np.stack([np.interp(grid, steps, value) for steps, value in runs])
    return Curve(grid, values.mean(axis=0), None, len(runs))


def eval_curve(paths: Sequence[Path]) -> EvalCurve | None:
    """Average one variant's evaluations across runs, at the steps every run reached."""
    predicted: list[pd.Series[float]] = []
    actual: list[pd.Series[float]] = []
    for path in paths:
        predicted.append(load_metric(path, "eval_predicted_q").set_index("step")["value"])
        actual.append(load_metric(path, "eval_actual_return").set_index("step")["value"])
    predicted_by_run = pd.concat(predicted, axis=1, join="inner")
    actual_by_run = pd.concat(actual, axis=1, join="inner")
    if predicted_by_run.empty or actual_by_run.empty:
        return None
    return EvalCurve(
        predicted_by_run.index.to_numpy(dtype=np.float64),
        predicted_by_run.mean(axis=1).to_numpy(dtype=np.float64),
        actual_by_run.mean(axis=1).to_numpy(dtype=np.float64),
        len(paths),
    )


def tail_mean(curve: Curve, fraction: float = 0.1) -> float:
    """Average a curve over the final ``fraction`` of its steps."""
    threshold = curve.steps[-1] - fraction * (curve.steps[-1] - curve.steps[0])
    return float(curve.values[curve.steps >= threshold].mean())


def _format_steps(value: float, _position: float) -> str:
    return f"{value / 1000:.0f}k" if value else "0"


def _style_axes(ax: Axes, title: str, ylabel: str, chrome: Chrome) -> None:
    ax.set_facecolor(chrome.surface)
    ax.set_title(title, loc="left", color=chrome.ink, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Environment steps", color=chrome.secondary)
    ax.set_ylabel(ylabel, color=chrome.secondary)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(axis="y", color=chrome.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=chrome.secondary, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(chrome.axis)


def _label_line_ends(
    ax: Axes, labels: Sequence[tuple[float, str]], chrome: Chrome, min_gap_fraction: float = 0.06
) -> None:
    """Write labels in the right margin, nudged apart so they cannot overlap."""
    bottom, top = ax.get_ylim()
    gap = min_gap_fraction * (top - bottom)
    ordered = sorted(labels)
    positions = [y for y, _ in ordered]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    overflow = positions[-1] - top if positions else 0.0
    if overflow > 0:
        positions = [y - overflow for y in positions]
    for y, (_, text) in zip(positions, ordered, strict=True):
        ax.text(
            1.012,
            y,
            text,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=chrome.secondary,
        )


def _draw_panel(
    ax: Axes,
    curves: Mapping[str, Curve],
    styles: Mapping[str, SeriesStyle],
    chrome: Chrome,
) -> None:
    for variant, style in styles.items():
        curve = curves.get(variant)
        if curve is None:
            continue
        if curve.spread is not None:
            ax.fill_between(
                curve.steps,
                curve.values - curve.spread,
                curve.values + curve.spread,
                color=style.color,
                alpha=0.13,
                linewidth=0,
            )
        ax.plot(
            curve.steps,
            curve.values,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=1.6,
            label=style.legend,
        )
    if not curves:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color=chrome.secondary)


def _draw_overestimation_gap(ax: Axes, curves: Mapping[str, Curve], chrome: Chrome) -> None:
    """Mark the signed Q-value gap between DQN and Double DQN at the same n."""
    dqn, ddqn = curves.get("dqn_n1"), curves.get("ddqn_n1")
    if dqn is None or ddqn is None:
        return
    dqn_level, ddqn_level = tail_mean(dqn), tail_mean(ddqn)
    x = float(min(dqn.steps[-1], ddqn.steps[-1])) * 0.95
    ax.annotate(
        "",
        xy=(x, dqn_level),
        xytext=(x, ddqn_level),
        arrowprops={"arrowstyle": "<->", "color": chrome.ink, "linewidth": 0.9},
    )
    ax.text(
        0.02,
        0.97,
        f"Arrow: dqn_n1 − ddqn_n1 = {dqn_level - ddqn_level:+.2f} (mean over final 10% of steps)",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=chrome.ink,
    )


def render_figure(
    returns: Mapping[str, Curve],
    q_values: Mapping[str, Curve],
    styles: Mapping[str, SeriesStyle],
    config: PlotConfig,
) -> None:
    """Draw both panels and save the figure."""
    chrome = Chrome()
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]}):
        figure, (return_ax, q_ax) = plt.subplots(
            1, 2, figsize=(12.5, 4.8), facecolor=chrome.surface
        )
        _style_axes(
            return_ax,
            f"A  Episodic return ({config.window}-episode rolling mean ± 1 SD)",
            "Episodic return",
            chrome,
        )
        _style_axes(q_ax, "B  Mean predicted Q-value", r"Mean $\max_a Q_\phi(s, a)$", chrome)

        _draw_panel(return_ax, returns, styles, chrome)
        _draw_panel(q_ax, q_values, styles, chrome)
        return_ax.set_ylim(bottom=0)
        _draw_overestimation_gap(q_ax, q_values, chrome)

        for ax, curves in ((return_ax, returns), (q_ax, q_values)):
            _label_line_ends(
                ax,
                [(float(curve.values[-1]), name) for name, curve in curves.items()],
                chrome,
            )

        handles, labels = (return_ax if returns else q_ax).get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncols=4,
            frameon=False,
            fontsize=9.5,
            labelcolor=chrome.ink,
            handlelength=3,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=5)
        config.comparison_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            config.comparison_path, dpi=config.dpi, facecolor=chrome.surface, bbox_inches="tight"
        )
        plt.close(figure)


def render_overestimation_figure(
    curves: Mapping[str, EvalCurve],
    styles: Mapping[str, SeriesStyle],
    config: PlotConfig,
) -> None:
    """Draw predicted value against earned return, in one panel per variant."""
    chrome = Chrome()
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]}):
        figure, axes = plt.subplots(
            1, len(styles), figsize=(13.5, 3.8), sharex=True, sharey=True, facecolor=chrome.surface
        )
        for ax, (variant, style) in zip(axes, styles.items(), strict=True):
            _style_axes(ax, style.legend, "", chrome)
            curve = curves.get(variant)
            if curve is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    color=chrome.secondary,
                )
                continue
            ax.plot(curve.steps, curve.predicted, color=style.color, linewidth=2)
            ax.plot(
                curve.steps, curve.actual, color=style.color, linewidth=2, linestyle=(0, (4, 2))
            )
        axes[0].set_ylabel("Discounted clipped return", color=chrome.secondary)
        for ax, variant in zip(axes, styles, strict=True):
            curve = curves.get(variant)
            if curve is not None:
                _label_line_ends(
                    ax,
                    [
                        (float(curve.predicted[-1]), "Predicted"),
                        (float(curve.actual[-1]), "Actual"),
                    ],
                    chrome,
                )

        handles = [
            Line2D([], [], color=chrome.ink, linewidth=2, label="Predicted value"),
            Line2D(
                [],
                [],
                color=chrome.ink,
                linewidth=2,
                linestyle=(0, (4, 2)),
                label="Actual discounted return",
            ),
        ]
        figure.legend(
            handles=handles,
            loc="upper center",
            ncols=2,
            frameon=False,
            fontsize=9.5,
            labelcolor=chrome.ink,
            handlelength=3,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.91), w_pad=4.5)
        config.overestimation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            config.overestimation_path,
            dpi=config.dpi,
            facecolor=chrome.surface,
            bbox_inches="tight",
        )
        plt.close(figure)


def print_summary(
    returns: Mapping[str, Curve], q_values: Mapping[str, Curve], variants: Sequence[str]
) -> None:
    """Print the plotted end-of-training values as a plain-text table."""
    print(f"{'variant':<9}{'runs':>5}{'last step':>11}{'return':>9}{'mean Q':>9}")
    for variant in variants:
        curve_a, curve_b = returns.get(variant), q_values.get(variant)
        if curve_a is None and curve_b is None:
            print(f"{variant:<9}  no data")
            continue
        reference = curve_a if curve_a is not None else curve_b
        assert reference is not None
        final_return = f"{curve_a.values[-1]:.2f}" if curve_a is not None else "-"
        final_q = f"{tail_mean(curve_b):.2f}" if curve_b is not None else "-"
        print(
            f"{variant:<9}{reference.num_runs:>5}{reference.steps[-1]:>11.0f}"
            f"{final_return:>9}{final_q:>9}"
        )


def build_figures(config: PlotConfig, log: Any) -> int:
    """Build both figures and return a process exit code."""
    matplotlib.use("Agg")
    styles = variant_styles()
    runs = discover_runs(config.csv_directory, list(styles), log)
    if not runs:
        log.error("no_recognized_runs", csv_dir=str(config.csv_directory))
        return 1

    returns: dict[str, Curve] = {}
    q_values: dict[str, Curve] = {}
    evaluations: dict[str, EvalCurve] = {}
    for variant in styles:
        paths = runs.get(variant, [])
        if not paths:
            log.warning("no_runs_for_variant", variant=variant)
            continue
        if (curve := return_curve(paths, config)) is not None:
            returns[variant] = curve
        else:
            log.warning("too_few_episodes", variant=variant)
        if (curve := q_curve(paths, config)) is not None:
            q_values[variant] = curve
        else:
            log.warning("too_few_updates", variant=variant)
        if (evaluation := eval_curve(paths)) is not None:
            evaluations[variant] = evaluation
        else:
            log.warning("no_shared_evaluations", variant=variant)

    render_figure(returns, q_values, styles, config)
    render_overestimation_figure(evaluations, styles, config)
    print_summary(returns, q_values, list(styles))
    print(f"Saved {config.comparison_path}")
    print(f"Saved {config.overestimation_path}")
    return 0


def main() -> int:
    """Parse the CLI, build the figures and return a process exit code."""
    config = tyro.cli(PlotConfig)
    log = make_structured_logger("plot_results", None)
    try:
        return build_figures(config, log)
    except (OSError, ValueError):
        log.exception("plotting_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
