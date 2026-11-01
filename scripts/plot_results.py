"""Generate the two-panel ablation figure from the CSV logs.

Run from the project root::

    uv run python scripts/plot_results.py

Panel A shows the 100-episode rolling mean of episodic return with a band of one
standard deviation. Panel B shows the mean predicted Q-value. Both use environment
steps on the x-axis.

Run files are named ``{variant}_metrics.csv`` or ``{variant}_seed{N}_metrics.csv``.
Runs of one variant are averaged on a shared step grid limited to the steps every
run reached. The band is the pooled standard deviation::

    sqrt(mean(rolling variance per run) + variance(rolling mean across runs))

so it combines episode-to-episode and seed-to-seed spread, and reduces to the plain
rolling standard deviation for a single run. Windows are partial for the first
episodes of a run; a point needs at least ``min_periods`` episodes.
"""

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tyro
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlotConfig:
    """Command-line options for the figure."""

    csv_dir: Path = Path("logs/csv")
    """Directory holding ``*_metrics.csv`` files."""
    output: Path = Path("docs/figures/ablation_comparison.png")
    """Output image path."""
    window: int = 100
    """Rolling window, in episodes, for Panel A."""
    min_periods: int = 10
    """Fewest episodes in a window for a point to be drawn."""
    grid_points: int = 500
    """Points per curve; also the number of step bins for Q-values."""
    dpi: int = 300
    """Output resolution."""


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


def discover_runs(csv_dir: Path, variants: Sequence[str]) -> dict[str, list[Path]]:
    """Group CSV paths by variant, warning about files that match no variant."""
    pattern = re.compile(
        rf"^(?P<variant>{'|'.join(map(re.escape, variants))})(?:_seed\d+)?_metrics\.csv$"
    )
    runs: dict[str, list[Path]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match is None:
            print(f"warning: skipping {path.name}: unrecognized run name", file=sys.stderr)
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
        config.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(config.output, dpi=config.dpi, facecolor=chrome.surface, bbox_inches="tight")
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


def main() -> int:
    """Build the figure and return a process exit code."""
    config = tyro.cli(PlotConfig)
    if not 2 <= config.min_periods <= config.window:
        print("error: --min-periods must be between 2 and --window", file=sys.stderr)
        return 2

    matplotlib.use("Agg")
    styles = variant_styles()
    runs = discover_runs(config.csv_dir, list(styles))
    if not runs:
        print(f"error: no recognized run CSVs in {config.csv_dir}", file=sys.stderr)
        return 1

    returns: dict[str, Curve] = {}
    q_values: dict[str, Curve] = {}
    for variant in styles:
        paths = runs.get(variant, [])
        if not paths:
            print(f"warning: no runs for {variant}", file=sys.stderr)
            continue
        if (curve := return_curve(paths, config)) is not None:
            returns[variant] = curve
        else:
            print(f"warning: too few episodes to plot {variant} returns", file=sys.stderr)
        if (curve := q_curve(paths, config)) is not None:
            q_values[variant] = curve
        else:
            print(f"warning: too few updates to plot {variant} Q-values", file=sys.stderr)

    render_figure(returns, q_values, styles, config)
    print_summary(returns, q_values, list(styles))
    print(f"Saved {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
