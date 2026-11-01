"""Run discovery, evaluation averaging and figure output, from synthetic CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_results import PlotConfig, build_figures, discover_runs, eval_curve
from src.exceptions import ConfigurationError
from src.utils.logger import DualLogger, make_structured_logger


def write_run(
    directory: Path, name: str, eval_rows: list[tuple[int, float, float]], *, episodes: int = 30
) -> Path:
    """Write a CSV with episode, train and evaluation rows."""
    rows: list[dict[str, float]] = []
    for index in range(episodes):
        rows.append({"step": 10.0 * index, "episode": index + 1.0, "episodic_return": 2.0})
        rows.append({"step": 10.0 * index + 5, "mean_q_value": 0.5, "td_loss": 0.1})
    for step, predicted, actual in eval_rows:
        rows.append({"step": step, "eval_predicted_q": predicted, "eval_actual_return": actual})
    path = directory / f"{name}_metrics.csv"
    # DualLogger always writes every column, even those with no values yet.
    columns = list(DualLogger.CSV_FIELDS)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def plot_config(tmp_path: Path) -> PlotConfig:
    return PlotConfig(
        csv_dir=tmp_path,
        output=tmp_path / "out" / "a.png",
        overestimation_output=tmp_path / "out" / "b.png",
        dpi=50,
    )


def test_discover_requires_the_seed_in_the_run_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = write_run(tmp_path, "dqn_n1_seed1", [])
    write_run(tmp_path, "dqn_n1", [])
    write_run(tmp_path, "other_seed1", [])

    runs = discover_runs(tmp_path, ["dqn_n1"], make_structured_logger("plot", None))

    assert runs == {"dqn_n1": [good]}
    errors = capsys.readouterr().err
    assert "skipping_unrecognized_run" in errors
    assert "other_seed1" in errors


def test_eval_curve_averages_across_seeds(tmp_path: Path) -> None:
    first = write_run(tmp_path, "dqn_n1_seed1", [(100, 1.0, 0.0), (200, 3.0, 1.0)])
    second = write_run(tmp_path, "dqn_n1_seed2", [(100, 2.0, 1.0), (200, 5.0, 2.0)])

    curve = eval_curve([first, second])

    assert curve is not None
    assert curve.num_runs == 2
    np.testing.assert_allclose(curve.steps, [100, 200])
    np.testing.assert_allclose(curve.predicted, [1.5, 4.0])
    np.testing.assert_allclose(curve.actual, [0.5, 1.5])


def test_eval_curve_keeps_only_steps_every_run_reached(tmp_path: Path) -> None:
    first = write_run(tmp_path, "dqn_n1_seed1", [(100, 1.0, 0.0), (200, 3.0, 1.0)])
    second = write_run(tmp_path, "dqn_n1_seed2", [(100, 2.0, 1.0)])

    curve = eval_curve([first, second])

    assert curve is not None
    np.testing.assert_allclose(curve.steps, [100])


def test_eval_curve_is_none_without_evaluations(tmp_path: Path) -> None:
    assert eval_curve([write_run(tmp_path, "dqn_n1_seed1", [])]) is None


def test_build_figures_writes_both_images(tmp_path: Path) -> None:
    write_run(tmp_path, "dqn_n1_seed1", [(100, 1.0, 0.5), (200, 3.0, 1.0)])
    write_run(tmp_path, "ddqn_n3_seed1", [(100, 1.0, 0.5), (200, 2.0, 1.5)])
    config = plot_config(tmp_path)

    assert build_figures(config, make_structured_logger("plot", None)) == 0

    assert config.output.stat().st_size > 0
    assert config.overestimation_output.stat().st_size > 0


def test_build_figures_reports_when_there_are_no_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_figures(plot_config(tmp_path), make_structured_logger("plot", None)) == 1
    assert "no_recognized_runs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides",
    [{"min_periods": 1}, {"min_periods": 200}, {"grid_points": 1}, {"dpi": 0}],
)
def test_invalid_config_raises(overrides: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError):
        PlotConfig(**overrides)
