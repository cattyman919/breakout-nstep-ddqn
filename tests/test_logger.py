"""DualLogger CSV behavior and structured diagnostics, in W&B offline mode."""

import csv
import io
import json
from pathlib import Path

import pytest

from src.exceptions import ConfigurationError
from src.utils.logger import DualLogger, make_structured_logger


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def open_logger(log_dir: Path) -> DualLogger:
    return DualLogger("dqn_n1_seed1", {"a": 1}, wandb_mode="offline", log_dir=log_dir)


def test_csv_has_three_row_kinds_with_other_columns_empty(tmp_path: Path) -> None:
    with open_logger(tmp_path) as logger:
        assert logger.rolling_return is None
        logger.log_episode(10, 1, 4.0, 10, 0.9)
        logger.log_train_step(20, 0.5, 1.5)
        logger.log_eval(30, 2.0, 1.0)
        assert logger.rolling_return == 4.0
        path = logger.csv_path

    episode, train, evaluation = read_rows(path)
    assert episode["episodic_return"] == "4.0"
    assert episode["eval_predicted_q"] == ""
    assert train["td_loss"] == "0.5"
    assert train["episodic_return"] == ""
    assert evaluation["step"] == "30"
    assert evaluation["eval_predicted_q"] == "2.0"
    assert evaluation["eval_actual_return"] == "1.0"
    assert evaluation["td_loss"] == ""


def test_existing_csv_is_not_overwritten(tmp_path: Path) -> None:
    open_logger(tmp_path).close()
    with pytest.raises(FileExistsError):
        open_logger(tmp_path)


def test_logger_closes_when_a_run_fails(tmp_path: Path) -> None:
    logger = open_logger(tmp_path)
    with pytest.raises(RuntimeError, match="boom"), logger:
        logger.log_eval(1, 0.0, 0.0)
        raise RuntimeError("boom")
    assert logger._closed
    assert len(read_rows(logger.csv_path)) == 1


def test_invalid_run_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="run_name"):
        DualLogger("a/b", {}, wandb_mode="offline", log_dir=tmp_path)


def test_structured_logger_emits_json_with_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = make_structured_logger("dqn_n1_seed1", 1)
    log.info("started")
    log.warning("late", step=5)
    log_buffer = io.StringIO(capsys.readouterr().err)

    first, second = (json.loads(line) for line in log_buffer)
    assert first["event"] == "started"
    assert first["level"] == "info"
    assert first["run_name"] == "dqn_n1_seed1"
    assert first["seed"] == 1
    assert first["step"] is None
    assert "timestamp" in first
    assert second["step"] == 5


def test_structured_logger_records_the_stack_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = make_structured_logger("run", 0)
    try:
        raise ValueError("bad value")
    except ValueError:
        log.exception("failed")
    entry = json.loads(capsys.readouterr().err)
    assert "ValueError: bad value" in entry["exception"]
    assert "Traceback" in entry["exception"]


def test_structured_log_lines_print_whole_above_a_progress_bar(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tqdm import tqdm

    log = make_structured_logger("run", 1)
    with tqdm(total=10, desc="bar") as bar:
        bar.update(3)
        log.info("first")
        log.info("second")
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [json.loads(line)["event"] for line in lines] == ["first", "second"]
