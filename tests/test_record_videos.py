"""Finding models and recording videos, with a scripted game and small real files."""

from pathlib import Path
from typing import Any

import imageio
import numpy as np
import pytest
import torch
from conftest import ScriptedEnv, ScriptedStep

from scripts.record_videos import (
    Clip,
    VideoConfig,
    find_clips,
    record_all,
    record_clip,
    scale_frame,
)
from src.agents import AgentConfig, DQNAgent
from src.exceptions import ConfigurationError, GameEnvironmentError
from src.utils.logger import make_structured_logger


def save_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent(AgentConfig(), torch.device("cpu"))
    torch.save({"online_state_dict": agent.online.state_dict()}, path)


def one_game() -> ScriptedEnv:
    """A press, a scored point, then game over: 3 steps and 4 frames with the first."""
    return ScriptedEnv(
        [
            ScriptedStep(),
            ScriptedStep(reward=7.0),
            ScriptedStep(reward=1.0, lives=0, terminated=True),
        ]
    )


def endless_game() -> ScriptedEnv:
    return ScriptedEnv([ScriptedStep()])


def count_frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return int(reader.count_frames())


def make_clip(tmp_path: Path) -> Clip:
    model = tmp_path / "model.pt"
    save_model(model)
    return Clip("dqn_n1_seed1", model, tmp_path / "videos" / "clip.mp4")


def test_find_clips_orders_snapshots_by_step_then_final(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    for name in ("step_100000.pt", "step_000000.pt", "step_020000.pt", "other.pt"):
        save_model(checkpoints / "dqn_n1_seed1" / name)
    save_model(checkpoints / "dqn_n1_seed1.pt")
    save_model(checkpoints / "dqn_n1_seed2.pt")

    clips = find_clips(VideoConfig(log_dir=tmp_path), "dqn_n1")

    assert [clip.video_path.name for clip in clips] == [
        "step_000000.mp4",
        "step_020000.mp4",
        "step_100000.mp4",
        "final.mp4",
    ]
    assert clips[-1].model_path == checkpoints / "dqn_n1_seed1.pt"
    assert clips[0].video_path.parent == tmp_path / "videos" / "dqn_n1_seed1"


def test_find_clips_is_empty_without_models(tmp_path: Path) -> None:
    assert find_clips(VideoConfig(log_dir=tmp_path), "dqn_n1") == []


def test_record_clip_writes_one_full_game_with_raw_return(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    config = VideoConfig(upscale=2)

    result = record_clip(clip, config, 0, one_game)

    assert result.frames == 4
    assert result.game_return == 8.0
    assert not result.cut_off
    assert count_frames(clip.video_path) == 4


def test_video_is_cut_off_at_the_length_limit(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    config = VideoConfig(max_seconds=1, fps=5)

    result = record_clip(clip, config, 0, endless_game)

    assert result.frames == 5
    assert result.cut_off
    assert count_frames(clip.video_path) == 5


def test_recording_closes_the_environment(tmp_path: Path) -> None:
    envs: list[ScriptedEnv] = []

    def make() -> ScriptedEnv:
        envs.append(one_game())
        return envs[-1]

    record_clip(make_clip(tmp_path), VideoConfig(), 0, make)
    assert envs[0].closed


def test_missing_frames_raise(tmp_path: Path) -> None:
    class NoFramesEnv(ScriptedEnv):
        def render(self) -> Any:
            return None

    with pytest.raises(GameEnvironmentError, match="no frame"):
        record_clip(make_clip(tmp_path), VideoConfig(), 0, lambda: NoFramesEnv([ScriptedStep()]))


def test_each_clip_of_a_run_gets_its_own_game(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    save_model(checkpoints / "dqn_n1_seed1" / "step_000000.pt")
    save_model(checkpoints / "dqn_n1_seed1.pt")
    envs: list[ScriptedEnv] = []

    def make() -> ScriptedEnv:
        envs.append(one_game())
        return envs[-1]

    config = VideoConfig(variants=("dqn_n1",), log_dir=tmp_path)
    results = record_all(config, make_structured_logger("test", 1), make)

    assert len(results) == 2
    assert [env.reset_seeds[0] for env in envs] == [20_001, 20_002]


def test_variant_without_models_is_skipped_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = VideoConfig(variants=("ddqn_n5",), log_dir=tmp_path)
    assert record_all(config, make_structured_logger("test", 1), one_game) == []
    assert "no_models_found" in capsys.readouterr().err


def test_scale_frame_repeats_pixels() -> None:
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    scaled = scale_frame(frame, 3)
    assert scaled.shape == (6, 6, 3)
    assert (scaled[:3, :3] == frame[0, 0]).all()


@pytest.mark.parametrize(
    "overrides",
    [{"max_seconds": 0}, {"fps": 0}, {"upscale": 0}, {"seed": -1}, {"epsilon": 1.5}],
)
def test_invalid_config_raises(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        VideoConfig(**overrides)
