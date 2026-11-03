"""Record trained models playing Breakout, one video per learning phase.

Run from the project root, after training::

    uv run python -m scripts.record_videos

For each variant this reads the snapshots and final model of one seed from
``logs/steps_<steps>``, or ``--log-dir``, and writes
``{log_dir}/videos/{run_name}/step_{step}.mp4`` and ``final.mp4``. Each video is one game, which
stops at game over or at the length limit. No window opens, so this runs much faster than the
game plays. Recording reads saved models only, so it cannot affect training.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import imageio
import numpy as np
import torch
import tyro

from scripts.train import TrainConfig, steps_log_dir
from src.agents import AgentConfig, DQNAgent
from src.environment import BreakoutGame, make_breakout_env
from src.exceptions import BaseProjectError, ConfigurationError, GameEnvironmentError
from src.utils.logger import make_structured_logger


@dataclass(frozen=True)
class VideoConfig:
    """Command-line options for recording."""

    variants: tuple[Literal["dqn_n1", "ddqn_n1", "ddqn_n3", "ddqn_n5"], ...] = (
        "dqn_n1",
        "ddqn_n1",
        "ddqn_n3",
        "ddqn_n5",
    )
    """Variants to record."""
    seed: int = 1
    """Which seed's models to record."""
    steps: int = TrainConfig.total_timesteps
    """Run length to record. Sets the default folder, logs/steps_<steps>."""
    log_dir: Path | None = None
    """Folder holding checkpoints/; videos go in videos/ beneath it. Defaults to logs/steps_<steps>."""
    max_seconds: int = 180
    """Longest video. A game still running at this point is cut off."""
    fps: int = 30
    """Video frames per second. Each frame is one agent step, so 15 is real game speed."""
    epsilon: float = 0.01
    """Random move rate, the same as evaluation, so a policy cannot get stuck forever."""
    upscale: int = 3
    """Whole-number enlargement of the 160x210 game picture."""
    game_seed_offset: int = 20_000
    """Added to seed for the games, which keeps them apart from training and evaluation."""

    def __post_init__(self) -> None:
        for name, value in (
            ("max_seconds", self.max_seconds),
            ("fps", self.fps),
            ("upscale", self.upscale),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must not be negative, got {self.seed}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ConfigurationError(f"epsilon must be in [0, 1], got {self.epsilon}")

    @property
    def run_dir(self) -> Path:
        """The folder holding this length's checkpoints and videos."""
        return self.log_dir if self.log_dir is not None else steps_log_dir(self.steps)

    @property
    def max_frames(self) -> int:
        """The largest number of frames in one video."""
        return self.max_seconds * self.fps


@dataclass(frozen=True)
class Clip:
    """One video to make: the model to load and the file to write."""

    run_name: str
    model_path: Path
    video_path: Path


@dataclass(frozen=True)
class ClipResult:
    """What happened in one recorded game."""

    clip: Clip
    frames: int
    game_return: float
    cut_off: bool


def find_clips(config: VideoConfig, variant: str) -> list[Clip]:
    """List the videos to make for one variant: snapshots by step, then the final model."""
    run_name = f"{variant}_seed{config.seed}"
    checkpoints = config.run_dir / "checkpoints"
    videos = config.run_dir / "videos" / run_name

    pattern = re.compile(r"^step_(\d+)\.pt$")
    snapshots = [
        path for path in (checkpoints / run_name).glob("step_*.pt") if pattern.match(path.name)
    ]
    snapshots.sort(key=lambda path: int(path.stem.removeprefix("step_")))
    clips = [Clip(run_name, path, videos / f"{path.stem}.mp4") for path in snapshots]

    final = checkpoints / f"{run_name}.pt"
    if final.exists():
        clips.append(Clip(run_name, final, videos / "final.mp4"))
    return clips


def load_agent(path: Path) -> DQNAgent:
    """Load a saved model into an agent on the CPU."""
    device = torch.device("cpu")
    agent = DQNAgent(AgentConfig(), device)
    checkpoint = torch.load(path, map_location=device)
    agent.online.load_state_dict(checkpoint["online_state_dict"])
    return agent


def scale_frame(frame: Any, upscale: int) -> Any:
    """Enlarge a frame by repeating pixels, which keeps the game picture sharp."""
    return np.repeat(np.repeat(frame, upscale, axis=0), upscale, axis=1)


def _open_video(path: Path, fps: int) -> Any:
    """Open an H.264 video writer. A block size of 1 keeps the frame size unchanged."""
    return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)


def record_clip(
    clip: Clip,
    config: VideoConfig,
    game_seed: int,
    env_factory: Callable[[], gym.Env[Any, Any]],
) -> ClipResult:
    """Play one game with the clip's model and write it to the clip's video file.

    Raises:
        GameEnvironmentError: If the environment does not return frames.
    """
    agent = load_agent(clip.model_path)
    rng = np.random.default_rng(game_seed)
    clip.video_path.parent.mkdir(parents=True, exist_ok=True)
    env = env_factory()

    def next_frame() -> Any:
        frame = env.render()
        if frame is None:
            raise GameEnvironmentError("environment returned no frame; use render_mode rgb_array")
        return scale_frame(frame, config.upscale)

    with (
        BreakoutGame(env, game_seed) as game,
        _open_video(clip.video_path, config.fps) as writer,
    ):
        observation = game.reset()
        writer.append_data(next_frame())
        frames = 1
        total = 0.0
        game_over = False
        while not game_over and frames < config.max_frames:
            if game.needs_fire:
                result = game.press_fire()
            else:
                result = game.step(agent.act(observation, config.epsilon, rng))
            observation = result.observation
            total += result.raw_reward
            game_over = result.game_over
            writer.append_data(next_frame())
            frames += 1
    return ClipResult(clip, frames, total, cut_off=not game_over)


def record_all(
    config: VideoConfig,
    log: Any,
    env_factory: Callable[[], gym.Env[Any, Any]],
) -> list[ClipResult]:
    """Record every clip of every selected variant."""
    results: list[ClipResult] = []
    for variant in config.variants:
        clips = find_clips(config, variant)
        if not clips:
            log.warning("no_models_found", variant=variant, log_dir=str(config.run_dir))
            continue
        for index, clip in enumerate(clips):
            # Each clip gets its own game, so the last snapshot and the final model, which hold
            # the same weights, still give two different videos.
            game_seed = config.seed + config.game_seed_offset + index
            result = record_clip(clip, config, game_seed, env_factory)
            results.append(result)
            if result.cut_off:
                log.warning("video_cut_off", video=str(clip.video_path), frames=result.frames)
    return results


def _make_rgb_env() -> gym.Env[Any, Any]:
    return make_breakout_env(render_mode="rgb_array")


def print_summary(results: list[ClipResult], fps: int) -> None:
    """Print one line per video."""
    print(f"{'video':<58}{'seconds':>8}{'return':>8}  note")
    for result in results:
        note = "cut off at the length limit" if result.cut_off else ""
        seconds = result.frames / fps
        print(f"{result.clip.video_path!s:<58}{seconds:>8.1f}{result.game_return:>8.0f}  {note}")


def main() -> int:
    """Record the videos and return a process exit code."""
    config = tyro.cli(VideoConfig)
    log = make_structured_logger("record_videos", config.seed)
    try:
        results = record_all(config, log, _make_rgb_env)
    except (BaseProjectError, OSError, RuntimeError, ValueError, KeyError):
        log.exception("recording_failed")
        return 1
    if not results:
        log.error("no_videos_recorded", log_dir=str(config.run_dir), seed=config.seed)
        return 1
    print_summary(results, config.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
