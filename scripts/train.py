"""Training entry point. See docs/ARCHITECTURE.md and docs/EXPERIMENTS.md.

Run from the project root::

    uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu
"""

import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, Self

import gymnasium as gym
import numpy as np
import torch
import tyro
import wandb
from tqdm import tqdm

from src.agents import AgentConfig, DQNAgent
from src.buffers import NStepWindow, ReplayBuffer
from src.environment import BreakoutGame, make_breakout_env
from src.evaluation import evaluate
from src.exceptions import BaseProjectError, ConfigurationError, TrainingError
from src.utils.logger import DualLogger, make_structured_logger


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one run. See docs/EXPERIMENTS.md for each value's source."""

    variant: Literal["dqn_n1", "ddqn_n1", "ddqn_n3", "ddqn_n5"] = "ddqn_n3"
    """Ablation configuration."""
    seed: int = 1
    """Seed for torch, numpy and gymnasium. Evaluation uses this plus eval_seed_offset."""
    device: Literal["mps", "cpu"] = "mps"
    """Compute device. A request for mps falls back to cpu if it is unavailable."""
    wandb_mode: Literal["online", "offline"] = "online"
    """Weights & Biases sync mode."""
    log_dir: Path | None = None
    """Root directory for all run outputs. Defaults to logs/steps_<total_timesteps>."""

    total_timesteps: int = 500_000
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_capacity: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.1

    learning_starts: int = 10_000
    train_frequency: int = 4
    huber_delta: float = 1.0
    clip_rewards: bool = True
    """Use the sign of each reward in training targets. Logged returns stay raw."""
    terminal_on_life_loss: bool = True
    """End training targets at a lost life, without resetting the environment."""
    auto_fire: bool = True
    """Press FIRE once after each reset and lost life."""

    log_interval: int = 100
    """Gradient updates averaged into one train row. A final partial group is dropped."""
    eval_interval: int = 25_000
    """Environment steps between evaluations."""
    eval_games: int = 5
    """Complete games played at each evaluation."""
    eval_epsilon: float = 0.01
    """Exploration rate during evaluation."""
    eval_seed_offset: int = 10_000
    """Added to seed for the evaluation environment and generator."""
    snapshot_interval: int = 100_000
    """Environment steps between model snapshots. The untrained model is saved too."""
    snapshot_seeds: tuple[int, ...] = (1,)
    """Seeds whose runs save snapshots, for scripts/record_videos.py."""
    progress_file: Path | None = None
    """Not a hyperparameter: a file to keep updated with the step count, for scripts/study.py."""

    def __post_init__(self) -> None:
        for name, value in (
            ("total_timesteps", self.total_timesteps),
            ("learning_rate", self.learning_rate),
            ("batch_size", self.batch_size),
            ("buffer_capacity", self.buffer_capacity),
            ("train_frequency", self.train_frequency),
            ("huber_delta", self.huber_delta),
            ("log_interval", self.log_interval),
            ("eval_interval", self.eval_interval),
            ("eval_games", self.eval_games),
            ("snapshot_interval", self.snapshot_interval),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must not be negative, got {self.seed}")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigurationError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 < self.tau <= 1.0:
            raise ConfigurationError(f"tau must be in (0, 1], got {self.tau}")
        for name, rate in (
            ("epsilon_start", self.epsilon_start),
            ("epsilon_end", self.epsilon_end),
            ("eval_epsilon", self.eval_epsilon),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1], got {rate}")
        if self.epsilon_end > self.epsilon_start:
            raise ConfigurationError("epsilon_end must not exceed epsilon_start")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ConfigurationError(
                f"epsilon_decay_fraction must be in (0, 1], got {self.epsilon_decay_fraction}"
            )
        if self.learning_starts < self.batch_size:
            raise ConfigurationError(
                f"learning_starts ({self.learning_starts}) must be at least "
                f"batch_size ({self.batch_size})"
            )
        if self.auto_fire and not self.terminal_on_life_loss:
            raise ConfigurationError(
                "auto_fire requires terminal_on_life_loss: otherwise an n-step window would "
                "span a FIRE press that is not stored in replay"
            )

    @property
    def output_dir(self) -> Path:
        """Where this run's outputs go: log_dir if set, else one folder per run length."""
        return self.log_dir if self.log_dir is not None else steps_log_dir(self.total_timesteps)

    @property
    def run_name(self) -> str:
        """Run identifier; scripts/plot_results.py parses this format."""
        return f"{self.variant}_seed{self.seed}"

    @property
    def double_q(self) -> bool:
        """Whether this variant uses the Double DQN target."""
        return self.variant.startswith("ddqn")

    @property
    def n_step(self) -> int:
        """The n of this variant's n-step return."""
        return int(self.variant.rsplit("_n", maxsplit=1)[1])

    @property
    def agent_config(self) -> AgentConfig:
        """The learning settings for this variant's agent."""
        return AgentConfig(
            gamma=self.gamma,
            tau=self.tau,
            learning_rate=self.learning_rate,
            huber_delta=self.huber_delta,
            double_q=self.double_q,
        )

    @property
    def saves_snapshots(self) -> bool:
        """Whether this run saves model snapshots during training."""
        return self.seed in self.snapshot_seeds

    def epsilon_at(self, step: int) -> float:
        """Return the exploration rate after ``step`` environment steps."""
        decay_steps = self.epsilon_decay_fraction * self.total_timesteps
        progress = min(step / decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


class RunProgress:
    """Reports a run's progress: a bar in a terminal, and optionally a step-count file.

    The bar shows only when ``show_bar`` is true, so log files never fill with redraws. The
    file is replaced atomically, so ``scripts/study.py`` never reads a half-written number.
    Neither affects training.
    """

    BAR_EVERY: ClassVar[int] = 1_000
    FILE_EVERY: ClassVar[int] = 10_000

    def __init__(self, total: int, label: str, file: Path | None, *, show_bar: bool) -> None:
        self._total = total
        self._file = file
        self._shown = 0
        self._bar: tqdm[Any] | None = None
        if show_bar:
            self._bar = tqdm(
                total=total,
                desc=label,
                unit="step",
                unit_scale=True,
                dynamic_ncols=True,
                smoothing=0.05,
                mininterval=1.0,
            )
        self._write_file(0)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def update(self, step: int, rolling_return: float | None, epsilon: float) -> None:
        """Record that ``step`` steps are done; cheap enough to call on every step."""
        finished = step == self._total
        if self._bar is not None and (step % self.BAR_EVERY == 0 or finished):
            self._bar.update(step - self._shown)
            self._shown = step
            score = "-" if rolling_return is None else f"{rolling_return:.1f}"
            self._bar.set_postfix_str(f"return {score}, eps {epsilon:.2f}", refresh=False)
        if step % self.FILE_EVERY == 0 or finished:
            self._write_file(step)

    def close(self) -> None:
        """Close the bar, leaving its last state on screen."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _write_file(self, step: int) -> None:
        if self._file is None:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        partial = self._file.with_suffix(self._file.suffix + ".tmp")
        partial.write_text(f"{step}\n", encoding="utf-8")
        os.replace(partial, self._file)


def steps_log_dir(total_timesteps: int) -> Path:
    """Return the default output folder for runs of a given length, such as logs/steps_500000.

    Runs of different lengths can't be compared, so each length gets its own folder.
    """
    return Path("logs") / f"steps_{total_timesteps}"


def resolve_device(requested: Literal["mps", "cpu"], log: Any) -> torch.device:
    """Return the requested device, falling back from mps to cpu if unavailable."""
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("mps_unavailable", fallback="cpu")
        return torch.device("cpu")
    return torch.device(requested)


def train(
    config: TrainConfig,
    device: torch.device,
    metrics: DualLogger,
    log: Any,
    env_factory: Callable[[], gym.Env[Any, Any]] = make_breakout_env,
    progress: RunProgress | None = None,
) -> DQNAgent:
    """Run one complete training run and return the trained agent.

    Every environment step, whether an automatic press or an agent decision, passes through
    the same loop body, so presses count toward the update and evaluation schedules.

    Raises:
        TrainingError: If the loss or a gradient becomes non-finite.
    """
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(config.seed)
    eval_seed = config.seed + config.eval_seed_offset
    eval_rng = np.random.default_rng(eval_seed)

    agent = DQNAgent(config.agent_config, device)
    window = NStepWindow(config.n_step, config.gamma)
    replay = ReplayBuffer(config.buffer_capacity)

    step = 0
    try:
        with (
            BreakoutGame(env_factory(), config.seed, auto_fire=config.auto_fire) as game,
            BreakoutGame(env_factory(), eval_seed, auto_fire=config.auto_fire) as eval_game,
        ):
            if config.saves_snapshots:
                save_snapshot(agent, config, step)
            observation = game.reset()
            episode = 0
            episode_return = 0.0
            episode_length = 0
            loss_sum = 0.0
            q_sum = 0.0
            pending_updates = 0

            while step < config.total_timesteps:
                pressed = game.needs_fire
                epsilon = config.epsilon_at(step)
                action = (
                    BreakoutGame.FIRE_ACTION if pressed else agent.act(observation, epsilon, rng)
                )
                result = game.step(action)
                step += 1
                episode_return += result.raw_reward
                episode_length += 1

                if not pressed:
                    reward = result.clipped_reward if config.clip_rewards else result.raw_reward
                    life_lost = result.life_lost and config.terminal_on_life_loss
                    for transition in window.push(
                        observation,
                        action,
                        reward,
                        result.observation,
                        terminated=result.terminated,
                        truncated=result.truncated,
                        life_lost=life_lost,
                    ):
                        replay.add(transition)
                observation = result.observation

                if result.game_over:
                    episode += 1
                    metrics.log_episode(
                        step, episode, episode_return, episode_length, config.epsilon_at(step)
                    )
                    episode_return = 0.0
                    episode_length = 0
                    observation = game.reset()

                if progress is not None:
                    progress.update(step, metrics.rolling_return, epsilon)

                if step > config.learning_starts and step % config.train_frequency == 0:
                    batch = replay.sample(config.batch_size, rng, device)
                    update = agent.update(batch, step)
                    loss_sum += update.td_loss
                    q_sum += update.mean_q
                    pending_updates += 1
                    if pending_updates == config.log_interval:
                        metrics.log_train_step(
                            step, loss_sum / pending_updates, q_sum / pending_updates
                        )
                        loss_sum = 0.0
                        q_sum = 0.0
                        pending_updates = 0

                if config.saves_snapshots and step % config.snapshot_interval == 0:
                    save_snapshot(agent, config, step)

                if step % config.eval_interval == 0:
                    evaluation = evaluate(
                        agent,
                        eval_game,
                        num_games=config.eval_games,
                        epsilon=config.eval_epsilon,
                        gamma=config.gamma,
                        rng=eval_rng,
                    )
                    if evaluation is None:
                        log.warning("evaluation_without_known_returns", step=step)
                    else:
                        metrics.log_eval(step, evaluation.predicted_q, evaluation.actual_return)
                        log.info(
                            "evaluation_finished",
                            step=step,
                            num_states=evaluation.num_states,
                        )
    except TrainingError:
        log.exception("run_diverged", step=step)
        raise
    except (BaseProjectError, OSError, RuntimeError, ValueError):
        log.exception("run_failed", step=step)
        raise
    return agent


def save_checkpoint(agent: DQNAgent, config: TrainConfig) -> Path:
    """Write the final model and return its path."""
    path = config.output_dir / "checkpoints" / f"{config.run_name}.pt"
    _save_model(agent, config, path, config.total_timesteps)
    return path


def save_snapshot(agent: DQNAgent, config: TrainConfig, step: int) -> Path:
    """Write the model as it is at ``step`` and return its path. Training is not affected."""
    path = config.output_dir / "checkpoints" / config.run_name / f"step_{step:06d}.pt"
    _save_model(agent, config, path, step)
    return path


def _save_model(agent: DQNAgent, config: TrainConfig, path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.cpu() for name, value in agent.online.state_dict().items()}
    torch.save({"online_state_dict": state, "config": _hyperparams(config), "step": step}, path)


def _hyperparams(config: TrainConfig) -> dict[str, Any]:
    """Return the config as plain values that W&B and checkpoints can store."""
    values = {**asdict(config), "log_dir": config.output_dir}
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def run(
    config: TrainConfig,
    env_factory: Callable[[], gym.Env[Any, Any]] = make_breakout_env,
) -> None:
    """Run one training run with its loggers. Failures are logged with a stack trace."""
    log = make_structured_logger(config.run_name, config.seed)
    device = resolve_device(config.device, log)
    hyperparams = {**_hyperparams(config), "resolved_device": str(device)}
    log.info("run_started", device=str(device), variant=config.variant)
    try:
        metrics = DualLogger(
            config.run_name, hyperparams, wandb_mode=config.wandb_mode, log_dir=config.output_dir
        )
    except (BaseProjectError, OSError, wandb.Error):
        log.exception("run_setup_failed")
        raise
    with (
        metrics,
        RunProgress(
            config.total_timesteps,
            config.run_name,
            config.progress_file,
            show_bar=sys.stderr.isatty(),
        ) as progress,
    ):
        agent = train(config, device, metrics, log, env_factory, progress)
        progress.close()
        try:
            checkpoint = save_checkpoint(agent, config)
        except OSError:
            log.exception("checkpoint_failed", step=config.total_timesteps)
            raise
        log.info(
            "run_finished",
            step=config.total_timesteps,
            final_return_100=metrics.rolling_return,
            checkpoint=str(checkpoint),
        )


def main() -> None:
    """Parse the CLI and start the run."""
    run(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
