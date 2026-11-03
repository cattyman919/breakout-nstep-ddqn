# Code guide

How the code is organized and how a training run works. For the reasoning behind the math see
[`ARCHITECTURE.md`](ARCHITECTURE.md), and for the study protocol see
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## The big picture

The project trains one agent per run on Breakout, using the game's 128 bytes of RAM as input.
The four variants share almost all their code and differ in only two settings:

| Variant | `double_q` | `n_step` |
|---|---|---|
| `dqn_n1` | `False` | 1 |
| `ddqn_n1` | `True` | 1 |
| `ddqn_n3` | `True` | 3 |
| `ddqn_n5` | `True` | 5 |

`TrainConfig` works out both settings from the variant name. There is no separate class per
algorithm: `double_q` changes how the agent computes targets, and `n_step` changes the size of
the window in front of the replay buffer.

## How the files fit together

```text
scripts/train.py            Runs the training loop and saves the final model
├── src/environment.py      Plays the game: lives, automatic FIRE, clipped rewards
├── src/buffers.py          Turns steps into n-step transitions and stores them
├── src/agents.py           Chooses actions and learns from sampled batches
│   └── src/networks.py     The Q-network and observation scaling
├── src/evaluation.py       Measures predicted values against returns earned
└── src/utils/logger.py     Writes metrics and JSON diagnostics

scripts/study.py            Trains many runs, several at a time
scripts/measure_random.py   Plays random games for the pilot baseline
scripts/plot_results.py     Reads the CSVs and draws the figures
scripts/record_videos.py    Records saved models playing the game
src/exceptions.py           Project error types, used throughout
```

## What happens in one training step

The loop is in `train()` in `scripts/train.py`. Each pass through it is exactly one environment
step:

1. **Choose an action.** If `game.needs_fire` is set, the action is FIRE. Otherwise the agent
   picks an $\epsilon$-greedy action.
2. **Step the game.** `step` goes up by one, and the raw reward is added to the episode's return.
3. **Store the experience.** For agent decisions only, the step goes into `NStepWindow`, and any
   transitions it completes go into `ReplayBuffer`. FIRE presses are never stored.
4. **Handle the end of a game.** On termination or truncation, the episode is logged and the game
   resets. A lost life does not reset the game.
5. **Learn.** If `step > learning_starts` and `step % train_frequency == 0`, the agent samples a
   batch and takes one gradient step. Loss and mean $Q$ are averaged over `log_interval` updates
   and then logged.
6. **Evaluate.** If `step % eval_interval == 0`, `evaluate()` plays complete games in a separate
   environment and logs the result.

FIRE presses go through this same loop on purpose. They count toward the update and evaluation
schedules, which is what makes a run produce exactly 122,500 updates and 20 evaluations.

## Modules

### `src/networks.py`

- **`QNetwork`:** an MLP with layers 128 → 256 → 256 → 4 and 99,844 parameters. It accepts one
  state or a batch, and raises `ValueError` for any other shape or for non-`float32` input.
- **`normalize_observation`:** converts RAM bytes to values from 0 to 1. It is the only place that
  divides by 255.

### `src/buffers.py`

**`NStepWindow`** builds transitions. `push()` takes one step and returns whichever transitions
that step completes, which may be none, one or several. It commits a transition when the window
is full, and commits everything left when a boundary is reached.

Here is an example with $n = 3$ and $\gamma = 0.5$, where the game ends on the fourth step and
every reward is 1:

| Starts at state | Next state | Span $m$ | Return | `done` |
|---|---|---|---|---|
| 0 | 3 | 3 | $1 + 0.5 + 0.25 = 1.75$ | `False` |
| 1 | 4 | 3 | $1 + 0.5 + 0.25 = 1.75$ | `True` |
| 2 | 4 | 2 | $1 + 0.5 = 1.5$ | `True` |
| 3 | 4 | 1 | $1$ | `True` |

Each step produces exactly one transition, and none crosses the boundary. `done` is set only by
termination or a lost life. If the game had been truncated instead, the same transitions would
be committed with `done=False`, so they would still bootstrap from the final observation.

**`ReplayBuffer`** is a fixed-size ring buffer. It stores states as `uint8` to save memory,
overwrites the oldest entry when full, and samples uniformly with replacement. `sample()` returns
a `TransitionBatch` of normalized tensors on the requested device.

### `src/agents.py`

**`DQNAgent`** owns the online network, the target network and the Adam optimizer.

- **`act`:** $\epsilon$-greedy action selection. It skips the network when exploring, which saves
  time during training.
- **`act_with_value`:** also returns $\max_a Q(s, a)$, which evaluation needs. The value is the
  greedy one even when the action chosen is random.
- **`compute_targets`:** calculates $G + (1 - d)\gamma^m\hat{Q}(s')$ without gradients. The stored
  span $m$ sets the discount, so short flushed windows are discounted correctly.
- **`update`:** runs one learning step in this order:
  1. Compute the targets and the Huber loss.
  2. Run the backward pass.
  3. Check the loss and gradients, raising `TrainingError` if any value is NaN or infinite.
  4. Step the optimizer.
  5. Move the target network toward the online network by $\tau$.

The check comes before the optimizer step, so a bad gradient can't corrupt the weights.

### `src/environment.py`

- **`make_breakout_env`:** the only place that creates the real environment.
- **`BreakoutGame`:** wraps any Gymnasium environment and is shared by training, evaluation and
  the random baseline, so all three follow the same rules. It:
  - seeds the environment on the first reset only
  - detects lost lives from `info["lives"]`
  - sets `needs_fire` after each reset and lost life
  - copies and validates every observation
  - returns a `StepResult` holding both the raw and clipped reward

`BreakoutGame` only says when a press is needed. The caller performs it, so the press counts as a
normal step in the caller's loop.

### `src/evaluation.py`

`evaluate()` plays complete games and records the predicted value of each state where the agent
chose an action. When a life ends, `discounted_returns()` calculates the return actually earned
from each of those states.

- FIRE presses add no state, and their rewards count toward no return.
- A life cut off by truncation alone is thrown away, because its returns are unknown.
- If no state has a known return, `evaluate()` returns `None` and the training loop logs a warning
  and writes no row.

Evaluation has its own environment and random generator, both seeded with `seed + 10,000`, so it
can't change the course of training. `tests/test_train.py` checks this by comparing final weights
with and without evaluation.

### `src/utils/logger.py`

The project has two separate logging systems:

| | `DualLogger` | `make_structured_logger` |
|---|---|---|
| Records | Experiment metrics | What the program is doing |
| Goes to | TensorBoard, W&B and CSV | stderr, as JSON |
| Examples | Episode return, TD loss, evaluation values | Run started, warnings, failures |

`DualLogger` refuses to overwrite an existing CSV, so a rerun can't destroy results. It is a
context manager, so its sinks are flushed and closed even when a run fails.

Each CSV row is one of three kinds, told apart by which columns are filled:

| Row | Filled columns, besides `step` |
|---|---|
| Episode | `episode`, `episodic_return`, `episode_length`, `epsilon` |
| Train | `mean_q_value`, `td_loss` |
| Evaluation | `eval_predicted_q`, `eval_actual_return` |

### `src/exceptions.py`

All project errors inherit from `BaseProjectError`:

| Exception | Raised when |
|---|---|
| `ConfigurationError` | A config value is invalid |
| `GameEnvironmentError` | The environment can't be created, or returns something unexpected |
| `TrainingError` | A loss or gradient is NaN or infinite |

Networks and buffers raise the built-in `ValueError` when a caller passes bad input.

## Scripts

### `scripts/train.py`

- **`steps_log_dir`:** returns the default results folder for a run length,
  `logs/steps_<total_timesteps>`. `TrainConfig.output_dir` uses it when `log_dir` isn't set, and
  `plot_results.py` and `record_videos.py` use it for their `--steps` option, so every script agrees
  on where a run length's results are. `{log_dir}` below means this folder.
- **`TrainConfig`:** holds every hyperparameter in a frozen dataclass, which `tyro` turns into
  command-line options. `__post_init__` rejects invalid values.
- **`train`:** the loop described above. Its `env_factory` argument lets tests swap in a scripted
  game.
- **`run`:** sets up the loggers, calls `train` and saves the checkpoint.
- **`save_checkpoint`:** writes the final model to `{log_dir}/checkpoints/{run_name}.pt`. It runs
  only after a successful run.
- **`RunProgress`:** shows a progress bar in a terminal and, when `progress_file` is set, keeps that
  file updated with the step count every 10,000 steps. The file is replaced atomically, so a reader
  never sees half a number. `progress_file` isn't a hyperparameter; `scripts/study.py` sets it.
- **`save_snapshot`:** writes the current model to
  `{log_dir}/checkpoints/{run_name}/step_{step}.pt`, at the start of a run and every
  `snapshot_interval` steps. Only runs whose seed is in `snapshot_seeds` save these (seed 1 by
  default). It only reads the weights, so it can't change training.

### `scripts/study.py`

Trains every combination of `--variants` and `--seeds`, up to `--jobs` runs at once. Any other option
is parsed as a `TrainConfig` and applied to every run, so invalid settings fail before training
starts.

- **`plan_runs`:** splits the runs into finished ones, which have a final model, and ones still to
  train.
- **`to_cli_args`:** turns a `TrainConfig` back into `scripts.train` options, so each process gets
  exactly the validated settings. A test checks that the options recreate the same config.
- **`StudyProgress`:** draws one bar per variant and one for the whole study, reading each running
  run's progress file, `{log_dir}/run_logs/{run_name}.progress`, every 2 seconds. A failed run's
  remaining steps leave its variant's and the study's totals, so the time estimates stay right.
- **`run_study`:** runs `train_in_subprocess` for each pending run from a thread pool, and collects
  which runs finished or failed and how long each took. Tests pass in a fake training function and
  clock in their place.
- **`timing_summary`:** builds the end-of-study table of training time per variant and in total.
  `append_timing` records each run's time in `{log_dir}/run_logs/timings.csv` as it ends.

Each run is a separate process using one CPU core, so parallel runs can't affect each other's
results.

### `scripts/measure_random.py`

Plays 100 games with a uniform random policy and automatic FIRE, then prints the mean return and
the pilot threshold of double that mean.

### `scripts/plot_results.py`

Finds the run CSVs, groups them by variant and draws two figures:

- **`ablation_comparison.png`:** episodic return and mean predicted $Q$ over training.
- **`overestimation_evaluation.png`:** one panel per variant, comparing predicted values with
  returns earned.

Seeds are averaged only over the steps every run reached.

### `scripts/record_videos.py`

Makes one MP4 per saved model, for one seed of each variant, and writes them to
`{log_dir}/videos/{run_name}/`. No window is needed, because frames come directly from the
environment with `render_mode="rgb_array"`.

- **`find_clips`:** lists the snapshots in step order, then the final model.
- **`record_clip`:** loads a model and plays one game through `BreakoutGame`, with the same FIRE
  rules and 1% random move rate as evaluation. It stops at game over or the `max_seconds` limit.
- **`record_all`:** gives each clip its own game seed. The last snapshot and the final model hold
  the same weights, so this is why `step_500000.mp4` and `final.mp4` show different games.

Videos are built only from saved files, so they can be re-recorded at any time.

## Tests

The tests in `tests/` run in a few seconds and need no W&B network access. Most use `ScriptedEnv`
from `tests/conftest.py`, a fake game that replays a fixed list of rewards, lives and endings. That
makes rare cases such as truncation easy to trigger and exact results easy to check.

| Test file | Covers |
|---|---|
| `test_networks.py` | Output shapes, parameter count, input validation |
| `test_buffers.py` | N-step windows and flushes for $n = 1, 3, 5$, replay overflow |
| `test_agents.py` | DQN and Double DQN targets by hand, Polyak updates, NaN handling |
| `test_environment.py` | FIRE, lives, reward clipping, seeding, the real game's spaces |
| `test_evaluation.py` | Returns stopping at a lost life, truncated lives excluded |
| `test_logger.py` | CSV rows, overwrite protection, JSON diagnostics |
| `test_train.py` | Schedule counts, replay contents, reproducibility, failure cleanup |
| `test_measure_random.py` | Baseline returns and FIRE presses |
| `test_plot_results.py` | Run discovery, seed averaging, figure output |
| `test_study.py` | Skipping finished runs, the parallel limit, failures, option passing |
| `test_record_videos.py` | Finding models, frame counts, the length limit, game seeds |

## Where to make common changes

| To change | Edit |
|---|---|
| A hyperparameter | `TrainConfig` in `scripts/train.py`, and the tables in `EXPERIMENTS.md` |
| The network's layer sizes | `NetworkConfig` in `src/networks.py` |
| The target calculation | `DQNAgent.compute_targets` in `src/agents.py` |
| How transitions are built | `NStepWindow` in `src/buffers.py` |
| Episode rules | `BreakoutGame` in `src/environment.py` |
| A logged metric | `DualLogger` in `src/utils/logger.py`, then `plot_results.py` if it should be plotted |

The settings are frozen now that the pilot has passed, so changing a shared setting means following
`EXPERIMENTS.md` section 5 again.
