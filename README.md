# breakout-nstep-ddqn

A reinforcement learning study of whether **n-step returns improve Double DQN** on Atari
Breakout, using the game's 128 bytes of RAM as the observation instead of pixels.

> **Status: implemented, and the pilot passed. The 20-run study has not been run yet.**

## Contents

- [What it compares](#what-it-compares)
- [Quick start](#quick-start)
- [Step 1: Set up](#step-1-set-up)
- [Step 2: Run the tests](#step-2-run-the-tests)
- [Step 3: Do a short trial run](#step-3-do-a-short-trial-run)
- [Step 4: Run the pilot (optional)](#step-4-run-the-pilot-optional)
- [Step 5: Train the full study](#step-5-train-the-full-study)
- [Step 6: Watch progress](#step-6-watch-progress)
- [Step 7: Make the figures](#step-7-make-the-figures)
- [Step 8: Record the videos](#step-8-record-the-videos)
- [Output files](#output-files)
- [Command options](#command-options)
- [Troubleshooting](#troubleshooting)
- [Reporting the results](#reporting-the-results)
- [Development checks](#development-checks)
- [Project layout](#project-layout)
- [Further reading](#further-reading)

## What it compares

| Variant | Algorithm | n | What it tests |
|---|---|---|---|
| `dqn_n1` | DQN | 1 | Baseline |
| `ddqn_n1` | Double DQN | 1 | Effect of Double Q-learning |
| `ddqn_n3` | Double DQN | 3 | Effect of n-step returns |
| `ddqn_n5` | Double DQN | 5 | Effect of longer n-step returns |

Each variant trains for 500,000 steps with 5 seeds, for 20 runs in total. All agents share the
same small network: 128 inputs, two hidden layers of 256 units, and 4 outputs.

The results are compared on three measures:

- **Episodic return:** how well each agent plays, in raw game points.
- **Mean predicted Q-value:** whether Double DQN predicts lower values than DQN.
- **Predicted vs. actual value:** every 25,000 steps, evaluation games compare what each agent
  predicts with the discounted return it actually earns. This is the direct test of
  overestimation.

## Quick start

Run these from the project root to go from a fresh checkout to figures and videos:

```bash
uv sync
uv run python -m pytest

for variant in dqn_n1 ddqn_n1 ddqn_n3 ddqn_n5; do
  for seed in 1 2 3 4 5; do
    uv run python -m scripts.train --variant "$variant" --seed "$seed" --device cpu --wandb-mode offline
  done
done

uv run python -m scripts.plot_results
uv run python -m scripts.record_videos
```

The steps below explain each command, what to expect from it, and what to do if it fails.

## Step 1: Set up

**You need:**

- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and all dependencies
- A Mac with Apple Silicon, which is the target hardware. Other machines can run on CPU, but only
  the Mac has been tested
- About 100 MB of free disk space for the study's outputs, on top of the dependencies
- A [Weights & Biases](https://wandb.ai) account, only if you want to sync results online

**Install the dependencies:**

```bash
uv sync
```

**Choose how to use W&B.** Offline is simpler and is what the commands in this guide use:

| Mode | Setup | How to use it |
|---|---|---|
| Offline | None | Add `--wandb-mode offline` to training commands. You can upload afterward |
| Online | Run `uv run wandb login` | Leave out `--wandb-mode`, because online is the default |

Online sync hasn't been tested on this project. All the runs so far used offline mode.

Run every command from the project root, in the form `python -m scripts.<name>`. Running
`python scripts/<name>.py` causes import errors.

## Step 2: Run the tests

```bash
uv run python -m pytest
```

This takes about 10 seconds and needs no network access or W&B account. Every test should pass.

To run part of the suite:

```bash
uv run python -m pytest tests/test_buffers.py   # one file
uv run python -m pytest -k polyak               # tests whose names match a word
uv run python -m pytest -v                      # list each test as it runs
```

[`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) lists what each test file covers.

## Step 3: Do a short trial run

Before starting the study, you can try the whole workflow on a short run. This takes less than a
minute and keeps its files in `logs/smoke`, apart from any real results:

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu --wandb-mode offline \
  --log-dir logs/smoke --total-timesteps 30000 --learning-starts 1000 \
  --eval-interval 10000 --eval-games 1 --snapshot-interval 10000

uv run python -m scripts.plot_results --csv-dir logs/smoke/csv \
  --output logs/smoke/comparison.png --overestimation-output logs/smoke/overestimation.png

uv run python -m scripts.record_videos --variants ddqn_n3 --log-dir logs/smoke
```

The figures will say "No data" for the variants you didn't train, which is expected here. Delete
the trial when you're done:

```bash
rm -rf logs/smoke
```

## Step 4: Run the pilot (optional)

The pilot checks that the agents learn before the settings freeze. **It has already passed, so you
can skip to step 5.** The recorded result was:

| Random-policy return | Pass mark (double) | `ddqn_n3` return | `dqn_n1` return | Result |
|---|---|---|---|---|
| 1.29 | 2.58 | 17.04 | 15.45 | Pass |

To repeat it, clear the old results first, because runs won't overwrite existing ones:

```bash
rm -rf logs/pilot
uv run python -m scripts.measure_random
uv run python -m scripts.train --variant ddqn_n3 --seed 0 --device cpu --wandb-mode offline --log-dir logs/pilot
uv run python -m scripts.train --variant dqn_n1 --seed 0 --device cpu --wandb-mode offline --log-dir logs/pilot
```

Each pilot run takes about 3 minutes and prints its final return as `final_return_100` in the
`run_finished` line. If a pilot falls short, follow
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 5.

## Step 5: Train the full study

```bash
mkdir -p logs
for variant in dqn_n1 ddqn_n1 ddqn_n3 ddqn_n5; do
  for seed in 1 2 3 4 5; do
    uv run python -m scripts.train --variant "$variant" --seed "$seed" --device cpu --wandb-mode offline 2>&1 | tee -a logs/study.log
  done
done
```

**How long it takes:** each run took about 3 minutes in the pilot on an M4 MacBook Air, so all 20
should take a little over an hour. The runs execute one at a time.

**Keep the Mac awake:** in a second terminal, run `caffeinate -i` until training finishes, then
stop it with `Ctrl+C`. This prevents idle sleep, but closing the lid can still put the Mac to sleep.

**What you'll see:** one JSON line for each diagnostic event, such as `run_started`,
`evaluation_finished` and `run_finished`, mixed with plain text from W&B and the game emulator.
`tee` also saves everything to `logs/study.log`.

**Keep these rules for the study:**

- Use `--device cpu` for every run. The default is `mps`, but CPU was about 4 times faster per
  update for this small network.
- Don't change any other settings. They're frozen now that the pilot has passed, and changing one
  means repeating the pilot.
- Seed 0 is reserved for pilots. The study uses seeds 1 to 5.

**Check that every run finished:**

```bash
ls logs/checkpoints/*.pt | wc -l
```

This should print 20, because a final model is saved only when a run succeeds. If a run stopped
early, the loop carries on with the next one, so check `logs/study.log` and see
[Troubleshooting](#troubleshooting).

**To train a single run:**

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu --wandb-mode offline
```

**To upload offline runs to W&B afterward:**

```bash
uv run wandb sync logs/wandb/offline-run-*
```

## Step 6: Watch progress

In a second terminal, start TensorBoard and open the address it prints, usually
<http://localhost:6006>:

```bash
uv run tensorboard --logdir logs/tensorboard
```

| Chart | What it shows | What to look for |
|---|---|---|
| `episode/return_rolling_100` | Mean game score over the last 100 games | Should rise over training |
| `episode/length` | Steps per game | Long games with low scores can mean the ball isn't launching |
| `train/td_loss` | How far predictions are from their targets | Should stay small, with no sudden blow-up |
| `train/mean_q_value` | The value the agent expects | Should level off, with no runaway growth |
| `eval/predicted_q`, `eval/actual_return` | Predicted vs. earned value | A persistent gap above the actual return suggests overestimation |
| `episode/epsilon` | Exploration rate | Falls to 0.05 by step 50,000, then stays there |

Training doesn't open a game window, because one would limit it to about 14 steps per second. To
see the agents play, use the videos in step 8.

## Step 7: Make the figures

```bash
uv run python -m scripts.plot_results
```

This reads `logs/csv/`, prints a summary table and writes two figures:

| Figure | What it shows |
|---|---|
| `docs/figures/ablation_comparison.png` | Panel A: episodic return. Panel B: mean predicted Q-value |
| `docs/figures/overestimation_evaluation.png` | Predicted value against actual return, one panel per variant |

Check the `last step` column in the summary table. Every variant should be close to 500,000. A much
lower number means a run stopped early and cut that variant's curves short.

Git ignores PNG files, so to commit the figures use `git add -f`:

```bash
git add -f docs/figures/ablation_comparison.png docs/figures/overestimation_evaluation.png
```

## Step 8: Record the videos

```bash
uv run python -m scripts.record_videos
```

This takes seed 1 of each variant and records one game per saved model, with no window and no
retraining. It should take less than a minute in total and print a table of each video's length and
score. You get 7 videos per variant, 28 in total:

```text
logs/videos/ddqn_n3_seed1/
├── step_000000.mp4   Untrained
├── step_100000.mp4
├── step_200000.mp4
├── step_300000.mp4
├── step_400000.mp4
├── step_500000.mp4
└── final.mp4         The same model as step_500000, playing a different game
```

**Reading the videos:**

- **Scores vary:** each video is a single game, so a later phase can score lower than an earlier
  one. Use the videos to illustrate play, and the figures to compare variants.
- **Double speed:** each video frame is one agent step, so the default 30 frames per second plays
  at twice the real game speed.
- **Length limit:** a game is cut off at 3 minutes, and the summary table says if one was.
- **Random moves:** the model makes a random move 1% of the time, the same as evaluation, so it
  can't get stuck forever.

**Common variations:**

```bash
uv run python -m scripts.record_videos --fps 15                       # real game speed
uv run python -m scripts.record_videos --variants ddqn_n3 ddqn_n5     # chosen variants only
uv run python -m scripts.record_videos --seed 0 --log-dir logs/pilot  # the pilot's final models
```

At `--fps 15` the 3-minute limit covers half as many game steps, so raise `--max-seconds` if a game
gets cut off. Pilot runs have no snapshots, so that last command records only `final.mp4`, and it
warns about variants without a pilot.

To capture other seeds or phases, set this **before training**. Snapshots can't be added afterward:

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 2 --device cpu --wandb-mode offline \
  --snapshot-seeds 1 2 --snapshot-interval 50000
```

Saving snapshots doesn't change learning. A repeat of the `ddqn_n3` seed 0 pilot with snapshots on
gave a byte-identical metrics CSV.

## Output files

| Output | Location |
|---|---|
| Raw metrics | `logs/csv/{variant}_seed{seed}_metrics.csv` |
| TensorBoard | `logs/tensorboard/{variant}_seed{seed}/` |
| W&B | Project `breakout-nstep-ddqn`, with local files in `logs/wandb/` |
| Final model | `logs/checkpoints/{variant}_seed{seed}.pt` |
| Model snapshots (seed 1) | `logs/checkpoints/{variant}_seed1/step_{step}.pt` |
| Videos (seed 1) | `logs/videos/{variant}_seed1/` |
| Training log, if you use `tee` | `logs/study.log` |
| Pilot runs | `logs/pilot/` |
| Figures | `docs/figures/` |

Git ignores `logs/`, `*.csv`, `*.pt` and `*.png`, so keep copies of any results you need.

Each CSV row is an episode, training or evaluation row, and columns that don't apply are empty:

| Row | Filled columns, besides `step` |
|---|---|
| Episode | `episode`, `episodic_return`, `episode_length`, `epsilon` |
| Train (one per 100 updates) | `mean_q_value`, `td_loss` |
| Evaluation (one per 25,000 steps) | `eval_predicted_q`, `eval_actual_return` |

## Command options

Add `--help` to any script to see all its options, such as
`uv run python -m scripts.train --help`.

### `scripts.train`

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `ddqn_n3` | `dqn_n1`, `ddqn_n1`, `ddqn_n3` or `ddqn_n5` |
| `--seed` | `1` | Random seed |
| `--device` | `mps` | `mps` or `cpu`. Use `cpu` for the study |
| `--wandb-mode` | `online` | `online` or `offline` |
| `--log-dir` | `logs` | Root folder for this run's outputs |
| `--snapshot-seeds` | `1` | Seeds whose runs save snapshots |
| `--snapshot-interval` | `100000` | Steps between snapshots |

These are the frozen study settings. Change them only for trial runs or a new pilot round:

| Option | Default | Option | Default |
|---|---|---|---|
| `--total-timesteps` | `500000` | `--epsilon-start` | `1.0` |
| `--learning-rate` | `0.0001` | `--epsilon-end` | `0.05` |
| `--batch-size` | `64` | `--epsilon-decay-fraction` | `0.1` |
| `--buffer-capacity` | `100000` | `--learning-starts` | `10000` |
| `--gamma` | `0.99` | `--train-frequency` | `4` |
| `--tau` | `0.005` | `--huber-delta` | `1.0` |
| `--log-interval` | `100` | `--eval-interval` | `25000` |
| `--eval-games` | `5` | `--eval-epsilon` | `0.01` |
| `--clip-rewards` | on | `--terminal-on-life-loss` | on |
| `--auto-fire` | on | `--eval-seed-offset` | `10000` |

### `scripts.record_videos`

| Option | Default | Meaning |
|---|---|---|
| `--variants` | All four | Variants to record |
| `--seed` | `1` | Which seed's models to record |
| `--log-dir` | `logs` | Folder holding `checkpoints/`. Videos go in `videos/` beneath it |
| `--max-seconds` | `180` | Longest video |
| `--fps` | `30` | Frames per second. `15` is real game speed |
| `--epsilon` | `0.01` | Random move rate |
| `--upscale` | `3` | Enlargement of the game picture |
| `--game-seed-offset` | `20000` | Added to the seed for the recorded games |

### `scripts.plot_results`

| Option | Default | Meaning |
|---|---|---|
| `--csv-dir` | `logs/csv` | Folder holding the run CSVs |
| `--output` | `docs/figures/ablation_comparison.png` | Comparison figure path |
| `--overestimation-output` | `docs/figures/overestimation_evaluation.png` | Overestimation figure path |
| `--window` | `100` | Rolling window, in episodes |
| `--min-periods` | `10` | Fewest episodes needed to draw a point |
| `--grid-points` | `500` | Points per curve |
| `--dpi` | `300` | Image resolution |

### `scripts.measure_random`

| Option | Default | Meaning |
|---|---|---|
| `--games` | `100` | Games to play |
| `--seed` | `0` | Random seed |

## Troubleshooting

### `FileExistsError` when starting a run

Runs won't overwrite existing results. If you do want to redo a run, delete its outputs first:

```bash
RUN=ddqn_n3_seed1
rm -rf "logs/csv/${RUN}_metrics.csv" "logs/tensorboard/$RUN" "logs/checkpoints/$RUN.pt" "logs/checkpoints/$RUN"
```

### A run crashes or is interrupted

Runs can't resume. Delete the run's outputs using the command above, then restart it from step 0.

### A run stops with `TrainingError`

This means the loss or a gradient became NaN or infinite. **Don't simply rerun it**, because
divergence may be a real result for that variant:

- **If it looks like real divergence:** keep its logs and report it as a diverged run.
- **If it looks like a bug:** fix the bug, then rerun every run it affected.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 6. A diverged run also limits that
variant's curves to the step where it failed, which you can see in the `last step` column.

### Other problems

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root, as `python -m scripts.train` |
| Training is slow | Use `--device cpu`. Also check that no game window is open |
| `no_models_found` or `no_videos_recorded` from `record_videos` | Train first, or check `--seed` and `--log-dir`. Models must be at `{log-dir}/checkpoints/` |
| `record_videos` makes only `final.mp4` | That run had no snapshots. Only seeds in `--snapshot-seeds` save them, and only during training |
| `no_recognized_runs` from `plot_results` | No CSVs in `--csv-dir`. Pilot results are in `logs/pilot/csv`, not `logs/csv` |
| `skipping_unrecognized_run` from `plot_results` | CSV names must look like `{variant}_seed{N}_metrics.csv` |
| `ConfigurationError` | An option has an invalid value. The message says which one |
| `wandb` asks you to log in | Add `--wandb-mode offline`, or run `uv run wandb login` |
| `NameNotFound` for `ALE/Breakout-ram-v5` | Current `ale-py` removed that ID. Use `gym.make("ALE/Breakout-v5", obs_type="ram")` |

## Reporting the results

These settings change what the agents learn, so results aren't directly comparable with studies
that don't use them:

- Training targets use the sign of each reward, while logged returns stay in raw points.
- Losing a life ends the training target, but the game continues until game over.
- FIRE is pressed automatically after each reset and lost life.

Because of the first two, Q-values mean discounted clipped return until the next lost life, so they
can't be read directly against game scores.

Keep these in mind too:

- **Overestimation claims:** base them on `overestimation_evaluation.png`, not on the mean Q panel.
  A better policy can legitimately have higher values.
- **Seeds:** five seeds show spread but can't support strong significance claims, so report results
  as observed differences.
- **Device:** report the device used, because MPS results can differ from CPU results.

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 8 covers interpretation in full.

## Development checks

Run these before committing a change:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --with mypy mypy --strict --ignore-missing-imports src scripts
uv run python -m pytest
```

To apply fixes automatically:

```bash
uv run ruff check --fix .
uv run ruff format .
```

[`AGENTS.md`](AGENTS.md) has the full coding standards and commit rules.

## Project layout

```text
├── AGENTS.md              Instructions and conventions for AI coding agents
├── README.md
├── pyproject.toml
├── docs/
│   ├── ARCHITECTURE.md    MDP, network, target equations, n-step rules
│   ├── CODE_GUIDE.md      How the code is organized and how a run works
│   └── EXPERIMENTS.md     Ablation matrix, hyperparameters, metrics, caveats
├── src/
│   ├── exceptions.py      Project exception types
│   ├── networks.py        Q-network
│   ├── buffers.py         N-step window and replay buffer
│   ├── agents.py          DQN and Double DQN agent
│   ├── environment.py     Game session: lives, automatic FIRE, clipped rewards
│   ├── evaluation.py      Predicted values against earned returns
│   └── utils/logger.py    TensorBoard, W&B and CSV logging; JSON diagnostics
├── scripts/
│   ├── train.py           Config, CLI and training loop
│   ├── measure_random.py  Random-policy baseline
│   ├── plot_results.py    Comparison and overestimation figures
│   └── record_videos.py   Videos of saved models playing
├── tests/                 Unit and integration tests
└── logs/                  Run outputs, ignored by git
```

## Further reading

- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md): how the code is organized and how a run works
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the math and implementation rules
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): the full protocol and how to interpret results
- [`AGENTS.md`](AGENTS.md): coding and engineering standards, and the checks to run before finishing
  a change
