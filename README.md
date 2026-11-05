# breakout-nstep-ddqn

A reinforcement learning study of whether **n-step returns improve Double DQN** on Atari
Breakout, using the game's 128 bytes of RAM as the observation instead of pixels.

> **Status: implemented, and the pilot passed. The `make` commands train for 12 million steps per
> run, with a snapshot every 2 million.** See [Run length and results folders](#run-length-and-results-folders).

## Run length and results folders

Every run length gets its own results folder, `logs/steps_<STEPS>`, because runs of different
lengths can't be compared. With the `make` defaults, everything goes in `logs/steps_12000000`:

| Setting | `make` default | Set it with |
|---|---|---|
| Steps per run | 12,000,000 | `STEPS=500000` |
| Steps between snapshots | 2,000,000 | `SNAPSHOT_INTERVAL=100000` |
| Results folder | `logs/steps_12000000` | Follows `STEPS`, or set `LOG_DIR` |

Pass the same `STEPS` to every command for that run length, such as `make status STEPS=500000`.
The paths in this README show the `make` default folder.

The training script's own default, and the protocol in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md),
are still 500,000 steps; only the `make` defaults are 12 million. A 500,000-step study was run, with
results in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and its files have since been deleted.

## Contents

- [Run length and results folders](#run-length-and-results-folders)
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
- [Training for longer](#training-for-longer)
- [Parallel training](#parallel-training)
- [Output files](#output-files)
- [Make commands](#make-commands)
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

Every task has a `make` command. Run these from the project root to go from a fresh checkout to
figures and videos:

```bash
make setup     # install everything
make test      # check the code works
make study     # train all 20 runs for 12 million steps each, many hours
make status    # confirm that all 20 finished
make figures   # make the comparison figures
make videos    # record the learning-phase videos
```

To list every command, run:

```bash
make help
```

Plain `make` and `make usage` show the same list. It's in color in a terminal, and plain text when
piped to a file or when the `NO_COLOR` environment variable is set. `make --help` shows GNU Make's own help instead,
because Make handles that flag itself before reading the project's Makefile.

To see every option a command accepts, pass `ARGS=--help`, as in `make train ARGS=--help`.

The steps below explain each command, what to expect from it, and what to do if it fails.

## Step 1: Set up

**You need:**

- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and all dependencies
- `make`, which macOS includes with the Xcode Command Line Tools (`xcode-select --install`)
- A Mac with Apple Silicon, which is the target hardware. Other machines can run on CPU, but only
  the Mac has been tested
- About 100 MB of free disk space for the study's outputs, on top of the dependencies
- A [Weights & Biases](https://wandb.ai) account, only if you want to sync results online

**Install the dependencies:**

```bash
make setup
```

**Choose how to use W&B.** Offline is the default and needs no account:

| Mode | Setup | How to use it |
|---|---|---|
| Offline | None | Nothing to add. Upload afterward with `make sync` if you want to |
| Online | Run `make wandb-login` | Add `WANDB=online`, as in `make study WANDB=online` |

Online sync hasn't been tested on this project. All the runs so far used offline mode.

Run every `make` command from the project root.

## Step 2: Run the tests

```bash
make test
```

This takes about 10 seconds and needs no network access or W&B account. Every test should pass.

To run part of the suite:

```bash
make test ARGS=tests/test_buffers.py   # one file
make test ARGS="-k polyak"             # tests whose names match a word
make test ARGS=-v                      # list each test as it runs
```

[`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) lists what each test file covers.

## Step 3: Do a short trial run

Before starting the study, you can try the whole workflow on a short run:

```bash
make trial
```

This trains for 30,000 steps, then makes figures and videos, all in less than a minute. Its files go
in `logs/smoke`, apart from any real results, and running it again replaces them. The figures say
"No data" for the variants it didn't train, which is expected.

Delete the trial when you're done:

```bash
make trial-clean
```

## Step 4: Run the pilot (optional)

The pilot checks that the agents learn before the settings freeze. **It has already passed, so you
can skip to step 5.** The recorded result was:

| Random-policy return | Pass mark (double) | `ddqn_n3` return | `dqn_n1` return | Result |
|---|---|---|---|---|
| 1.29 | 2.58 | 17.04 | 15.45 | Pass |

To repeat it, clear the old results first, because runs won't overwrite existing ones:

```bash
make pilot-clean
make pilot
```

This measures the random-policy baseline, then runs both pilots on seed 0, in about 6 minutes. Each
run prints its final return as `final_return_100` in the `run_finished` line. If a pilot falls short,
follow [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 5.

To measure only the baseline, run `make baseline`.

## Step 5: Train the full study

```bash
make study
```

**How long it takes:** run one at a time, the 20 runs took 74 minutes on an M4 MacBook Air, about
3.5 minutes each.

**Training several runs at once:** `make study` trains 4 runs at a time by default. Set `JOBS` to
change that, for example `make study JOBS=1` to train one at a time.

Each run uses one CPU core, so 4 matches the M4's 4 performance cores. Results are identical either
way: each run's metrics come out byte for byte the same as when trained alone. Only the speed
changes. See [Parallel training](#parallel-training).

**Keep the Mac awake:** in a second terminal, run `caffeinate -i` until training finishes, then
stop it with `Ctrl+C`. This prevents idle sleep, but closing the lid can still put the Mac to sleep.

**What you'll see:** one line when each run starts and one when it finishes or fails, progress bars
at the bottom of the terminal, and a timing summary at the end. There's one bar per variant, plus a
bar for the whole study. The times below are an illustration:

```text
21:32:34 Finished dqn_n1_seed3 in 1h 58m
dqn_n1 :  84%|█████████████████▌   | 50.4M/60.0M [5:51:10<1:07:02, ... 3/5 done, seed 4: 62% seed 5: 57%]
ddqn_n1:   8%|█▋                   | 4.80M/60.0M [0:25:12<4:49:39, ... 0/5 done, seed 1: 21% seed 2: 19%]
ddqn_n3:   0%|                     | 0.00/60.0M [00:00<?, ?step/s, 0/5 done]
ddqn_n5:   0%|                     | 0.00/60.0M [00:00<?, ?step/s, 0/5 done]
study  :  23%|████▌                | 55.2M/240M [6:16:22<21:01:40, ... 3/20 done, 4 running]
```

A variant's bar counts the steps of all its seeds, and the study bar counts every run, so each time
estimate covers that whole set. After the speed, each bar shows how many runs are done and the
percentage of each seed that's training. A variant's estimate is rough until it has started, since
the runs train in variant order. Each run's full output goes to its own file in
`logs/steps_12000000/run_logs/`, such as `ddqn_n3_seed1.log`, so parallel runs never mix their output.

`make train` shows a bar for its one run instead, with its current 100-game average score and
exploration rate:

```text
ddqn_n5_seed1:  50%|██████████          | 6.00M/12.0M [58:12<58:12, 1.72kstep/s, return 87.3, eps 0.05]
```

The bars appear only in a terminal. When output goes to a file, such as a run's log, they're left
out so the file stays readable. They don't affect training or results.

**If it stops partway:** run `make study` again. It skips runs that already finished. A run that
stopped partway won't restart by itself, because its results are protected from being overwritten.
`make study` carries on with the other runs, marks the failed ones as `FAILED` with the path to
their log, and exits with an error. See [Troubleshooting](#troubleshooting) for what to do with them.

**Keep these rules for the study:**

- Keep the default `DEVICE=cpu`, which was about 4 times faster per update than `mps` for this small
  network.
- Don't change any other settings. They're frozen now that the pilot has passed, and changing one
  means repeating the pilot.
- Seed 0 is reserved for pilots. The study uses seeds 1 to 5.

**Check that every run finished:**

```bash
make status
```

This shows each run as finished, incomplete or not started, and should end with
`20 of 20 runs finished.`

**To train a single run:**

```bash
make train VARIANT=ddqn_n3 SEED=1
```

**To train only some runs:**

```bash
make study VARIANTS="ddqn_n3 ddqn_n5" SEEDS="1 2"
```

**To upload offline runs to W&B afterward:**

```bash
make sync
```

## Step 6: Watch progress

In a second terminal, start TensorBoard and open <http://localhost:6006>:

```bash
make tensorboard
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
make figures
```

This reads `logs/steps_12000000/csv/`, prints a summary table and writes two figures to
`logs/steps_12000000/figures/`:

| Figure | What it shows |
|---|---|
| `ablation_comparison.png` | Panel A: episodic return. Panel B: mean predicted Q-value |
| `overestimation_evaluation.png` | Predicted value against actual return, one panel per variant |

Check the `last step` column in the summary table. Every variant should be close to the run length,
12,000,000 by default. A much lower number means a run stopped early and cut that variant's curves
short.

Git ignores `logs/` and PNG files. To commit the figures for a report, copy them into
`docs/figures/` and add them with `git add -f`:

```bash
mkdir -p docs/figures
cp logs/steps_12000000/figures/*.png docs/figures/
git add -f docs/figures/ablation_comparison.png docs/figures/overestimation_evaluation.png
```

## Step 8: Record the videos

```bash
make videos
```

This takes seed 1 of each variant and records one game per saved model, with no window and no
retraining. It prints a table of each video's length and score. With the defaults you get 8 videos
per variant, 32 in total:

```text
logs/steps_12000000/videos/ddqn_n3_seed1/
├── step_000000.mp4   Untrained
├── step_2000000.mp4
├── step_4000000.mp4
├── step_6000000.mp4
├── step_8000000.mp4
├── step_10000000.mp4
├── step_12000000.mp4
└── final.mp4         The same model as step_12000000, playing a different game
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
make videos ARGS="--fps 15"                 # real game speed
make videos VARIANTS="ddqn_n3 ddqn_n5"      # chosen variants only
make videos SEED=0 LOG_DIR=logs/pilot       # the pilot's final models
```

At `--fps 15` the 3-minute limit covers half as many game steps, so raise it with
`ARGS="--fps 15 --max-seconds 360"` if a game gets cut off. Pilot runs have no snapshots, so the last
command records only `final.mp4`, and it warns about variants without a pilot.

To capture other seeds or phases, set this **before training**. Snapshots can't be added afterward:

```bash
make train VARIANT=ddqn_n3 SEED=2 SNAPSHOT_INTERVAL=1000000 ARGS="--snapshot-seeds 1 2"
```

Saving snapshots doesn't change learning. A repeat of the `ddqn_n3` seed 0 pilot with snapshots on
gave a byte-identical metrics CSV.

## Training for longer

`STEPS` sets the length of every run, and `SNAPSHOT_INTERVAL` how often a model snapshot is saved:

```bash
make study STEPS=5000000 SNAPSHOT_INTERVAL=500000
```

**Where the results go:** each `STEPS` value has its own folder, `logs/steps_<STEPS>`, such as
`logs/steps_5000000`. Runs of different lengths never mix, and `make study` only skips runs of the
same length that already finished. Pass the same `STEPS` to the other commands so they read that
folder:

```bash
make status STEPS=5000000
make videos STEPS=5000000
make figures STEPS=5000000   # writes to logs/steps_5000000/figures
make tensorboard STEPS=5000000
```

**How long it takes:** the 500,000-step study's runs took 3.2 to 4.8 minutes each, 74 minutes for
all 20, one at a time. Time
grows roughly in line with `STEPS`, so expect:

| `STEPS` | One run | All 20 runs |
|---|---|---|
| 500,000 | About 3.5 minutes | About 75 minutes |
| 2,000,000 | About 15 minutes | About 5 hours |
| 5,000,000 | About 50 minutes | About 17 hours |
| 12,000,000 (the `make` default) | About 2 hours or more | About 40 hours |

These are one-at-a-time times. A 4.25-million-step test run took 41 minutes. Later training is
likely slower per step, because better agents play longer evaluation games. With the default
`JOBS=4`, all 20 runs finish in roughly a quarter of the one-at-a-time time, plus some slowdown
from sharing the machine: plan on at least 10 hours for 12 million steps. The parallel speedup
hasn't been measured on full-length runs yet.

At 12 million steps the outputs are larger too, about 1 GB for all 20 runs, mostly W&B and
TensorBoard logs. With `SNAPSHOT_INTERVAL=2000000` each seed 1 run saves 7 snapshots, at 0, 2, 4, 6,
8, 10 and 12 million steps, so `make videos` makes 8 videos per variant, including `final.mp4`.

**What changes with `STEPS`:**

- **Exploration:** the random-move rate still decays over the first 10% of training, so at 5 million
  steps it reaches 0.05 at step 500,000 instead of 50,000.
- **Logging:** evaluation still runs every 25,000 steps, so 5 million steps gives 200 evaluation
  points.
- **Snapshots:** one at the start plus one every `SNAPSHOT_INTERVAL` steps, so the number of videos
  is `STEPS / SNAPSHOT_INTERVAL + 2`, counting `final.mp4`. Keep it to about 10 or fewer by raising
  `SNAPSHOT_INTERVAL` with `STEPS`.

**Comparing results:** runs of different lengths can't be compared directly. Compare the four
variants within one run length, and report each length separately.

**Winning the game:** clearing both walls is worth 864 points. At 500,000 steps the best variant
averaged about 22 points per game, and a single `ddqn_n5` run averaged about 90 at 4 million steps,
with a best game of 392. More training helps, but this setup is unlikely to clear both walls.

## Parallel training

`make study` trains several runs at the same time, 4 by default. Set `JOBS` to choose how many:

```bash
make study JOBS=2
```

Without Make, the default is one at a time, so pass `--jobs`:

```bash
uv run python -m scripts.study --jobs 4
```

**How it works:** `scripts/study.py` starts each run as its own `scripts.train` process, up to `JOBS`
at once, and starts the next as soon as one finishes. Each run uses one CPU core.

**Results don't change:** a run's results are identical whether it trains alone or beside others.
Training the same four runs both ways gave metrics files that were identical byte for byte. Only
the time taken changes.

**Choosing `JOBS`:** the M4 has 4 performance cores and 6 efficiency cores, so the default of 4 is a
good starting point. More jobs use the slower efficiency cores. A MacBook Air has no fan and slows down
under long, heavy load, so the speedup is less than the number of jobs. Each run also needs a few
hundred MB of memory.

**Output:** the terminal shows one line when each run starts and one when it finishes or fails, with
how long it took. Each run's full output goes to `logs/steps_<STEPS>/run_logs/{variant}_seed{seed}.log`.

**Timing:** at the end, `make study` prints a table of training time per variant and in total:

```text
Training time for the runs trained now:
  variant    runs      total    per run
  dqn_n1        5     18m 02s     3m 36s
  ...
  all          20     1h 14m      3m 42s
Wall-clock time: 22m 10s, for 1h 14m of training
```

The numbers above are an illustration, not a measurement. The first figure is how long the command
took; the second is the runs' times added up, which is larger when runs overlap. Runs trained in
parallel each take a little longer than they would alone, because they share the machine. Runs
skipped as already finished aren't included. Every run's time is also appended to
`logs/steps_<STEPS>/run_logs/timings.csv` as soon as it ends, so the record survives an interruption and builds up
across invocations.

**Stopping:** `Ctrl+C` stops every run that's training. Those runs then show as incomplete in
`make status`, and each needs `make clean-run RUN=<name>` before `make study` can retrain it.

**Options:** `scripts.study` accepts `--variants`, `--seeds` and `--jobs`, plus every training option,
such as `--total-timesteps`, which it applies to every run. Run
`uv run python -m scripts.study --help` for its own options.

## Output files

| Output | Location |
|---|---|
| Raw metrics | `csv/{variant}_seed{seed}_metrics.csv` |
| TensorBoard | `tensorboard/{variant}_seed{seed}/` |
| W&B | Project `breakout-nstep-ddqn`, with local files in `wandb/` |
| Final model | `checkpoints/{variant}_seed{seed}.pt` |
| Model snapshots (seed 1) | `checkpoints/{variant}_seed1/step_{step}.pt` |
| Videos (seed 1) | `videos/{variant}_seed1/` |
| Training output, one file per run | `run_logs/{variant}_seed{seed}.log` |
| Training times | `run_logs/timings.csv` |
| Figures | `figures/` |

All of these are inside the run length's folder, `logs/steps_<STEPS>/`, such as
`logs/steps_12000000/csv/`. Pilot runs are in `logs/pilot/`, and the trial run in `logs/smoke/`.

Git ignores `logs/`, `*.csv`, `*.pt` and `*.png`, so keep copies of any results you need.

Each CSV row is an episode, training or evaluation row, and columns that don't apply are empty:

| Row | Filled columns, besides `step` |
|---|---|
| Episode | `episode`, `episodic_return`, `episode_length`, `epsilon` |
| Train (one per 100 updates) | `mean_q_value`, `td_loss` |
| Evaluation (one per 25,000 steps) | `eval_predicted_q`, `eval_actual_return` |

## Make commands

Run `make help` to see this list in your terminal.

| Command | What it does |
|---|---|
| `make help`, `make usage` | List every command, with the variables and examples |
| `make setup` | Install Python and all dependencies |
| `make wandb-login` | Log in to W&B, needed only for `WANDB=online` |
| `make test` | Run the tests |
| `make lint` | Check the code with Ruff, without changing it |
| `make format` | Fix lint problems and format the code with Ruff |
| `make typecheck` | Check types with mypy |
| `make check` | Run lint, type checks and tests |
| `make trial` | Try the whole workflow on a short run in `logs/smoke` |
| `make trial-clean` | Delete the trial run's files |
| `make baseline` | Measure the random-policy return |
| `make pilot` | Run the baseline and the two seed-0 pilots in `logs/pilot` |
| `make pilot-clean` | Delete the pilot's results so it can be repeated |
| `make train` | Train one run, chosen with `VARIANT` and `SEED` |
| `make study` | Train all 20 runs, 4 at a time, skipping finished ones |
| `make status` | Show which study runs are finished, incomplete or not started |
| `make tensorboard` | Open training charts at <http://localhost:6006> |
| `make sync` | Upload offline runs to W&B |
| `make figures` | Make the figures from `LOG_DIR`'s runs, saved in `LOG_DIR/figures` |
| `make videos` | Record videos of each variant's saved models |
| `make clean-run RUN=name` | Delete one run's results so it can be redone |
| `make clean-results` | Delete every result in `LOG_DIR` so training can start over, after asking |

### Variables

Set these after the command, as in `make train VARIANT=ddqn_n5 SEED=2`.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `VARIANT` | `ddqn_n3` | `train` | `dqn_n1`, `ddqn_n1`, `ddqn_n3` or `ddqn_n5` |
| `SEED` | `1` | `train`, `videos` | Random seed, or which seed's models to record |
| `VARIANTS` | All four | `study`, `status`, `videos` | Variants to include, in quotes |
| `SEEDS` | `1 2 3 4 5` | `study`, `status` | Seeds to include, in quotes |
| `DEVICE` | `cpu` | Training commands | `cpu` or `mps`. Use `cpu` for the study |
| `WANDB` | `offline` | `train`, `study` | `offline` or `online`. Trials and pilots are always offline |
| `STEPS` | `12000000` | All run commands | Training steps per run, which also picks the results folder. See [Training for longer](#training-for-longer) |
| `SNAPSHOT_INTERVAL` | `2000000` | `train`, `study` | Steps between model snapshots |
| `JOBS` | `4` | `study` | Runs to train at the same time. See [Parallel training](#parallel-training) |
| `LOG_DIR` | `logs/steps_<STEPS>` | `train`, `study`, `status`, `tensorboard`, `sync`, `figures`, `videos`, `clean-run` | Root folder for run outputs |
| `RUN` | None | `clean-run` | The run to delete, such as `ddqn_n3_seed1` |
| `CONFIRM` | None | `clean-results` | Set to `yes` to delete without being asked |
| `ARGS` | None | `test`, `baseline`, `train`, `study`, `figures`, `videos` | Extra options for the underlying command |

### Options you can pass with `ARGS`

For the full list with descriptions, run:

```bash
make train ARGS=--help
make videos ARGS=--help
make figures ARGS=--help
make baseline ARGS=--help
```

**`make train` and `make study`:**

| Option | Default | Meaning |
|---|---|---|
| `--snapshot-seeds` | `1` | Seeds whose runs save snapshots |

`STEPS` and `SNAPSHOT_INTERVAL` set `--total-timesteps` and `--snapshot-interval`. A value in `ARGS`
takes priority over them.

These are the frozen study settings. Change them only for trial runs or a new pilot round:

| Option | Default | Option | Default |
|---|---|---|---|
| `--total-timesteps` (`STEPS`) | `500000` | `--epsilon-start` | `1.0` |
| `--learning-rate` | `0.0001` | `--epsilon-end` | `0.05` |
| `--batch-size` | `64` | `--epsilon-decay-fraction` | `0.1` |
| `--buffer-capacity` | `100000` | `--learning-starts` | `10000` |
| `--gamma` | `0.99` | `--train-frequency` | `4` |
| `--tau` | `0.005` | `--huber-delta` | `1.0` |
| `--log-interval` | `100` | `--eval-interval` | `25000` |
| `--eval-games` | `5` | `--eval-epsilon` | `0.01` |
| `--clip-rewards` | on | `--terminal-on-life-loss` | on |
| `--auto-fire` | on | `--eval-seed-offset` | `10000` |

**`make videos`:**

| Option | Default | Meaning |
|---|---|---|
| `--steps` | `500000` | Run length, which sets the default folder `logs/steps_<steps>` |
| `--max-seconds` | `180` | Longest video |
| `--fps` | `30` | Frames per second. `15` is real game speed |
| `--epsilon` | `0.01` | Random move rate |
| `--upscale` | `3` | Enlargement of the game picture |
| `--game-seed-offset` | `20000` | Added to the seed for the recorded games |

**`make figures`:**

| Option | Default | Meaning |
|---|---|---|
| `--steps` | `500000` | Run length, which sets the default folders below |
| `--csv-dir` | `logs/steps_<steps>/csv` | Folder holding the run CSVs |
| `--output` | `logs/steps_<steps>/figures/ablation_comparison.png` | Comparison figure path |
| `--overestimation-output` | `logs/steps_<steps>/figures/overestimation_evaluation.png` | Overestimation figure path |
| `--window` | `100` | Rolling window, in episodes |
| `--min-periods` | `10` | Fewest episodes needed to draw a point |
| `--grid-points` | `500` | Points per curve |
| `--dpi` | `300` | Image resolution |

**`make baseline`:**

| Option | Default | Meaning |
|---|---|---|
| `--games` | `100` | Games to play |
| `--seed` | `0` | Random seed |

### Running without Make

Each command wraps a script you can also run directly, always from the project root and in the form
`uv run python -m scripts.<name>`. Running `python scripts/<name>.py` causes import errors. For
example:

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu --wandb-mode offline
uv run python -m scripts.study --jobs 4
```

## Troubleshooting

### `FileExistsError` when starting a run

Runs won't overwrite existing results. If you do want to redo a run, delete its outputs first:

```bash
make clean-run RUN=ddqn_n3_seed1
```

### Starting the study over

To delete every study result and train again from the beginning:

```bash
make clean-results
make study
```

`make clean-results` lists what it will delete, with sizes, and deletes nothing unless you type
`yes`. It removes the metrics, TensorBoard logs, models, snapshots, W&B files, videos, run logs and
figures in the run length's folder, `logs/steps_12000000` by default. Everything outside that folder
is kept, including `logs/pilot` and other run lengths. Clean another length with
`make clean-results STEPS=<STEPS>`.

It refuses to run while training is still writing to that folder. In a script, where it can't ask,
use `make clean-results CONFIRM=yes`. Online runs already uploaded to W&B stay there; delete those on
the W&B website.

### A run crashes or is interrupted

Runs can't resume. Delete the run's outputs with `make clean-run`, then run `make study` again. It
restarts that run from step 0 and skips the ones that already finished.

### A run stops with `TrainingError`

This means the loss or a gradient became NaN or infinite. **Don't simply rerun it**, because
divergence may be a real result for that variant:

- **If it looks like real divergence:** keep its logs and report it as a diverged run. `make status`
  keeps showing it as incomplete, and `make study` keeps listing it as not finished, which is
  expected.
- **If it looks like a bug:** fix the bug, then rerun every run it affected.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 6. A diverged run also limits that
variant's curves to the step where it failed, which you can see in the `last step` column.

### Other problems

| Problem | Cause and fix |
|---|---|
| `No rule to make target` or `No targets specified` | Run `make` from the project root, where the Makefile is |
| `ModuleNotFoundError: No module named 'src'` | Run from the project root, using `make` or `python -m scripts.train` |
| Training is slow | Keep `DEVICE=cpu`, the default |
| `no_models_found` or `no_videos_recorded` from `make videos` | Train first, or check `SEED` and `LOG_DIR`. Models must be in `{LOG_DIR}/checkpoints/` |
| `make videos` records only `final.mp4` | That run had no snapshots. Only seeds in `--snapshot-seeds` save them, and only during training |
| `no_recognized_runs` from `make figures` | No CSVs for that run length. Train first, or check `STEPS`. Pilot results are in `logs/pilot/csv` and aren't used |
| `skipping_unrecognized_run` from `make figures` | CSV names must look like `{variant}_seed{N}_metrics.csv` |
| `ConfigurationError` | An option has an invalid value. The message says which one |
| `wandb` asks you to log in | Use the default `WANDB=offline`, or run `make wandb-login` |
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

Run this before committing a change:

```bash
make check
```

It runs `make lint`, `make typecheck` and `make test`. To fix lint and formatting problems
automatically, run `make format`.

[`AGENTS.md`](AGENTS.md) has the full coding standards and commit rules.

## Project layout

```text
├── AGENTS.md              Instructions and conventions for AI coding agents
├── README.md
├── Makefile               Commands for every task; run `make help`
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
│   ├── study.py           Trains many runs, several at a time
│   ├── measure_random.py  Random-policy baseline
│   ├── plot_results.py    Comparison and overestimation figures
│   └── record_videos.py   Videos of saved models playing
├── tests/                 Unit and integration tests
└── logs/                  Run outputs, ignored by git
```

## Further reading

- [`docs/results/steps_15000000.md`](docs/results/steps_15000000.md): results of the 15-million-step study
- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md): how the code is organized and how a run works
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the math and implementation rules
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): the full protocol and how to interpret results
- [`AGENTS.md`](AGENTS.md): coding and engineering standards, and the checks to run before finishing
  a change
