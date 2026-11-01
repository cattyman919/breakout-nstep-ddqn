# breakout-nstep-ddqn

A reinforcement learning study of whether **n-step returns improve Double DQN** on Atari
Breakout, using the game's 128-byte RAM as the observation instead of pixels.

> **Status: implemented; the 20-run study has not been run.** Pilot results are in
> [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 5.

## What it compares

| Config | Algorithm | n | What it tests |
|---|---|---|---|
| `dqn_n1` | DQN | 1 | Baseline |
| `ddqn_n1` | Double DQN | 1 | Effect of Double Q-learning |
| `ddqn_n3` | Double DQN | 3 | Effect of n-step returns |
| `ddqn_n5` | Double DQN | 5 | Effect of longer n-step returns |

Each config trains for 500,000 steps with 5 seeds, for 20 runs in total. All agents share
the same small MLP: 128 inputs, two hidden layers of 256 units, and 4 outputs.

The results are compared on three measures:

- **Episodic return:** how well each agent plays, in raw game points.
- **Mean predicted Q-value:** whether Double DQN predicts lower values than DQN.
- **Predicted vs. actual value:** every 25,000 steps, evaluation games compare what each agent
  predicts with the discounted return it actually earns. This is the direct test of
  overestimation.

## Requirements

- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and all dependencies
- A Mac with Apple Silicon is the target hardware; the code can also run on CPU
- A [Weights & Biases](https://wandb.ai) account, only if you want online sync

## Setup

```bash
uv sync
uv run wandb login   # skip this if you use offline mode
```

## How to run

Run all commands from the project root.

**Run the pilot first.** This checks that the agents learn before the settings freeze:

```bash
uv run python -m scripts.measure_random
uv run python -m scripts.train --variant ddqn_n3 --seed 0 --device cpu --wandb-mode offline --log-dir logs/pilot
uv run python -m scripts.train --variant dqn_n1 --seed 0 --device cpu --wandb-mode offline --log-dir logs/pilot
```

Both variants must reach at least double the random-policy return. No full run has been timed
yet, so use these to estimate how long the study will take. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 5 for what to do if they fall short.

**Train the full study:**

```bash
for config in dqn_n1 ddqn_n1 ddqn_n3 ddqn_n5; do
  for seed in 1 2 3 4 5; do
    uv run python -m scripts.train --variant "$config" --seed "$seed" --device cpu
  done
done
```

This syncs to W&B online by default. To run without a login, add `--wandb-mode offline`,
then upload afterward with `uv run wandb sync logs/wandb/offline-run-*`.

**Watch progress:**

```bash
uv run tensorboard --logdir logs/tensorboard
```

**Make the figures:**

```bash
uv run python -m scripts.plot_results
```

**Run the tests:**

```bash
uv run python -m pytest
```

**Lint and format with Ruff:**

```bash
uv run ruff check --fix .
uv run ruff format .
```

**See all training options:**

```bash
uv run python -m scripts.train --help
```

### Common problems

- **`FileExistsError` when rerunning:** the logger won't overwrite existing results. To redo
  a run, including after a crash, delete its CSV, TensorBoard folder and any checkpoint, then
  restart it from step 0. Runs can't resume.
- **Slow training:** `--device` defaults to `mps`, but this small network ran faster on CPU
  in benchmarks. Pass `--device cpu`, and use the same device for every run in the study.
- **Import errors from `src`:** use `python -m scripts.train`, not `python scripts/train.py`.
  The same applies to the other scripts.
- **`NameNotFound` for `ALE/Breakout-ram-v5`:** current `ale-py` no longer has that ID. Use
  `gym.make("ALE/Breakout-v5", obs_type="ram")`.

## Output

| Output | Location |
|---|---|
| Raw metrics | `logs/csv/{variant}_seed{seed}_metrics.csv` |
| TensorBoard | `logs/tensorboard/{variant}_seed{seed}/` |
| W&B | Project `breakout-nstep-ddqn`, with local files in `logs/wandb/` |
| Final model | `logs/checkpoints/{variant}_seed{seed}.pt` |
| Pilot runs | `logs/pilot/` |
| Figures | `docs/figures/ablation_comparison.png`, `docs/figures/overestimation_evaluation.png` |

Git ignores `logs/`, `*.csv`, `*.pt` and `*.png`, so keep copies of any results you need. To commit
a report figure, use `git add -f`, for example
`git add -f docs/figures/ablation_comparison.png`.

## Training setup to report

These settings change what the agents learn, so results aren't directly comparable with
studies that don't use them:

- Training targets use the sign of each reward, while logged returns stay in raw points.
- Losing a life ends the training target, but the game continues until game over.
- FIRE is pressed automatically after each reset and lost life.

Because of the first two, Q-values mean discounted clipped return until the next lost life,
so they can't be read directly against game scores.

## Project layout

```text
├── AGENTS.md              Instructions and conventions for AI coding agents
├── README.md
├── pyproject.toml
├── docs/
│   ├── ARCHITECTURE.md    MDP, network, target equations, n-step rules
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
│   └── plot_results.py    Comparison and overestimation figures
├── tests/                 Unit and integration tests
└── logs/                  Run outputs, ignored by git
```

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the math and implementation rules
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): the full protocol and how to interpret results
- [`AGENTS.md`](AGENTS.md): coding and engineering standards, and the checks to run before
  finishing a change
