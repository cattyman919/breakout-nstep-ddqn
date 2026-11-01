# breakout-nstep-ddqn

A reinforcement learning study of whether **n-step returns improve Double DQN** on Atari
Breakout, using the game's 128-byte RAM as the observation instead of pixels.

> **Status: training can't run yet.** The docs, project setup, config and CLI, logger and
> plotting script are done. The networks, buffers, agents and training loop are not
> implemented, so `scripts.train` currently stops with `NotImplementedError`.

## What it compares

| Config | Algorithm | n | What it tests |
|---|---|---|---|
| `dqn_n1` | DQN | 1 | Baseline |
| `ddqn_n1` | Double DQN | 1 | Effect of Double Q-learning |
| `ddqn_n3` | Double DQN | 3 | Effect of n-step returns |
| `ddqn_n5` | Double DQN | 5 | Effect of longer n-step returns |

Each config trains for 500,000 steps with 5 seeds, for 20 runs in total. All agents share
the same small MLP: 128 inputs, two hidden layers of 256 units, and 4 outputs.

The results are compared on two measures:

- **Episodic return:** how well each agent plays, in raw game points.
- **Mean predicted Q-value:** whether Double DQN predicts lower values than DQN, which is
  consistent with reduced overestimation.

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

Run all commands from the project root. The training commands work only after the training
loop is implemented.

**Train one run:**

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu --wandb-mode offline
```

No full run has been timed yet, so use this one to estimate how long the study will take.

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

**Make the comparison figure:**

```bash
uv run python scripts/plot_results.py
```

**See all training options:**

```bash
uv run python -m scripts.train --help
```

### Common problems

- **`FileExistsError` when rerunning:** the logger won't overwrite existing results. To redo
  a run, delete its CSV first, for example `logs/csv/ddqn_n3_seed1_metrics.csv`.
- **Slow training:** `--device` defaults to `mps`, but this small network ran faster on CPU
  in benchmarks. Pass `--device cpu`, and use the same device for every run in the study.
- **Import errors from `src`:** use `python -m scripts.train`, not `python scripts/train.py`.
- **`NameNotFound` for `ALE/Breakout-ram-v5`:** current `ale-py` no longer has that ID. Use
  `gym.make("ALE/Breakout-v5", obs_type="ram")`.

## Output

| Output | Location |
|---|---|
| Raw metrics | `logs/csv/{variant}_seed{seed}_metrics.csv` |
| TensorBoard | `logs/tensorboard/{variant}_seed{seed}/` |
| W&B | Project `breakout-nstep-ddqn`, with local files in `logs/wandb/` |
| Figure | `docs/figures/ablation_comparison.png` |

Git ignores `logs/`, `*.csv` and `*.png`, so keep copies of any results you need. To commit
the report figure, use `git add -f docs/figures/ablation_comparison.png`.

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
│   ├── networks.py        Not implemented
│   ├── buffers.py         Not implemented
│   ├── agents.py          Not implemented
│   └── utils/logger.py    TensorBoard, W&B and CSV logging
├── scripts/
│   ├── train.py           Config and CLI; training loop not implemented
│   └── plot_results.py    Two-panel comparison figure
└── logs/                  Run outputs, ignored by git
```

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the math and implementation rules
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): the full protocol and how to interpret results
- [`AGENTS.md`](AGENTS.md): coding conventions and the checks to run before finishing a change
