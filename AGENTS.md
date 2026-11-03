# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

`breakout-nstep-ddqn` implements and evaluates an **n-step Double DQN** against
1-step DQN and 1-step Double DQN baselines on Atari Breakout with 128-byte RAM
observations.

| Topic | Source |
|---|---|
| Overview and how to run | [`README.md`](README.md) |
| MDP, network, target equations, n-step rules | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Ablation matrix, hyperparameters, metrics | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| How the code is organized | [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) |

### Implementation status

| Component | Status |
|---|---|
| `src/` and `scripts/` | Implemented, to the engineering standards below |
| `tests/` | Covers every area listed under Tests |
| Pilot | See the table in `docs/EXPERIMENTS.md` section 5 |
| 20-run study | Not started |

## Project tree

```text
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CODE_GUIDE.md
│   └── EXPERIMENTS.md
├── src/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── networks.py
│   ├── buffers.py
│   ├── agents.py
│   ├── environment.py
│   ├── evaluation.py
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── train.py
│   ├── study.py
│   ├── measure_random.py
│   ├── plot_results.py
│   └── record_videos.py
├── tests/
└── logs/
```

Generated, not part of the hand-written tree: `uv.lock` and `.python-version`
(commit both), `.venv/`, the contents of `logs/`, and `docs/figures/`.

## Initialization

```bash
git init
echo "logs/\n.venv/\n__pycache__/\n*.csv\n*.pt\n*.png" > .gitignore
uv init
uv add torch numpy "gymnasium[atari]" ale-py wandb tensorboard matplotlib pandas tyro
```

Things to know about these commands:

- **`echo`:** the `\n` escapes expand in zsh, the macOS default shell. In bash they
  are written literally, so use `printf` there.
- **`uv init` Python version:** plain `uv init` pins the newest interpreter it finds
  (3.14 on the development machine). This repo was created with
  `uv init --python 3.12`, and `uv.lock` was resolved on that basis.
- **`uv init` template files:** it also creates `main.py`, which isn't in the tree, so
  delete it. It creates an empty `README.md` too; this repo's is written by hand.

For an existing checkout, `uv sync` is all that is needed.

## Package manager

Use `uv` only: `uv add`, `uv sync`, `uv run`. Do not use any other Python package
manager or edit dependency versions by hand.

## Environment

The environment ID `ALE/Breakout-ram-v5` was removed in `ale-py` 0.11. This project
locks `ale-py` 0.12.1, where that ID raises `NameNotFound`. Create the same
environment with:

```python
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", obs_type="ram")
```

- **Observations:** `Box(0, 255, (128,), uint8)`. Normalize to `float32` in
  $[0.0, 1.0]$ by dividing by $255.0$ before the network. Store `uint8` in the replay
  buffer and normalize when sampling, which takes a quarter of the memory of `float32`.
- **Actions:** `Discrete(4)`: 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT.
- **Training signal:** targets use sign-clipped rewards and treat a lost life
  (`info["lives"]` dropping) as terminal, without resetting the environment. Logged
  episodic return and length stay raw and per full game. See
  `docs/ARCHITECTURE.md` sections 1 and 4.
- **Automatic FIRE:** the training loop presses FIRE once after each reset and lost life.
  The press counts as an environment step and is not stored in replay.

## Device selection

- Use `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` semantics.
- `--device {mps,cpu}` selects the device. `mps` is the default and falls back to
  `cpu` with a warning when unavailable. See `resolve_device` in `scripts/train.py`.
- Do not assume MPS is faster. See the measurements in `docs/EXPERIMENTS.md`.

## Coding conventions

- **Typing:** Python 3.12+ with explicit type hints on every function, method and class
  attribute, including return types. `mypy --strict` must pass.
- **No global variables:** no mutable or constant module-level state. Put fixed values
  in a dataclass field, a `ClassVar`, or a function's local scope. Imports, function
  and class definitions, and the `__main__` guard are the only module-level statements.
- **Configuration:** hyperparameters live in frozen dataclasses parsed with `tyro`.
  Pass config objects explicitly; never hard-code a hyperparameter at its point of use.
  Validate values in `__post_init__` and raise `ConfigurationError` on invalid input. Do not
  add `pydantic`; strict dataclasses cover this and already work with `tyro`.
- **Reproducible seeding:** every run seeds all of the following from a single `seed`:

  ```python
  torch.manual_seed(seed)
  torch.set_num_threads(1)
  np.random.default_rng(seed)  # pass this generator explicitly; do not use np.random.seed
  env.reset(seed=seed)  # first reset only
  env.action_space.seed(seed)
  ```

  One generator serves $\epsilon$-greedy and replay sampling. Every variant uses the same
  seed list, with evaluation seeded separately from `seed + 10,000`. Seed 0 is reserved for
  pilots. MPS results may still differ between runs and from CPU results, so report the
  device used with each result.
- **Linting and formatting:** use Ruff for both, and do not add other linters or formatters
  such as `black`, `isort` or `flake8`. Ruff is a dev dependency, configured in
  `pyproject.toml` with line length 100. Run `uv run ruff format .` to format, and
  `uv run ruff check --fix .` to apply safe lint fixes.

### Error handling

- **Specific exceptions only.** Never use a bare `except:`. Catch named exceptions such as
  `ValueError`, `RuntimeError` or `KeyError`. A broad `except BaseException:` is allowed only
  to release resources, and it must re-raise.
- **Custom hierarchy.** `src/exceptions.py` defines `BaseProjectError` and its subclasses
  `ConfigurationError`, `GameEnvironmentError` and `TrainingError`. Do not name a class
  `EnvironmentError`, because that shadows Python's built-in alias for `OSError`.
- **Context in every failure.** An exception either propagates with a descriptive message,
  using `raise ... from err` when translating, or is caught, logged with its full stack trace
  and handled. Never swallow one silently.
- **Non-finite values stop the run.** Check the loss and gradients after each backward pass,
  and raise `TrainingError` if any value is NaN or infinite, naming the step. See
  `docs/EXPERIMENTS.md` section 6 for how to treat such a run.

### Structured logging

- **Diagnostics are JSON.** Use `structlog` to emit one JSON object per entry, including the
  timestamp, run name, seed and step count. Before training starts, `step` is `null`.
- **No module-level loggers.** The no-globals rule applies here too. Create a bound logger at
  the entry point and pass it explicitly.
- **Stderr is not purely JSON.** Diagnostics go to stderr, where W&B and ALE also print their
  own plain text. Filter for lines starting with `{` when parsing.
- **Two separate systems.** `structlog` records what the program is doing, such as startup,
  warnings, evaluation points and failures. `DualLogger` records experiment metrics. Do not
  send metrics through `structlog`, and do not write a diagnostic entry on every step.
- **`print` is for intended command output only**, such as the summary table from
  `scripts/plot_results.py`. Warnings and errors go through the structured logger.

### Resource management

Use `with` statements for anything that must be released: file I/O, Gymnasium environments
and `DualLogger`. Flushing and closing must also happen when a run fails.

### Checks

Check before finishing a change, with `make check` or these commands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --with mypy mypy --strict --ignore-missing-imports src scripts
uv run python -m pytest   # once tests/ exists
```

## Tests

Silent bugs in target calculation invalidate results without crashing, so `tests/`, written
with `pytest`, must have unit and integration coverage of:

- **Networks:** output dimensions for single states and batches, and rejection of invalid
  input shapes.
- **Replay buffer overflow:** at capacity, the oldest transitions are overwritten and the size
  stays at capacity. Filling the buffer is normal behavior, not an error.
- **Invalid input:** wrong observation shapes and invalid configuration raise the documented
  exceptions.
- **Gradient stability:** an agent update changes the online network, leaves all values finite,
  and raises `TrainingError` when given a NaN loss or gradient.
- **Integration:** a very short run, on a reduced step budget in W&B offline mode, writes the
  expected CSV and closes its environment and logger, including when it fails.
- **N-step windows:** full windows, flushes at termination, truncation and lost lives, and no
  duplicate or dropped transitions, for $n = 1, 3, 5$.
- **Terminal mask:** $d_t$ is set by termination or a lost life, never by truncation, and spans
  shorter than $n$ use $\gamma^m$.
- **Targets:** DQN and Double DQN against hand-computed values.
- **Rewards:** targets use clipped rewards, and logged returns use raw rewards.
- **Polyak updates:** $\phi' \leftarrow \tau \phi + (1 - \tau)\phi'$ with the correct weighting.
- **Evaluation returns:** $R_i$ stops at a lost life, and truncated final lives are excluded.

Tests must not depend on W&B network access or a full training run.

## Git commits

Commit every completed change to the project's code or docs, before moving on to the next
one:

1. Run the checks above and fix any failure. If a change can't pass them yet, say so in
   your report rather than committing it as finished.
2. Review what will be committed with `git status` and `git diff`.
3. Stage files by name, not with `git add .` or `git add -A`.
4. Commit with a message in the format below.

```bash
git add src/buffers.py docs/ARCHITECTURE.md
git commit -m "feat(buffers): add n-step transition assembly"
```

**Message format:** `<type>(<optional scope>): <summary>`, with the summary in the imperative
mood and under 72 characters. Add a body when the reason for the change isn't obvious.

| Type | Use for |
|---|---|
| `feat` | New behavior |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `refactor` | Code changes that keep behavior the same |
| `test` | Tests only |
| `chore` | Dependencies, tooling, project setup |

**Rules:**

- **One purpose per commit.** Keep unrelated changes in separate commits.
- **Keep docs in step.** If a change affects the math, protocol or status table, update
  `docs/` or this file in the same commit.
- **Commit dependency changes together.** Include `pyproject.toml` and `uv.lock` in the same
  commit.
- **Don't commit run outputs or secrets.** That covers `logs/`, checkpoints, CSVs and W&B API
  keys. The one exception to the ignore rules is the report figures, using the `git add -f`
  command under Logging contract.
- **Don't rewrite shared history.** No `git push --force`, and no amending or rebasing commits
  that have been pushed.
- **Don't push unless asked.** Committing is expected; pushing is not.

## Running

Run everything from the project root. Paths such as `logs/` are relative to it. The `Makefile`
wraps every command below; run `make help` to list them. Add new commands to it with a `## `
description, which is what `make help` displays.

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu
uv run python -m scripts.plot_results
```

Use `python -m scripts.<name>` for every script, not `python scripts/<name>.py`. The project
is not installed as a package, so `src` is importable only with the root on `sys.path`.

`src/utils/` intentionally has no `__init__.py`, to match the tree. It imports as a
namespace package: `from src.utils.logger import DualLogger`.

## Logging contract

- All training metrics go through `DualLogger`, never directly to a single sink.
- The training loop averages TD loss and mean $Q$ over 100 updates before calling
  `log_train_step`. Evaluation results go through `log_eval`.
- Each run length has its own results folder, `logs/steps_<total_timesteps>`, which is
  `TrainConfig`'s default `log_dir` and the Makefile's default `LOG_DIR`. Never mix run lengths in
  one folder.
- Pilot runs use `--log-dir logs/pilot`, so `scripts/plot_results.py` never reads them. Follow
  the pilot procedure in `docs/EXPERIMENTS.md` section 5 before starting study runs.
- Run names are `{variant}_seed{seed}`. `scripts/plot_results.py` parses this format
  from `logs/steps_<N>/csv/{run_name}_metrics.csv` and skips files that do not match.
- `DualLogger` refuses to overwrite an existing CSV unless `overwrite=True`. Do not
  pass that flag for ablation runs.
- `.gitignore` excludes `*.png`, `*.csv` and `*.pt`. Figures are written to
  `logs/steps_<N>/figures/`. To commit a report figure, copy it into `docs/figures/` and use
  `git add -f`, for example `git add -f docs/figures/ablation_comparison.png`.
