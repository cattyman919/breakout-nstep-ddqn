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

### Implementation status

| Component | Status |
|---|---|
| `src/utils/logger.py`, `scripts/plot_results.py` | Implemented |
| `scripts/train.py` | Config and CLI only; the training loop raises `NotImplementedError` |
| `src/networks.py`, `src/buffers.py`, `src/agents.py` | Docstring-only stubs |

## Project tree

```text
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   └── EXPERIMENTS.md
├── src/
│   ├── __init__.py
│   ├── networks.py
│   ├── buffers.py
│   ├── agents.py
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── train.py
│   └── plot_results.py
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

- **Typing:** Python 3.12+ with complete type hints on every function, including
  return types. `mypy --strict` must pass.
- **No global variables:** no mutable or constant module-level state. Put fixed values
  in a dataclass field, a `ClassVar`, or a function's local scope. Imports, function
  and class definitions, and the `__main__` guard are the only module-level statements.
- **Configuration:** hyperparameters live in frozen dataclasses parsed with `tyro`.
  Pass config objects explicitly; never hard-code a hyperparameter at its point of use.
- **Reproducible seeding:** every run seeds all of the following from a single `seed`:

  ```python
  torch.manual_seed(seed)
  np.random.default_rng(seed)  # pass this generator explicitly; do not use np.random.seed
  env.reset(seed=seed)         # first reset only
  env.action_space.seed(seed)
  ```

  MPS results may still differ between runs and from CPU results, so report the device
  used with each result.
- **Formatting:** `ruff check` and `ruff format` with line length 100.

Check before finishing a change:

```bash
uvx ruff check --line-length 100 src scripts
uvx ruff format --check --line-length 100 src scripts
uv run --with mypy mypy --strict --ignore-missing-imports src scripts
```

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
  keys. The one exception to the ignore rules is the report figure, using the `git add -f`
  command under Logging contract.
- **Don't rewrite shared history.** No `git push --force`, and no amending or rebasing commits
  that have been pushed.
- **Don't push unless asked.** Committing is expected; pushing is not.

## Running

Run everything from the project root. Paths such as `logs/` are relative to it.

```bash
uv run python -m scripts.train --variant ddqn_n3 --seed 1 --device cpu
uv run python scripts/plot_results.py
```

Use `python -m scripts.train`, not `python scripts/train.py`. The project is not
installed as a package, so `src` is importable only with the root on `sys.path`.

`src/utils/` intentionally has no `__init__.py`, to match the tree. It imports as a
namespace package: `from src.utils.logger import DualLogger`.

## Logging contract

- All training metrics go through `DualLogger`, never directly to a single sink.
- Run names are `{variant}_seed{seed}`. `scripts/plot_results.py` parses this format
  from `logs/csv/{run_name}_metrics.csv` and skips files that do not match.
- `DualLogger` refuses to overwrite an existing CSV unless `overwrite=True`. Do not
  pass that flag for ablation runs.
- `.gitignore` excludes `*.png`, `*.csv` and `*.pt`, so
  `docs/figures/ablation_comparison.png` is untracked. To commit a report figure, use
  `git add -f docs/figures/ablation_comparison.png`.
