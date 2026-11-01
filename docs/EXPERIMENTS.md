# Experiments

Protocol for the ablation study. Only the factors in section 1 vary between runs.

## 1. Ablation matrix

| Config | Algorithm | Target | $n$ | Isolates |
|---|---|---|---|---|
| `dqn_n1` | Standard DQN | $y_t^{\text{DQN}}$ | 1 | Baseline |
| `ddqn_n1` | Double DQN | $y_t^{\text{DDQN}}$ | 1 | Effect of Double Q-learning (against `dqn_n1`) |
| `ddqn_n3` | $n$-step Double DQN | $y_t^{n\text{-DDQN}}$ | 3 | Effect of $n$ (against `ddqn_n1`) |
| `ddqn_n5` | $n$-step Double DQN | $y_t^{n\text{-DDQN}}$ | 5 | Effect of $n$ (against `ddqn_n1`, `ddqn_n3`) |

Each config runs with seeds **1, 2, 3, 4 and 5**, for 20 runs in total. Run names are
`{config}_seed{seed}`, for example `ddqn_n3_seed2`.

## 2. Fixed hyperparameters

### Specified for this study

| Parameter | Value |
|---|---|
| Total timesteps | $500{,}000$ |
| Learning rate | $\alpha = 1 \times 10^{-4}$ (Adam) |
| Batch size | $64$ |
| Replay buffer capacity | $100{,}000$ |
| Discount factor | $\gamma = 0.99$ |
| Polyak coefficient | $\tau = 0.005$ |
| $\epsilon$-greedy schedule | Linear from $\epsilon_{\text{start}} = 1.0$ to $\epsilon_{\text{end}} = 0.05$ over the first $10\%$ of total timesteps ($50{,}000$ steps), then constant |

A timestep is one agent decision, which is 4 emulator frames, so a run spans 2,000,000
frames.

### Implementation defaults

These values are needed to make runs reproducible, and must stay fixed across all runs
once training starts.

| Parameter | Value | Reason |
|---|---|---|
| Seeds per config | 5 (1 to 5) | Shows seed-to-seed spread at an affordable compute cost |
| Learning starts | $10{,}000$ steps | Fills the buffer with varied transitions before the first update |
| Train frequency | 1 update every 4 steps | Standard DQN ratio; gives $122{,}500$ updates per run |
| Polyak cadence | After each gradient update | |
| Huber $\delta$ | $1.0$ | PyTorch `SmoothL1Loss` default |
| Gradient clipping | None | The Huber loss already bounds per-sample gradients |
| Reward clipping | Sign of the reward, per step, before $n$-step summation | Keeps target scale comparable across $n$; standard for Atari DQN |
| Life-loss handling | Terminal for training targets; no environment reset | Gives a direct signal for dropping the ball; standard Atari DQN heuristic |
| Frame stacking | None | Single RAM snapshot; matches the 128-dimension network input |
| Ball launch | One automatic FIRE after each reset and lost life | Avoids spending steps waiting for FIRE; each press counts as one environment step and is not stored in replay |

The sticky-action probability ($0.25$) and frame skip ($4$) are environment defaults.

## 3. Evaluation metrics

| Metric | Definition | Logged by | CSV column |
|---|---|---|---|
| Episodic return | Undiscounted sum of raw (unclipped) rewards per full game of 5 lives, plotted as a 100-episode rolling average | `log_episode` | `episodic_return` |
| Episode length | Total steps survived per full game of 5 lives | `log_episode` | `episode_length` |
| Mean predicted $Q$-value | Average of $\max_a Q_{\phi}(s, a)$ across each training mini-batch | `log_train_step` | `mean_q_value` |
| TD loss | Huber (Smooth L1) loss per gradient update | `log_train_step` | `td_loss` |
| $\epsilon$ | Exploration rate at episode end | `log_episode` | `epsilon` |

Compute the mean predicted $Q$ under `torch.no_grad()` on the states $s_t$ of the same
mini-batch used for the update, with the online network $\phi$.

## 4. Telemetry and output

| Output | Location |
|---|---|
| TensorBoard | `logs/tensorboard/{run_name}` |
| Weights & Biases | Project `breakout-nstep-ddqn`, local files in `logs/wandb/` |
| Raw CSV | `logs/csv/{run_name}_metrics.csv` |
| Figure | `docs/figures/ablation_comparison.png` (300 DPI) |

Each CSV row is either an episode row or a train row; columns that do not apply are
empty. See the module docstring of `src/utils/logger.py`.

## 5. Running the study

`scripts/train.py` currently provides the CLI only. These commands work once its
training loop is implemented:

```bash
for config in dqn_n1 ddqn_n1 ddqn_n3 ddqn_n5; do
  for seed in 1 2 3 4 5; do
    uv run python -m scripts.train --variant "$config" --seed "$seed" --device cpu
  done
done

uv run python scripts/plot_results.py
```

Add `--wandb-mode offline` to run without network access, then upload the results
with `wandb sync logs/wandb/offline-run-*`.

### Choosing a device

Both devices are supported through `--device`. Measured on the development MacBook Air
M4 with this project's MLP (torch 2.15.0):

| Operation | `cpu` | `mps` |
|---|---|---|
| One gradient update (batch 64) | 0.34 ms | 1.26 ms |
| One single-state action selection | 0.01 ms | 0.23 ms |

For this small network the CPU is about 4 times faster per update and roughly 20 times
faster per action. The loop above therefore uses `--device cpu`. These are
micro-benchmarks of network operations only, not of whole training runs. Use the same
device for every run in the study.

## 6. Figure

`scripts/plot_results.py` reads every CSV in `logs/csv/`.

- **Panel A:** episodic return against environment steps for all four variants, as a
  100-episode rolling mean with a shaded band of one standard deviation. Windows are
  partial until 100 episodes have finished, with a minimum of 10. Seeds are averaged,
  and the band pools variation within and across seeds.
- **Panel B:** mean predicted $Q$-value against environment steps, averaged into 500
  step bins and across seeds. An arrow marks the signed gap between `dqn_n1` and
  `ddqn_n1` over the final 10% of steps.

## 7. Interpreting the results

- **What Panel B can show.** A lower mean $Q$ for Double DQN is consistent with reduced
  overestimation, but it is not proof on its own. The variants visit different states
  and earn different returns, and a better policy should legitimately have higher
  values. Stronger evidence is to compare each curve with the discounted return the
  policy actually earns, as van Hasselt et al. (2016) did. That is outside the metrics
  specified here.
- **Comparing $n$.** Differences among `ddqn_n1`, `ddqn_n3` and `ddqn_n5` in Panel B
  also reflect the uncorrected off-policy bias of $n$-step returns, not only
  overestimation.
- **Training returns.** Reported returns come from the $\epsilon$-greedy behavior policy
  at $\epsilon = 0.05$ after decay, with no separate greedy evaluation.
- **Q-value scale.** With clipped rewards and terminal life loss, Panel B is in units of
  discounted clipped return until the next lost life, not game points. It cannot be read
  against Panel A directly. Any comparison with an empirical return must use that same
  definition.
- **Training heuristics.** Reward clipping and terminal life loss change the objective:
  a 7-point brick is worth the same as a 1-point brick, and the agent is not optimizing
  the true game return. Report both as part of the method. Machado et al. (2018)
  recommend against using life-loss information when evaluating agents, so results are
  not directly comparable to studies that omit it.
- **Statistical power.** Five seeds show spread but cannot support strong significance
  claims. Report results as observed differences.
- **Automatic FIRE.** The agent never has to learn to launch the ball, so report this as
  part of the method. One press failed to launch in about $0.5\%$ of test trials, and a
  policy that then avoids FIRE launches only through exploration, at about 1 step in 80
  when $\epsilon = 0.05$. Long episodes with low return in `episode_length` indicate this.
