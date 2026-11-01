# Architecture

Mathematical and algorithmic specification of the agents compared in this project.

## 1. MDP formulation

Breakout with RAM observations is modeled as the tuple
$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

The project refers to this environment as `ALE/Breakout-ram-v5`. Current `ale-py`
releases no longer register that ID, so it is created with
`gym.make("ALE/Breakout-v5", obs_type="ram")`, which has the same observation and
action spaces.

| Element | Definition |
|---|---|
| State space $\mathcal{S}$ | $\mathcal{S} \in [0, 1]^{128}$: the 128 bytes of Atari 2600 RAM, $s = o / 255.0$ with $o \in \{0, \dots, 255\}^{128}$, as `float32` |
| Action space $\mathcal{A}$ | $\lvert\mathcal{A}\rvert = 4$: 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT |
| Transition $\mathcal{P}(s' \mid s, a)$ | Unknown to the agent. Stochastic because of sticky actions: with probability $0.25$ the previous action repeats. One step spans 4 emulator frames |
| Reward $\mathcal{R}(s, a)$ | Points scored by bricks broken during the step. Training uses the clipped signal defined below |
| Discount $\gamma$ | $\gamma = 0.99$ |

The environment settings above were read from the registered spec of
`ALE/Breakout-v5` under `ale-py` 0.12.1.

**Episode end.** An episode *terminates* at game over, after all lives are lost. It is
*truncated* by the emulator limit of 108,000 frames, or 27,000 agent steps. The two
are handled differently; see section 4.

**Training signal.** Two standard Atari heuristics change what the agents optimize. They
apply to all variants and to training targets only:

- **Reward clipping.** Targets use $r_{t+1} = \operatorname{sign}(r^{\text{raw}}_{t+1})$,
  applied per step before any $n$-step sum. This keeps target scale comparable across
  $n$, since $G_t^{(n)} \le \sum_{k=0}^{n-1} \gamma^k$ (about $4.90$ for $n = 5$), where
  unclipped rewards could reach about $34$. All rewards in this document are clipped
  unless marked raw.
- **Terminal on life loss.** A lost life ends the *training* episode: the target does not
  bootstrap past it. It is detected as `info["lives"]` dropping below its previous value.
  The environment is **not** reset; play continues until game over.

$Q_\phi$ therefore estimates the discounted clipped return until the next lost life,
not the game score. Reported metrics are unaffected: episodic return and episode length
use raw rewards and span a full game of 5 lives, ending at game over or truncation.

**Automatic FIRE.** The ball does not move until FIRE is pressed, so the training loop
presses FIRE once after every environment reset and every lost life. This changes the
dynamics the agent experiences: it effectively starts each life with the ball in play.
The press works as follows:

- It counts as one environment step toward `total_timesteps`, the $\epsilon$ schedule,
  the train frequency and episode length, and any raw reward counts toward episodic
  return.
- It is not stored in the replay buffer. The next transition starts from the observation
  returned by the press.
- If it ends the game or the time limit, the normal end-of-episode handling applies.
- The action space stays $\lvert\mathcal{A}\rvert = 4$, so the agent can still select FIRE.

One press launched the ball in $99.5\%$ of 200 trials, both after a reset and after a
lost life, despite sticky actions. In the remaining cases the agent's own FIRE action is
still available, so the loop does not press again.

**Partial observability.** Each decision uses a single RAM snapshot with no frame
stacking. Any property of the game not encoded in RAM is hidden from the agent, so the
process is treated as an MDP by assumption.

## 2. Neural network architecture

The RAM state is a flat vector with no spatial structure, so the Q-network is a
lightweight multi-layer perceptron rather than a CNN.

| Layer | Definition | Output shape |
|---|---|---|
| Input | Normalized RAM state | $128$ |
| Hidden 1 | `Linear(128, 256)` + `ReLU` | $256$ |
| Hidden 2 | `Linear(256, 256)` + `ReLU` | $256$ |
| Output | `Linear(256, 4)` | $4$, the action-values $Q_{\phi}(s, a)$ |

The network has $33{,}024 + 65{,}792 + 1{,}028 = 99{,}844$ trainable parameters.

All variants use two networks of this architecture: an online network with parameters
$\phi$, trained by gradient descent, and a target network with parameters $\phi'$,
initialized as $\phi' \leftarrow \phi$.

## 3. The three agent variants

### Target network update

All variants use Polyak averaging after each gradient update:

$$\phi' \leftarrow \tau \phi + (1 - \tau)\phi', \qquad \tau = 0.005$$

$\tau$ is the fraction of the online network mixed in per update, so $\phi'$ changes
slowly. The opposite weighting, $\tau \phi' + (1 - \tau)\phi$, would need
$\tau = 0.995$ to behave the same way. With $\tau = 0.005$ it would copy $99.5\%$ of
$\phi$ at every update and remove the stabilizing effect of the target network.

### Variant 1: Standard 1-step DQN

$$y_t^{\text{DQN}} = r_{t+1} + \gamma \max_{a'} Q_{\phi'}(s_{t+1}, a')$$

The target network both selects and evaluates the next action. Because
$\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$, taking a maximum
over noisy estimates is biased upward, and bootstrapping propagates that bias.

### Variant 2: 1-step Double DQN (DDQN)

$$y_t^{\text{DDQN}} = r_{t+1} + \gamma Q_{\phi'}\left(s_{t+1}, \arg\max_{a'} Q_{\phi}(s_{t+1}, a')\right)$$

Decoupling action selection ($\phi$) from action evaluation ($\phi'$) reduces the
overestimation bias. It does not eliminate it: $\phi$ and $\phi'$ are correlated,
particularly because $\phi'$ is a moving average of $\phi$, so their errors are not
independent.

### Variant 3: $n$-step Double DQN (core innovation)

The agent uses the $n$-step truncated return:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1}$$

Target:

$$y_t^{n\text{-DDQN}} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n Q_{\phi'}\left(s_{t+n}, \arg\max_{a'} Q_{\phi}(s_{t+n}, a')\right)$$

With $n = 1$ this reduces exactly to Variant 2.

Larger $n$ propagates reward information faster and depends less on the bootstrapped
estimate. In exchange, the return has higher variance. It is also uncorrected
off-policy: the intermediate rewards come from an older $\epsilon$-greedy behavior
policy, not the current greedy policy, and no importance-sampling correction is
applied. This bias grows with $n$ and is part of what the ablation measures.

### Terminal handling (applies to all variants)

The equations above show the non-terminal case. The implemented target, for a
transition that spans $m \le n$ steps, is:

$$y_t = G_t^{(m)} + (1 - d_t)\, \gamma^{m}\, \hat{Q}(s_{t+m})$$

where $d_t = 1$ only if the window ended in a lost life or environment termination, and

$$\hat{Q}(s) = \begin{cases} \max_{a'} Q_{\phi'}(s, a') & \text{DQN} \\ Q_{\phi'}\left(s, \arg\max_{a'} Q_{\phi}(s, a')\right) & \text{Double DQN} \end{cases}$$

### Loss

Targets are constants for differentiation (no gradient flows through $y_t$). The loss
is the Huber loss (Smooth L1, $\delta = 1$) over a uniformly sampled mini-batch
$\mathcal{B}$:

$$\mathcal{L}(\phi) = \frac{1}{\lvert\mathcal{B}\rvert} \sum_{i \in \mathcal{B}} \text{Huber}\left(y_i - Q_{\phi}(s_i, a_i)\right)$$

## 4. $n$-step transition assembly

A sliding-window `collections.deque(maxlen=n)` sits in front of the replay buffer.

**Stored transition.** The tuple $(s_t, a_t, G_t^{(n)}, s_{t+n}, \text{done})$ is
extended with the span $m$, because windows flushed at a boundary are
shorter than $n$ and need $\gamma^m$ rather than $\gamma^n$:

$$(s_t,\ a_t,\ G_t^{(m)},\ s_{t+m},\ d_t,\ m)$$

**Algorithm.** After each environment step:

1. Append $(s_t, a_t, r_{t+1})$ to the deque, with $r_{t+1}$ clipped.
2. If the deque holds $n$ entries, commit one transition using the entire window as
   its return and the new observation as $s_{t+n}$, then `popleft()` the entry just
   committed. Without the `popleft()`, the flush below would commit it a second time.
3. If a boundary was reached (lost life, termination or truncation), flush: while the
   deque is non-empty, commit a transition for the oldest entry using the rewards
   remaining in the window, then `popleft()` it. All flushed transitions share the same
   final observation, and the deque is empty afterward.

The rule that sets $d_t$ for each committed transition is
`d = terminated or life_lost`. A window that fills exactly on the boundary step, in
step 2, takes this value too; any window ending earlier has $d_t = 0$.

Four rules prevent the common n-step bugs:

- **Never drop the tail before a boundary.** Without the flush, the last $n - 1$ steps,
  including the terminal signal, never reach the replay buffer.
- **Never let a window cross a boundary.** Rewards from the next life must not count
  toward a return that ended with this one. The flush guarantees this.
- **Never set $d_t$ from `truncated`.** A time limit is not a true end of the task, so a
  truncated window still bootstraps from $s_{t+m}$ with $\gamma^m$. For this reason
  $s_{t+m}$ must be the real final observation.
- **Keep raw and clipped rewards separate.** Sum raw rewards for the logged episodic
  return; only the deque receives clipped rewards.

## 5. Training components

| Component | Choice |
|---|---|
| Exploration | $\epsilon$-greedy with linear decay |
| Replay | Uniform sampling; states stored as `uint8` and normalized at sampling time |
| Reward signal | Sign-clipped per step for targets; raw for reported return |
| Training boundary | Lost life or game over; environment resets only at game over or truncation |
| Ball launch | One automatic FIRE after each reset and lost life |
| Optimizer | Adam with PyTorch defaults for the remaining parameters |
| Network initialization | PyTorch default for `Linear` layers |
| Loss | Huber (Smooth L1) |
| Telemetry | `DualLogger` (`src/utils/logger.py`) |

Replay sampling is with replacement. Numeric values are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## 6. Periodic evaluation

Mean predicted $Q$ alone cannot show overestimation, so each run also compares predicted
values with the return the policy actually earns.

**Procedure.** At each evaluation point, pause training and play complete games using the
online network $\phi$ with a small fixed $\epsilon$. Automatic FIRE stays on. Evaluation
performs no gradient updates, stores nothing in replay, and does not count toward
`total_timesteps`.

**Values compared.** For each state $s_i$ where the policy chose an action, across all games
in the evaluation:

$$\hat{V}_i = \max_a Q_{\phi}(s_i, a), \qquad R_i = \sum_{k=0}^{T_i - 1} \gamma^k r_{i+k+1}$$

where $r$ is the clipped reward and $T_i$ is the number of steps until the next lost life or
game over. This is the quantity $Q_\phi$ is trained to estimate. The logged values are the
means $\overline{\hat{V}}$ and $\overline{R}$, and the estimated bias is their difference.

**Rules.**

- **Exclude unknown returns.** If a game is truncated, exclude the states of its final life,
  because their returns are incomplete.
- **Exclude automatic presses.** The FIRE press is not an agent decision, so it adds no $s_i$.
  Its clipped reward, if any, is not part of any $R_i$.
- **Isolate randomness.** Evaluation uses its own environment and random generator, both
  seeded with `seed + 10,000`, so it does not change the course of training. The environment
  is seeded on its first reset only.

Because the policy keeps a small $\epsilon$, $\overline{R}$ slightly understates the return of
the purely greedy policy that $\max_a Q_\phi$ refers to.
