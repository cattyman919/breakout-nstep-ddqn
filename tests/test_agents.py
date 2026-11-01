"""Targets, Polyak averaging, gradient stability and action selection."""

import math

import numpy as np
import pytest
import torch

from src.agents import AgentConfig, DQNAgent
from src.buffers import TransitionBatch
from src.exceptions import ConfigurationError, TrainingError


def make_agent(*, double_q: bool = True, tau: float = 0.005, gamma: float = 0.5) -> DQNAgent:
    torch.manual_seed(0)
    config = AgentConfig(gamma=gamma, tau=tau, double_q=double_q)
    return DQNAgent(config, torch.device("cpu"))


def make_batch(
    *,
    size: int = 1,
    returns: float = 1.0,
    dones: float = 0.0,
    spans: float = 1.0,
) -> TransitionBatch:
    return TransitionBatch(
        states=torch.zeros(size, 128),
        actions=torch.zeros(size, dtype=torch.int64),
        n_step_returns=torch.full((size,), returns),
        next_states=torch.ones(size, 128),
        dones=torch.full((size,), dones),
        spans=torch.full((size,), spans),
    )


class ScriptedQ(torch.nn.Module):
    """A network returning fixed action-values, so targets can be worked out by hand."""

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.expand(states.shape[0], -1)


def script_networks(agent: DQNAgent, online: list[float], target: list[float]) -> None:
    agent.online = ScriptedQ(online)  # type: ignore[assignment]
    agent.target = ScriptedQ(target)  # type: ignore[assignment]


def test_dqn_target_selects_and_evaluates_with_target_network() -> None:
    agent = make_agent(double_q=False, gamma=0.5)
    script_networks(agent, online=[9.0, 0.0, 0.0, 0.0], target=[1.0, 2.0, 8.0, 3.0])
    # 1.0 + 0.5 * max(target) = 1.0 + 0.5 * 8.0
    assert agent.compute_targets(make_batch()).item() == pytest.approx(5.0)


def test_double_dqn_target_selects_online_and_evaluates_target() -> None:
    agent = make_agent(double_q=True, gamma=0.5)
    script_networks(agent, online=[9.0, 0.0, 0.0, 0.0], target=[1.0, 2.0, 8.0, 3.0])
    # The online network picks action 0, evaluated by the target network as 1.0.
    assert agent.compute_targets(make_batch()).item() == pytest.approx(1.0 + 0.5 * 1.0)


def test_target_discount_uses_the_span_not_n() -> None:
    agent = make_agent(double_q=True, gamma=0.5)
    script_networks(agent, online=[1.0, 0.0, 0.0, 0.0], target=[8.0, 0.0, 0.0, 0.0])
    # gamma^3 * 8.0 = 1.0 on top of the 3-step return of 2.0.
    assert agent.compute_targets(make_batch(returns=2.0, spans=3.0)).item() == pytest.approx(3.0)


@pytest.mark.parametrize("double_q", [False, True])
def test_done_removes_the_bootstrap(double_q: bool) -> None:
    agent = make_agent(double_q=double_q)
    script_networks(agent, online=[9.0, 0.0, 0.0, 0.0], target=[9.0, 0.0, 0.0, 0.0])
    assert agent.compute_targets(make_batch(returns=1.5, dones=1.0)).item() == pytest.approx(1.5)


def test_targets_carry_no_gradient() -> None:
    assert not make_agent().compute_targets(make_batch()).requires_grad


def test_target_network_starts_as_a_copy_and_is_not_trainable() -> None:
    agent = make_agent()
    for online, target in zip(agent.online.parameters(), agent.target.parameters(), strict=True):
        assert torch.equal(online, target)
        assert target is not online
        assert not target.requires_grad


def test_polyak_update_weights_online_by_tau() -> None:
    agent = make_agent(tau=0.25)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.fill_(4.0)
        for parameter in agent.target.parameters():
            parameter.fill_(0.0)

    agent._soft_update_target()

    # phi' = 0.25 * 4.0 + 0.75 * 0.0, and the online network stays put.
    for parameter in agent.target.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 1.0))
    for parameter in agent.online.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 4.0))


def test_update_changes_online_network_and_keeps_values_finite() -> None:
    agent = make_agent()
    before = [parameter.clone() for parameter in agent.online.parameters()]

    result = agent.update(make_batch(size=8), step=1)

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online.parameters(), strict=True)
    )
    assert math.isfinite(result.td_loss)
    assert math.isfinite(result.mean_q)
    assert all(torch.isfinite(parameter).all() for parameter in agent.online.parameters())


def test_update_moves_target_network_only_slightly() -> None:
    agent = make_agent(tau=0.005)
    before = [parameter.clone() for parameter in agent.target.parameters()]
    online_before = [parameter.clone() for parameter in agent.online.parameters()]

    agent.update(make_batch(size=8), step=1)

    for old, new, online_old, online_new in zip(
        before,
        agent.target.parameters(),
        online_before,
        agent.online.parameters(),
        strict=True,
    ):
        assert torch.allclose(new, 0.995 * old + 0.005 * online_new)
        assert (new - old).abs().max() <= (online_new - online_old).abs().max()


def test_mean_q_comes_from_the_batch_states_before_the_update() -> None:
    agent = make_agent()
    batch = make_batch(size=8)
    with torch.no_grad():
        expected = agent.online(batch.states).max(dim=1).values.mean().item()
    assert agent.update(batch, step=1).mean_q == pytest.approx(expected)


def test_huber_loss_is_linear_beyond_delta() -> None:
    agent = make_agent()
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.fill_(0.0)
        for parameter in agent.target.parameters():
            parameter.fill_(0.0)
    # The error is 3.0, so Huber(delta=1) = 1.0 * (3.0 - 0.5).
    result = agent.update(make_batch(size=2, returns=3.0, dones=1.0), step=1)
    assert result.td_loss == pytest.approx(2.5)


def test_nan_loss_raises_naming_the_step_and_leaves_weights_unchanged() -> None:
    agent = make_agent()
    before = [parameter.clone() for parameter in agent.online.parameters()]

    with pytest.raises(TrainingError, match="loss.*step 1234"):
        agent.update(make_batch(returns=math.nan), step=1234)

    for old, new in zip(before, agent.online.parameters(), strict=True):
        assert torch.equal(old, new)


def test_nan_gradient_raises_naming_the_parameter_and_step() -> None:
    agent = make_agent()
    next(agent.online.parameters()).register_hook(lambda grad: grad * math.nan)

    with pytest.raises(TrainingError, match="gradient in layers.0.weight at step 77"):
        agent.update(make_batch(size=4), step=77)


def test_act_is_greedy_at_zero_epsilon() -> None:
    agent = make_agent()
    observation = np.zeros(128, dtype=np.uint8)
    agent.online.forward = lambda states: torch.tensor([0.0, 0.0, 5.0, 0.0])  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    assert agent.act(observation, 0.0, rng) == 2
    assert agent.act_with_value(observation, 0.0, rng) == (2, 5.0)


def test_act_explores_uniformly_at_full_epsilon() -> None:
    agent = make_agent()
    observation = np.zeros(128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert {agent.act(observation, 1.0, rng) for _ in range(100)} == {0, 1, 2, 3}


def test_exploring_value_is_still_the_greedy_value() -> None:
    agent = make_agent()
    observation = np.zeros(128, dtype=np.uint8)
    agent.online.forward = lambda states: torch.tensor([1.0, 7.0, 2.0, 3.0])  # type: ignore[method-assign]
    _, value = agent.act_with_value(observation, 1.0, np.random.default_rng(0))
    assert value == 7.0


@pytest.mark.parametrize(
    "kwargs",
    [{"gamma": 0.0}, {"tau": 0.0}, {"tau": 1.5}, {"learning_rate": 0.0}, {"huber_delta": -1.0}],
)
def test_invalid_agent_config_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigurationError):
        AgentConfig(**kwargs)  # type: ignore[arg-type]
