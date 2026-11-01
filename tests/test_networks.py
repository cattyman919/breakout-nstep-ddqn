"""Q-network shapes, parameter count and input validation."""

import numpy as np
import pytest
import torch

from src.exceptions import ConfigurationError
from src.networks import NetworkConfig, QNetwork, normalize_observation


def test_output_shape_for_single_state() -> None:
    assert QNetwork()(torch.zeros(128)).shape == (4,)


def test_output_shape_for_batch() -> None:
    assert QNetwork()(torch.zeros(64, 128)).shape == (64, 4)


def test_parameter_count_matches_architecture_doc() -> None:
    assert sum(p.numel() for p in QNetwork().parameters() if p.requires_grad) == 99_844


@pytest.mark.parametrize("shape", [(127,), (2, 127), (1, 2, 128), ()])
def test_rejects_invalid_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        QNetwork()(torch.zeros(shape))


def test_rejects_unnormalized_integer_input() -> None:
    with pytest.raises(ValueError, match="float32"):
        QNetwork()(torch.zeros(128, dtype=torch.uint8))


def test_rejects_invalid_config() -> None:
    with pytest.raises(ConfigurationError, match="hidden_size"):
        NetworkConfig(hidden_size=0)


def test_normalize_observation_maps_bytes_to_unit_range() -> None:
    normalized = normalize_observation(np.array([0, 51, 255], dtype=np.uint8))
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [0.0, 0.2, 1.0])
