import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[3] / "scripts/research/crp_vla/evaluate_maturation_factual.py"
SPEC = importlib.util.spec_from_file_location("evaluate_maturation_factual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_factual_mse_slices_real_action_dimensions_and_executed_prefix():
    target = torch.zeros(1, 50, 7)
    predicted = torch.zeros(1, 50, 32)
    predicted[:, :10, :7] = 2
    predicted[:, 10:, :7] = 1
    predicted[:, :, 7:] = 100
    metrics = MODULE.mse_metrics(predicted, target, action_dim=7, execution_horizon=10)
    assert metrics["executed_prefix_normalized_mse"] == 4
    assert metrics["full_normalized_mse"] == pytest.approx(1.6)


def test_factual_mse_supports_frozen_raw_metric_names():
    target = torch.zeros(1, 50, 7)
    predicted = torch.ones(1, 50, 7)
    metrics = MODULE.mse_metrics(predicted, target, action_dim=7, prefix="raw_7d")
    assert metrics == {"full_raw_7d_mse": 1.0, "executed_prefix_raw_7d_mse": 1.0}


def test_fixed_noise_and_time_are_seed_deterministic():
    class Config:
        chunk_size = 50
        max_action_dim = 32

    class Policy:
        config = Config()

    batch = {"observation.state": torch.zeros(1, 32)}
    left = MODULE.fixed_noise_time(Policy(), batch, 7)
    right = MODULE.fixed_noise_time(Policy(), batch, 7)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])


def test_fixture_action_fields_gain_batch_dimension_without_double_batching_observations():
    batch = {
        "observation.state": torch.zeros(1, 8),
        "action": torch.zeros(50, 7),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
    }
    result = MODULE.ensure_action_batch_dimension(batch)
    assert result["observation.state"].shape == (1, 8)
    assert result["action"].shape == (1, 50, 7)
    assert result["action_is_pad"].shape == (1, 50)
