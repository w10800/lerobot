import importlib.util
from pathlib import Path

import pytest
import torch

from lerobot.utils.constants import ACTION, OBS_STATE


def load_script_module():
    path = Path(__file__).parents[3] / "scripts/research/crp_vla/verify_snapflow_loss_accounting.py"
    spec = importlib.util.spec_from_file_location("verify_snapflow_loss_accounting", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_single_sample_action_and_padding_gain_batch_dimension():
    module = load_script_module()
    batch = {
        OBS_STATE: torch.zeros(1, 8),
        ACTION: torch.zeros(50, 7),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
    }
    result = module.ensure_training_batch_dimensions(batch)
    assert result[ACTION].shape == (1, 50, 7)
    assert result["action_is_pad"].shape == (1, 50)


def test_incompatible_action_batch_is_rejected():
    module = load_script_module()
    batch = {OBS_STATE: torch.zeros(1, 8), ACTION: torch.zeros(2, 50, 7)}
    with pytest.raises(ValueError, match="incompatible"):
        module.ensure_training_batch_dimensions(batch)
