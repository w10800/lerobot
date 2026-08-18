import torch

from lerobot.policies.smolvla_crp.target_time import ZeroInitTargetTimeMLP


def test_target_time_projection_is_exactly_zero_at_initialization():
    module = ZeroInitTargetTimeMLP(hidden_size=8)
    encoded = torch.randn(3, 8)
    assert torch.equal(module(encoded), torch.zeros_like(encoded))


def test_target_time_output_layer_receives_gradient_at_initialization():
    module = ZeroInitTargetTimeMLP(hidden_size=8)
    encoded = torch.randn(3, 8)
    module(encoded).sum().backward()
    assert module.out_proj.weight.grad is not None
    assert torch.count_nonzero(module.out_proj.weight.grad) > 0
    assert module.in_proj.weight.grad is not None
    assert torch.count_nonzero(module.in_proj.weight.grad) == 0
