import pytest
import torch

from lerobot.policies.smolvla_crp.loss_accounting import reduce_action_loss
from lerobot.policies.smolvla_crp.snapflow_loss import compute_snapflow_losses


class ScalarVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x_t, target_time, current_time):
        conditioning = (current_time - target_time)[:, None, None]
        return self.scale * x_t + conditioning


def test_shortcut_target_matches_two_step_equations_and_is_detached():
    predictor = ScalarVelocity()
    actions = torch.zeros(1, 1, 1)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.25])
    output = compute_snapflow_losses(
        predictor, actions, noise, time, alpha=0.5, shortcut_weight=0.1, prediction_clamp=None
    )
    # v1=2, x0.5=0, v0.5=0, so the two-step average is 1.
    assert torch.equal(output.shortcut_target, torch.ones_like(actions))
    assert not output.shortcut_target.requires_grad
    assert output.one_step_velocity.requires_grad


def test_alpha_endpoints_reduce_to_expected_objective():
    predictor = ScalarVelocity()
    actions = torch.zeros(2, 2, 1)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.25, 0.75])
    fm_only = compute_snapflow_losses(
        predictor, actions, noise, time, alpha=1.0, shortcut_weight=0.1, prediction_clamp=None
    )
    shortcut_only = compute_snapflow_losses(
        predictor, actions, noise, time, alpha=0.0, shortcut_weight=0.1, prediction_clamp=None
    )
    assert torch.equal(fm_only.combined, fm_only.flow_matching)
    assert torch.allclose(shortcut_only.combined, 0.1 * shortcut_only.shortcut)


def test_only_student_and_fm_paths_contribute_gradients():
    predictor = ScalarVelocity()
    actions = torch.zeros(1, 1, 1)
    noise = torch.ones_like(actions)
    output = compute_snapflow_losses(
        predictor,
        actions,
        noise,
        torch.tensor([0.5]),
        alpha=0.5,
        shortcut_weight=0.1,
        prediction_clamp=None,
    )
    output.combined.mean().backward()
    assert predictor.scale.grad is not None
    assert torch.isfinite(predictor.scale.grad)


def test_reduced_components_reconstruct_total_with_padding_and_action_crop():
    predictor = ScalarVelocity()
    actions = torch.zeros(2, 3, 4)
    noise = torch.ones_like(actions)
    alpha = 0.5
    shortcut_weight = 0.1
    output = compute_snapflow_losses(
        predictor,
        actions,
        noise,
        torch.tensor([0.25, 0.75]),
        alpha=alpha,
        shortcut_weight=shortcut_weight,
        prediction_clamp=None,
    )
    actions_is_pad = torch.tensor([[False, False, True], [False, True, True]])
    reduced_total = reduce_action_loss(
        output.combined, action_dim=2, actions_is_pad=actions_is_pad
    )
    reduced_fm = reduce_action_loss(
        output.flow_matching, action_dim=2, actions_is_pad=actions_is_pad
    )
    reduced_shortcut = reduce_action_loss(
        output.shortcut, action_dim=2, actions_is_pad=actions_is_pad
    )
    accounted = alpha * reduced_fm + (1.0 - alpha) * shortcut_weight * reduced_shortcut
    assert torch.allclose(reduced_total, accounted, atol=1e-7, rtol=0.0)


def test_loss_reducer_rejects_mismatched_padding_shape():
    losses = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="does not match"):
        reduce_action_loss(losses, action_dim=2, actions_is_pad=torch.zeros(2, 2, dtype=torch.bool))
