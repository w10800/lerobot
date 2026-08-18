"""Shared reduction rules for auditable SmolVLA loss components."""

import torch
from torch import Tensor


def reduce_action_loss(
    losses: Tensor,
    *,
    action_dim: int,
    actions_is_pad: Tensor | None = None,
) -> Tensor:
    """Reduce one element-wise loss component exactly like SmolVLA's scalar loss.

    SnapFlow produces losses over ``max_action_dim`` while a dataset may expose
    fewer physical action dimensions. Padding in the temporal action chunk must
    also be excluded from every logged component, otherwise the displayed
    component values cannot be reconciled with the optimized scalar loss.
    """
    if losses.ndim != 3:
        raise ValueError(f"losses must have shape (batch, horizon, action_dim), got {losses.shape}")
    if not 1 <= action_dim <= losses.shape[-1]:
        raise ValueError(f"action_dim must be in [1, {losses.shape[-1]}], got {action_dim}")

    selected = losses[:, :, :action_dim]
    if actions_is_pad is None:
        return selected.mean()
    if actions_is_pad.shape != selected.shape[:2]:
        raise ValueError(
            f"actions_is_pad shape {actions_is_pad.shape} does not match {selected.shape[:2]}"
        )
    valid = ~actions_is_pad.to(dtype=torch.bool)
    selected = selected * valid.unsqueeze(-1)
    denominator = (valid.sum() * action_dim).clamp_min(1)
    return selected.sum() / denominator
