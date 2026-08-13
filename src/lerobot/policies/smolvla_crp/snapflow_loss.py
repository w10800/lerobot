"""Equation-level SnapFlow loss primitives.

The implementation follows arXiv:2604.05656 equations (9)-(12). The two
teacher-style marginal velocity calls run under ``no_grad``; only the FM
prediction and one-step student prediction receive gradients.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class SnapFlowLossOutput:
    combined: Tensor
    flow_matching: Tensor
    shortcut: Tensor
    shortcut_target: Tensor
    one_step_velocity: Tensor


def _clamp(value: Tensor, limit: float | None) -> Tensor:
    if limit is None:
        return value
    return value.clamp(min=-limit, max=limit)


def compute_snapflow_losses(
    predict_velocity: VelocityFn,
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    *,
    alpha: float = 0.5,
    shortcut_weight: float = 0.1,
    prediction_clamp: float | None = 20.0,
) -> SnapFlowLossOutput:
    """Compute element-wise FM, shortcut, and weighted SnapFlow losses.

    ``predict_velocity(x_t, target_time, current_time)`` must return a tensor
    shaped like ``actions``. Inputs follow SmolVLA's convention where t=1 is
    noise and t=0 is data/action.
    """
    if actions.shape != noise.shape:
        raise ValueError(f"actions/noise shape mismatch: {actions.shape} != {noise.shape}")
    if time.ndim != 1 or time.shape[0] != actions.shape[0]:
        raise ValueError(f"time must have shape ({actions.shape[0]},), got {time.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if shortcut_weight < 0.0:
        raise ValueError("shortcut_weight must be non-negative")
    if prediction_clamp is not None and prediction_clamp <= 0.0:
        raise ValueError("prediction_clamp must be positive or None")

    expanded_time = time[:, None, None]
    x_t = expanded_time * noise + (1.0 - expanded_time) * actions
    conditional_velocity = noise - actions
    fm_velocity = _clamp(predict_velocity(x_t, time, time), prediction_clamp)
    fm_loss = F.mse_loss(fm_velocity, conditional_velocity, reduction="none")

    one = torch.ones_like(time)
    half = torch.full_like(time, 0.5)
    zero = torch.zeros_like(time)
    with torch.no_grad():
        velocity_one = _clamp(predict_velocity(noise, one, one), prediction_clamp)
        midpoint = noise - 0.5 * velocity_one
        velocity_half = _clamp(predict_velocity(midpoint, half, half), prediction_clamp)
        shortcut_target = 0.5 * (velocity_one + velocity_half)

    one_step_velocity = _clamp(predict_velocity(noise, zero, one), prediction_clamp)
    shortcut_loss = F.mse_loss(one_step_velocity, shortcut_target, reduction="none")
    combined = alpha * fm_loss + (1.0 - alpha) * shortcut_weight * shortcut_loss
    return SnapFlowLossOutput(
        combined=combined,
        flow_matching=fm_loss,
        shortcut=shortcut_loss,
        shortcut_target=shortcut_target,
        one_step_velocity=one_step_velocity,
    )
