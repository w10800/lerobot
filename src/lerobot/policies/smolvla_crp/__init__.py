"""CRP-VLA primitives for one-step SmolVLA distillation and diagnostics."""

from .paired_inputs import assert_matched_batches, evaluate_condition_pair, tensor_sha256
from .response_metrics import compute_response_metrics
from .snapflow_loss import SnapFlowLossOutput, compute_snapflow_losses
from .target_time import ZeroInitTargetTimeMLP

__all__ = [
    "SnapFlowLossOutput",
    "ZeroInitTargetTimeMLP",
    "assert_matched_batches",
    "compute_response_metrics",
    "compute_snapflow_losses",
    "evaluate_condition_pair",
    "tensor_sha256",
]
