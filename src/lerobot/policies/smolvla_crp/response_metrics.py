"""Conditional-response metrics for matched factual/counterfactual actions."""

from typing import Any

import torch
from torch import Tensor


def _validate_actions(name: str, value: Tensor, reference: Tensor | None = None) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (batch, horizon, action_dim), got {value.shape}")
    if reference is not None and value.shape != reference.shape:
        raise ValueError(f"{name} shape {value.shape} does not match {reference.shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def compute_response_metrics(
    teacher_factual: Tensor,
    teacher_counterfactual: Tensor,
    student_factual: Tensor,
    student_counterfactual: Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute global, per-sample, and horizon-wise CRP metrics.

    Global norm ratio and normalized error use a ratio of mean L2 norms, matching
    the project definition. Cosine is averaged over examples after flattening the
    action chunk. Horizon-wise values apply the same definitions at each position.
    """
    _validate_actions("teacher_factual", teacher_factual)
    for name, value in (
        ("teacher_counterfactual", teacher_counterfactual),
        ("student_factual", student_factual),
        ("student_counterfactual", student_counterfactual),
    ):
        _validate_actions(name, value, teacher_factual)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    delta_teacher = teacher_factual - teacher_counterfactual
    delta_student = student_factual - student_counterfactual
    batch_size = delta_teacher.shape[0]

    flat_teacher = delta_teacher.reshape(batch_size, -1)
    flat_student = delta_student.reshape(batch_size, -1)
    teacher_norm = torch.linalg.vector_norm(flat_teacher, dim=-1)
    student_norm = torch.linalg.vector_norm(flat_student, dim=-1)
    error_norm = torch.linalg.vector_norm(flat_student - flat_teacher, dim=-1)
    cosine = torch.sum(flat_student * flat_teacher, dim=-1) / (student_norm * teacher_norm + epsilon)

    teacher_horizon_norm = torch.linalg.vector_norm(delta_teacher, dim=-1)
    student_horizon_norm = torch.linalg.vector_norm(delta_student, dim=-1)
    error_horizon_norm = torch.linalg.vector_norm(delta_student - delta_teacher, dim=-1)
    horizon_cosine = torch.sum(delta_student * delta_teacher, dim=-1) / (
        student_horizon_norm * teacher_horizon_norm + epsilon
    )

    return {
        "delta_teacher": delta_teacher,
        "delta_student": delta_student,
        "per_sample": {
            "teacher_response_norm": teacher_norm,
            "student_response_norm": student_norm,
            "response_cosine": cosine,
            "response_error_norm": error_norm,
        },
        "global": {
            "response_norm_ratio": student_norm.mean() / (teacher_norm.mean() + epsilon),
            "response_cosine": cosine.mean(),
            "normalized_response_error": error_norm.mean() / (teacher_norm.mean() + epsilon),
        },
        "horizon": {
            "response_norm_ratio": student_horizon_norm.mean(dim=0)
            / (teacher_horizon_norm.mean(dim=0) + epsilon),
            "response_cosine": horizon_cosine.mean(dim=0),
            "normalized_response_error": error_horizon_norm.mean(dim=0)
            / (teacher_horizon_norm.mean(dim=0) + epsilon),
        },
    }
