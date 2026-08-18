"""Matched-noise factual/counterfactual evaluation helpers."""

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

DEFAULT_CONDITION_KEYS = frozenset({OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, "task"})


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def assert_matched_batches(
    factual: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    *,
    condition_keys: Iterable[str] = DEFAULT_CONDITION_KEYS,
) -> None:
    """Require two processed batches to differ only in declared condition keys."""
    condition_keys = set(condition_keys)
    factual_keys = set(factual) - condition_keys
    counterfactual_keys = set(counterfactual) - condition_keys
    if factual_keys != counterfactual_keys:
        raise ValueError(
            "Factual/counterfactual non-condition keys differ: "
            f"only factual={sorted(factual_keys - counterfactual_keys)}, "
            f"only counterfactual={sorted(counterfactual_keys - factual_keys)}"
        )
    for key in sorted(factual_keys):
        left = factual[key]
        right = counterfactual[key]
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                raise ValueError(f"Matched-pair invariant failed for tensor key {key!r}")
        elif left != right:
            raise ValueError(f"Matched-pair invariant failed for key {key!r}")


@torch.no_grad()
def evaluate_condition_pair(
    policy,
    factual_batch: Mapping[str, Any],
    counterfactual_batch: Mapping[str, Any],
    noise: Tensor,
    *,
    num_steps: int,
    target_time: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate a condition pair with identical noise and sampling settings."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if target_time is not None and num_steps != 1:
        raise ValueError("A fixed target_time is only valid for 1-NFE inference")
    assert_matched_batches(factual_batch, counterfactual_batch)

    original_steps = policy.config.num_steps
    kwargs = {} if target_time is None else {"target_time": target_time}
    try:
        policy.config.num_steps = num_steps
        policy.reset()
        factual_actions = policy.predict_action_chunk(dict(factual_batch), noise=noise.clone(), **kwargs)
        policy.reset()
        counterfactual_actions = policy.predict_action_chunk(
            dict(counterfactual_batch), noise=noise.clone(), **kwargs
        )
    finally:
        policy.config.num_steps = original_steps
        policy.reset()
    return factual_actions, counterfactual_actions
