#!/usr/bin/env python
"""Pure helpers for the Task 14 position-component causal audit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

TASK14_NAME = "TASK14_PROSPECTIVE_POSITION_CAUSAL_AUDIT"
TASK14_SEED_STRING = "CRP-VLA-TASK14-POSITION-CAUSAL-V1"
TASK14_HORIZONS = (1, 3, 5, 10)
TASK14_ARMS = ("noop", "pos_swap", "rot_swap", "full_swap")
ACTION_DIM = 7
POSITION_INDICES = (0, 1, 2)
ORIENTATION_INDICES = (3, 4, 5)
GRIPPER_INDICES = (6,)


def derive_seed(value: str = TASK14_SEED_STRING) -> dict[str, str | int]:
    """Derive the registered non-negative 63-bit integer seed from SHA-256."""
    digest = hashlib.sha256(value.encode()).hexdigest()
    integer = int.from_bytes(bytes.fromhex(digest)[:8], byteorder="big") & ((1 << 63) - 1)
    return {"string": value, "sha256": digest, "integer": integer}


def _validate_chunks(base: Any, snap: Any) -> None:
    if type(base) is not type(snap):
        raise TypeError("Base and Snap chunks must use the same array type")
    if tuple(base.shape) != tuple(snap.shape):
        raise ValueError("Base and Snap chunks must have identical shapes")
    if base.ndim < 2 or int(base.shape[-1]) != ACTION_DIM:
        raise ValueError(f"Expected action chunks with final dimension {ACTION_DIM}")
    if not isinstance(base, np.ndarray | torch.Tensor):
        raise TypeError("Action chunks must be NumPy arrays or Torch tensors")


def compose_action_chunks(base: Any, snap: Any, arm: str) -> Any:
    """Compose aligned denormalized action chunks without changing chunk indices."""
    _validate_chunks(base, snap)
    if arm not in TASK14_ARMS:
        raise ValueError(f"Unknown Task 14 arm: {arm}")
    if arm == "full_swap":
        return base.clone() if isinstance(base, torch.Tensor) else base.copy()
    output = snap.clone() if isinstance(snap, torch.Tensor) else snap.copy()
    if arm == "pos_swap":
        output[..., list(POSITION_INDICES)] = base[..., list(POSITION_INDICES)]
    elif arm == "rot_swap":
        output[..., list(ORIENTATION_INDICES)] = base[..., list(ORIENTATION_INDICES)]
    return output


def controller_clipping_audit(
    actions: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
) -> dict[str, Any]:
    """Describe Robosuite controller-input clipping without hiding excursions."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected actions with final dimension {ACTION_DIM}")
    if not np.isfinite(values).all():
        raise ValueError("Action clipping audit received a non-finite value")
    clipped = np.clip(values, low, high)
    difference = clipped - values
    per_dimension = []
    for index in range(ACTION_DIM):
        changed = difference[..., index] != 0
        per_dimension.append(
            {
                "dimension": index,
                "count": int(np.sum(changed)),
                "maximum_absolute_magnitude": float(np.max(np.abs(difference[..., index]), initial=0.0)),
            }
        )
    return {
        "bounds": [low, high],
        "total_count": int(np.sum(difference != 0)),
        "per_dimension": per_dimension,
        "clipped": clipped,
    }


def collect_used_initial_state_ids(
    sources: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[tuple[str, int], set[int]], list[dict[str, Any]]]:
    """Collect typed initial-state IDs without pooling unrelated payload hashes."""
    used: dict[tuple[str, int], set[int]] = {}
    source_summary = []
    for name, rows in sources:
        row_count = 0
        unique_cases: set[tuple[str, int, int]] = set()
        for row in rows:
            suite = str(row["suite"])
            task_id = int(row["suite_task_id"] if "suite_task_id" in row else row["task_id"])
            state_id = int(row.get("init_state_id", row.get("initial_state_id")))
            used.setdefault((suite, task_id), set()).add(state_id)
            unique_cases.add((suite, task_id, state_id))
            row_count += 1
        source_summary.append(
            {
                "name": name,
                "row_count": row_count,
                "unique_initial_cases": len(unique_cases),
                "minimum_state_id": min((item[2] for item in unique_cases), default=None),
                "maximum_state_id": max((item[2] for item in unique_cases), default=None),
            }
        )
    return used, source_summary


def audit_formal_state_capacity(
    available_counts: Mapping[tuple[str, int], int],
    used_ids: Mapping[tuple[str, int], set[int]],
    required_per_task: int = 10,
) -> dict[str, Any]:
    """Audit whether every task has the fixed number of untouched init states."""
    task_rows = []
    for key in sorted(available_counts):
        total = int(available_counts[key])
        used = sorted(index for index in used_ids.get(key, set()) if 0 <= index < total)
        untouched = sorted(set(range(total)) - set(used))
        task_rows.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "available_state_count": total,
                "used_state_count": len(used),
                "untouched_state_count": len(untouched),
                "untouched_state_ids": untouched,
                "required_untouched_state_count": required_per_task,
                "eligible": len(untouched) >= required_per_task,
            }
        )
    eligible_tasks = sum(bool(row["eligible"]) for row in task_rows)
    status = (
        "READY_FOR_FORMAL_MANIFEST"
        if task_rows and eligible_tasks == len(task_rows)
        else "INSUFFICIENT_UNTOUCHED_FORMAL_STATES"
    )
    return {
        "status": status,
        "task_count": len(task_rows),
        "eligible_task_count": eligible_tasks,
        "required_cases_per_task": required_per_task,
        "required_case_count": len(task_rows) * required_per_task,
        "available_untouched_case_count": sum(int(row["untouched_state_count"]) for row in task_rows),
        "minimum_untouched_per_task": min(
            (int(row["untouched_state_count"]) for row in task_rows), default=0
        ),
        "maximum_untouched_per_task": max(
            (int(row["untouched_state_count"]) for row in task_rows), default=0
        ),
        "tasks": task_rows,
    }
