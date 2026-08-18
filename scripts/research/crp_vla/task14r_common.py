#!/usr/bin/env python
"""Pure helpers for TASK14R fresh-state recovery and engineering audits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

TASK14R_NAME = "TASK14R_FRESH_STATE_BANK_RECOVERY"
TASK14R_EVIDENCE_LABELS = ("ENGINEERING_ONLY", "OUTCOME_CONDITIONED", "NON_PRIMARY")
TASK14R_STATE_SEED_STRING = "CRP-VLA-TASK14R-FRESH-STATE-BANK-V1"
TASK14R_PHASE_A_ARMS = (
    "base_repeat",
    "snap_repeat",
    "noop",
    "pos_swap",
    "rot_swap",
    "full_swap",
)
TASK14R_CANDIDATES_PER_TASK = 20
TASK14R_FORMAL_PER_TASK = 10
TASK14R_CAPACITY_PROBE_COUNT = 64
TASK14R_CAPACITY_REQUIRED_VALID = 51

_ACTION_DIM = 7
_POSITION_INDICES = (0, 1, 2)
_ORIENTATION_INDICES = (3, 4, 5)
_FORBIDDEN_SELECTION_KEYS = {
    "base_success",
    "snap_success",
    "success_difference",
    "action_discrepancy",
    "eef_position_discrepancy",
    "harmful",
    "outcome_label",
    "policy_score",
}


def file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def structured_seed(
    suite: str,
    task_id: int,
    candidate_index: int,
    *,
    namespace: str = "formal_candidate",
    seed_string: str = TASK14R_STATE_SEED_STRING,
) -> dict[str, str | int]:
    """Derive an outcome-independent uint32 seed from a frozen string identity."""
    identity = f"{seed_string}|{namespace}|{suite}|{int(task_id)}|{int(candidate_index)}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return {
        "identity": identity,
        "sha256": digest,
        "numpy_seed": int.from_bytes(bytes.fromhex(digest)[:4], byteorder="big"),
    }


def _validate_action_chunks(base: Any, snap: Any) -> None:
    if type(base) is not type(snap):
        raise TypeError("Base and Snap chunks must have the same array type")
    if not isinstance(base, np.ndarray | torch.Tensor):
        raise TypeError("Action chunks must be NumPy arrays or Torch tensors")
    if tuple(base.shape) != tuple(snap.shape):
        raise ValueError("Base and Snap chunks must have identical shapes")
    if base.ndim != 2 or int(base.shape[-1]) != _ACTION_DIM:
        raise ValueError("Expected aligned [horizon, 7] action chunks")


def compose_online_action_chunk(base: Any, snap: Any, arm: str) -> Any:
    """Compose an intervention after both policies queried the arm's live observation."""
    _validate_action_chunks(base, snap)
    if arm not in TASK14R_PHASE_A_ARMS:
        raise ValueError(f"Unknown TASK14R arm: {arm}")
    if arm in {"base_repeat", "full_swap"}:
        return base.clone() if isinstance(base, torch.Tensor) else base.copy()
    output = snap.clone() if isinstance(snap, torch.Tensor) else snap.copy()
    if arm == "pos_swap":
        output[..., list(_POSITION_INDICES)] = base[..., list(_POSITION_INDICES)]
    elif arm == "rot_swap":
        output[..., list(_ORIENTATION_INDICES)] = base[..., list(_ORIENTATION_INDICES)]
    return output


def clipping_record(action: Sequence[float], low: float = -1.0, high: float = 1.0) -> dict[str, Any]:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (_ACTION_DIM,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite seven-dimensional action")
    clipped = np.clip(values, low, high)
    return {
        "before": values.tolist(),
        "after": clipped.tolist(),
        "changed": (values != clipped).tolist(),
        "changed_count": int(np.count_nonzero(values != clipped)),
    }


def compare_closed_loop_steps(
    reference: Sequence[Mapping[str, Any]],
    intervention: Sequence[Mapping[str, Any]],
    *,
    action_atol: float = 0.0,
) -> dict[str, Any]:
    """Compare per-step action, state, observation, queue, and termination identity."""
    paired = min(len(reference), len(intervention))
    action_max_abs = 0.0
    first_action_mismatch = None
    first_state_mismatch = None
    first_observation_mismatch = None
    first_queue_mismatch = None
    for index in range(paired):
        left = reference[index]
        right = intervention[index]
        difference = np.abs(
            np.asarray(left["composed_action_before_clipping"], dtype=np.float64)
            - np.asarray(right["composed_action_before_clipping"], dtype=np.float64)
        )
        maximum = float(np.max(difference, initial=0.0))
        action_max_abs = max(action_max_abs, maximum)
        if first_action_mismatch is None and maximum > action_atol:
            first_action_mismatch = index
        if first_state_mismatch is None and (
            left["physical_state_before_sha256"] != right["physical_state_before_sha256"]
            or left["physical_state_after_sha256"] != right["physical_state_after_sha256"]
        ):
            first_state_mismatch = index
        if first_observation_mismatch is None and left["observation_sha256"] != right["observation_sha256"]:
            first_observation_mismatch = index
        if first_queue_mismatch is None and (
            int(left["replan_index"]) != int(right["replan_index"])
            or int(left["chunk_index"]) != int(right["chunk_index"])
        ):
            first_queue_mismatch = index
    lengths_equal = len(reference) == len(intervention)
    termination_equal = (
        bool(reference)
        and bool(intervention)
        and (reference[-1].get("termination_reason") == intervention[-1].get("termination_reason"))
    )
    return {
        "reference_step_count": len(reference),
        "intervention_step_count": len(intervention),
        "lengths_equal": lengths_equal,
        "maximum_action_absolute_difference": action_max_abs,
        "first_action_mismatch_step": first_action_mismatch,
        "first_physical_state_mismatch_step": first_state_mismatch,
        "first_observation_mismatch_step": first_observation_mismatch,
        "first_queue_alignment_mismatch_step": first_queue_mismatch,
        "termination_equal": termination_equal,
        "action_identity": first_action_mismatch is None and lengths_equal,
        "physical_state_identity": first_state_mismatch is None and lengths_equal,
        "observation_identity": first_observation_mismatch is None and lengths_equal,
        "queue_alignment_identity": first_queue_mismatch is None and lengths_equal,
        "closed_loop_identity": (
            lengths_equal
            and termination_equal
            and first_action_mismatch is None
            and first_state_mismatch is None
            and first_queue_mismatch is None
        ),
    }


def _check_selection_row(row: Mapping[str, Any]) -> None:
    if int(row.get("policy_query_count", -1)) != 0:
        raise ValueError("Fresh-state acceptance must have zero policy queries")
    if bool(row.get("outcomes_accessed", True)):
        raise ValueError("Fresh-state acceptance must not access policy outcomes")
    present_forbidden = sorted(_FORBIDDEN_SELECTION_KEYS.intersection(row))
    if present_forbidden:
        raise ValueError(f"Outcome-conditioned fields entered state selection: {present_forbidden}")
    checks = row.get("acceptance_checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("Fresh-state row lacks frozen acceptance checks")
    if bool(row.get("accepted")) != all(bool(value) for value in checks.values()):
        raise ValueError("Accepted flag does not equal the conjunction of acceptance checks")


def freeze_state_bank(
    rows: Sequence[Mapping[str, Any]],
    expected_tasks: Sequence[str],
    *,
    candidates_per_task: int = TASK14R_CANDIDATES_PER_TASK,
    formal_per_task: int = TASK14R_FORMAL_PER_TASK,
) -> dict[str, Any]:
    """Hash-sort accepted, policy-independent states without adaptive reserve activation."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    raw_hashes: set[str] = set()
    settled_hashes: set[str] = set()
    for row in rows:
        _check_selection_row(row)
        task_key = str(row["task_key"])
        grouped[task_key].append(row)
        raw_hash = str(row["raw_state_sha256"])
        settled_hash = str(row["settled_physical_state_sha256"])
        if raw_hash in raw_hashes or settled_hash in settled_hashes:
            raise ValueError("Duplicate generated state payload detected")
        raw_hashes.add(raw_hash)
        settled_hashes.add(settled_hash)

    task_records = []
    formal_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    for task_key in sorted(expected_tasks):
        candidates = grouped.get(task_key, [])
        indices = sorted(int(row["candidate_index"]) for row in candidates)
        if len(candidates) != candidates_per_task or indices != list(range(candidates_per_task)):
            raise ValueError(f"Task {task_key} does not contain the frozen candidate set")
        accepted = sorted(
            (row for row in candidates if bool(row["accepted"])),
            key=lambda row: (str(row["raw_state_sha256"]), int(row["candidate_index"])),
        )
        selected = accepted[:formal_per_task]
        reserve = accepted[formal_per_task:]
        for rank, row in enumerate(selected):
            formal_rows.append({**dict(row), "bank_role": "formal", "hash_sort_rank": rank})
        for rank, row in enumerate(reserve, start=formal_per_task):
            reserve_rows.append({**dict(row), "bank_role": "reserve", "hash_sort_rank": rank})
        task_records.append(
            {
                "task_key": task_key,
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "formal_count": len(selected),
                "reserve_count": len(reserve),
                "eligible": len(selected) == formal_per_task,
                "formal_raw_state_sha256": [str(row["raw_state_sha256"]) for row in selected],
            }
        )
    unknown_tasks = sorted(set(grouped) - set(expected_tasks))
    if unknown_tasks:
        raise ValueError(f"Unexpected generated tasks: {unknown_tasks}")
    eligible = sum(bool(row["eligible"]) for row in task_records)
    status = (
        "FRESH_STATE_BANK_400_FROZEN"
        if eligible == len(expected_tasks) and len(formal_rows) == len(expected_tasks) * formal_per_task
        else "INSUFFICIENT_VALID_GENERATED_STATES"
    )
    return {
        "status": status,
        "task_count": len(expected_tasks),
        "eligible_task_count": eligible,
        "formal_case_count": len(formal_rows),
        "reserve_case_count": len(reserve_rows),
        "selection_rule": "accepted states sorted by raw_state_sha256, then candidate_index",
        "adaptive_reserve_activation": False,
        "tasks": task_records,
        "formal_states": formal_rows,
        "reserve_states": reserve_rows,
    }


def _family(feature: str) -> str:
    return feature.split("/", 1)[0]


def _absolute_margin(feature: str) -> float:
    family = _family(feature)
    return {
        "robot_joint_pos": 0.05,
        "eef_position": 0.02,
        "eef_orientation": 0.10,
        "object_position": 0.02,
        "object_orientation": 0.10,
        "object_pair_distance": 0.02,
        "target_region_distance": 0.02,
        "contact": 0.005,
        "settling_displacement": 0.01,
        "task_initial_predicate": 0.0,
    }.get(family, 0.01)


def audit_generated_distribution(
    official_rows: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
    expected_tasks: Sequence[str],
) -> dict[str, Any]:
    """Compare generated formal states with fixed official-state support per task.

    A task/family is flagged when more than two of ten cases leave an expanded
    official support interval, or when the median standardized feature-mean
    shift in that family exceeds one official standard deviation. A global
    mismatch requires at least four of forty tasks in any feature family.
    """
    official_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    formal_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in official_rows:
        official_by_task[str(row["task_key"])].append(row)
    for row in formal_rows:
        formal_by_task[str(row["task_key"])].append(row)

    task_reports = []
    family_mismatch_tasks: dict[str, list[str]] = defaultdict(list)
    for task_key in sorted(expected_tasks):
        official = official_by_task.get(task_key, [])
        generated = formal_by_task.get(task_key, [])
        if len(official) != 50 or len(generated) != 10:
            raise ValueError(f"Distribution audit requires 50 official and 10 formal rows for {task_key}")
        feature_names = sorted(set.intersection(*(set(row["features"]) for row in official + generated)))
        if not feature_names:
            raise ValueError(f"No comparable state features for {task_key}")
        family_details: dict[str, dict[str, Any]] = {}
        for family in sorted({_family(name) for name in feature_names}):
            family_features = [name for name in feature_names if _family(name) == family]
            outlier_cases: set[str] = set()
            standardized_shifts = []
            feature_reports = []
            for feature in family_features:
                reference = np.asarray([row["features"][feature] for row in official], dtype=np.float64)
                sample = np.asarray([row["features"][feature] for row in generated], dtype=np.float64)
                if not np.isfinite(reference).all() or not np.isfinite(sample).all():
                    raise ValueError(f"Non-finite distribution feature: {task_key}/{feature}")
                span = float(reference.max() - reference.min())
                margin = max(_absolute_margin(feature), 0.10 * span)
                low = float(reference.min() - margin)
                high = float(reference.max() + margin)
                mask = (sample < low) | (sample > high)
                for row, outside in zip(generated, mask, strict=True):
                    if bool(outside):
                        outlier_cases.add(str(row["state_id"]))
                scale = max(float(reference.std(ddof=1)), _absolute_margin(feature), 1e-12)
                shift = abs(float(sample.mean() - reference.mean())) / scale
                standardized_shifts.append(shift)
                feature_reports.append(
                    {
                        "feature": feature,
                        "official_min": float(reference.min()),
                        "official_max": float(reference.max()),
                        "expanded_low": low,
                        "expanded_high": high,
                        "generated_outlier_count": int(mask.sum()),
                        "standardized_mean_shift": shift,
                    }
                )
            median_shift = float(np.median(np.asarray(standardized_shifts)))
            mismatch = len(outlier_cases) > 2 or median_shift > 1.0
            if mismatch:
                family_mismatch_tasks[family].append(task_key)
            family_details[family] = {
                "outlier_case_count": len(outlier_cases),
                "outlier_case_ids": sorted(outlier_cases),
                "median_standardized_mean_shift": median_shift,
                "mismatch": mismatch,
                "features": feature_reports,
            }
        task_reports.append({"task_key": task_key, "families": family_details})

    global_families = {
        family: {
            "mismatched_task_count": len(tasks),
            "mismatched_tasks": sorted(tasks),
            "global_mismatch": len(tasks) >= 4,
        }
        for family, tasks in sorted(family_mismatch_tasks.items())
    }
    status = (
        "GENERATED_STATE_DISTRIBUTION_MISMATCH"
        if any(row["global_mismatch"] for row in global_families.values())
        else "GENERATED_STATE_DISTRIBUTION_AUDIT_PASSED"
    )
    return {
        "status": status,
        "official_states_per_task": 50,
        "generated_formal_states_per_task": 10,
        "task_family_outlier_limit": 2,
        "task_family_median_standardized_shift_limit": 1.0,
        "global_mismatched_task_limit": 4,
        "feature_families": global_families,
        "tasks": task_reports,
    }


def terminal_status(
    *,
    bank_status: str,
    distribution_status: str,
    capacity_probe_passed: bool,
    zero_historical_overlap: bool,
    all_formal_reproducible: bool,
) -> str:
    if distribution_status == "GENERATED_STATE_DISTRIBUTION_MISMATCH":
        return "GENERATED_STATE_DISTRIBUTION_MISMATCH"
    if (
        bank_status == "FRESH_STATE_BANK_400_FROZEN"
        and distribution_status == "GENERATED_STATE_DISTRIBUTION_AUDIT_PASSED"
        and capacity_probe_passed
        and zero_historical_overlap
        and all_formal_reproducible
    ):
        return "READY_FOR_TASK14_CAUSAL_RELAUNCH"
    return "TASK14R_RECOVERY_INCOMPLETE"
