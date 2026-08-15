#!/usr/bin/env python
"""Fail-closed pure helpers for CRP-VLA Task 10.

This module cannot load a policy, step a simulator, start a rollout, or train a
model.  It freezes and validates the 1,200-case confirmation contract and
computes raw transition-divergence components for diagnostic branches.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from task9_common import (
    NONINFERIORITY_MARGIN,
    PRIMARY_ARMS,
    SELECTED_MODEL_SHA256,
    SELECTED_STEP,
    canonical_json_sha256,
    deterministic_diagnostic_subset,
)

PRIMARY_CASES = 1_200
TASK_COUNT = 40
CASES_PER_TASK = 30
DIAGNOSTIC_CASES = 200
DIAGNOSTIC_PER_TASK = 5
DIAGNOSTIC_SEED = 20260816
COUNTERFACTUAL_HORIZONS = (1, 3, 5, 10)

STATE_IDENTITY_FIELDS = (
    "initial_state_hash",
    "simulator_state_hash",
    "qpos_qvel_hash",
)


def validate_confirmation1200_manifest(record: Mapping[str, Any]) -> None:
    """Validate the exact frozen Task 10 cardinality and state uniqueness."""
    if record.get("schema_version") != 2:
        raise ValueError("Task 10 confirmation manifest schema mismatch")
    if record.get("status") != "CONFIRMATION1200_MANIFEST_FROZEN":
        raise ValueError("Task 10 confirmation manifest is not frozen")
    if int(record.get("selected_checkpoint_step", -1)) != SELECTED_STEP:
        raise ValueError("Selected checkpoint step changed")
    if record.get("selected_model_sha256") != SELECTED_MODEL_SHA256:
        raise ValueError("Selected model SHA-256 changed")
    if float(record.get("noninferiority_margin", math.nan)) != NONINFERIORITY_MARGIN:
        raise ValueError("Non-inferiority margin changed")

    cases = record.get("cases")
    if not isinstance(cases, list) or len(cases) != PRIMARY_CASES:
        observed = len(cases) if isinstance(cases, list) else None
        raise ValueError(f"Task 10 manifest must contain exactly {PRIMARY_CASES} cases, got {observed}")
    if int(record.get("case_count", -1)) != PRIMARY_CASES:
        raise ValueError("Task 10 manifest case_count mismatch")
    if int(record.get("task_count", -1)) != TASK_COUNT:
        raise ValueError("Task 10 manifest task_count mismatch")
    if int(record.get("cases_per_task", -1)) != CASES_PER_TASK:
        raise ValueError("Task 10 manifest cases_per_task mismatch")

    required = {
        "case_id",
        "task_id",
        "task_name",
        "initial_state_id",
        "initial_state_source",
        "initial_state_hash",
        "simulator_state_hash",
        "qpos_qvel_hash",
        "observation_hash",
        "processor_hash",
        "evaluator_hash",
        "runtime_metadata",
        "noise_schedule_hash",
        "formal100_overlap_check",
        "dev40_overlap_check",
        "task9_600_membership",
    }
    for case in cases:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Task 10 confirmation case missing fields: {missing}")
        if case["formal100_overlap_check"] or case["dev40_overlap_check"]:
            raise ValueError(f"Task 10 confirmation overlap detected: {case['case_id']}")

    by_task = Counter(str(case["task_id"]) for case in cases)
    if len(by_task) != TASK_COUNT or set(by_task.values()) != {CASES_PER_TASK}:
        raise ValueError(f"Task-balanced 40x30 design violated: {dict(sorted(by_task.items()))}")

    for field in ("case_id", *STATE_IDENTITY_FIELDS):
        values = [str(case[field]) for case in cases]
        if len(values) != len(set(values)):
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            raise ValueError(f"Duplicate Task 10 {field}: {duplicates[:5]}")

    old_cases = [case for case in cases if case["task9_600_membership"]]
    new_cases = [case for case in cases if not case["task9_600_membership"]]
    if len(old_cases) != 600 or len(new_cases) != 600:
        raise ValueError(f"Task 9 subset/new split must be 600/600, got {len(old_cases)}/{len(new_cases)}")
    for field in STATE_IDENTITY_FIELDS:
        old_hashes = {str(case[field]) for case in old_cases}
        new_hashes = {str(case[field]) for case in new_cases}
        if old_hashes & new_hashes:
            raise ValueError(f"New 600 overlap Task 9 600 on {field}")


def validate_task9_subset(
    task10_cases: Iterable[Mapping[str, Any]], task9_cases: Iterable[Mapping[str, Any]]
) -> None:
    """Require every Task 9 case to appear byte-semantically unchanged."""
    task10_by_id = {str(case["case_id"]): case for case in task10_cases}
    task9_cases = list(task9_cases)
    if len(task9_cases) != 600:
        raise ValueError(f"Expected the frozen Task 9 600-case manifest, got {len(task9_cases)}")
    for old_case in task9_cases:
        case_id = str(old_case["case_id"])
        candidate = task10_by_id.get(case_id)
        if candidate is None:
            raise ValueError(f"Task 9 subset case missing: {case_id}")
        projected = {key: candidate.get(key) for key in old_case}
        if canonical_json_sha256(projected) != canonical_json_sha256(dict(old_case)):
            raise ValueError(f"Task 9 subset case changed: {case_id}")
        if not candidate.get("task9_600_membership"):
            raise ValueError(f"Task 9 subset membership flag missing: {case_id}")


def assert_no_external_state_overlap(
    cases: Iterable[Mapping[str, Any]],
    *,
    formal_hashes: set[str],
    development_hashes: set[str],
) -> None:
    """Reject any initial/simulator/qpos identity found in prior supports."""
    for case in cases:
        identities = {str(case[field]) for field in STATE_IDENTITY_FIELDS}
        if identities & formal_hashes:
            raise ValueError(f"formal100 state overlap: {case['case_id']}")
        if identities & development_hashes:
            raise ValueError(f"dev40 state overlap: {case['case_id']}")


def confirmation1200_diagnostic_subset(cases: list[dict[str, Any]]) -> dict[str, Any]:
    subset = deterministic_diagnostic_subset(
        cases,
        per_task=DIAGNOSTIC_PER_TASK,
        seed=DIAGNOSTIC_SEED,
    )
    subset.update(
        {
            "schema_version": 2,
            "status": "CONFIRMATION1200_DIAGNOSTIC_SUBSET_FROZEN",
            "outcome_blind": True,
            "confirmation_outcomes_available_at_freeze": False,
            "primary_aggregation_eligible": False,
        }
    )
    if subset["case_count"] != DIAGNOSTIC_CASES:
        raise ValueError(f"Diagnostic subset must contain {DIAGNOSTIC_CASES} cases")
    return subset


def strict_primary_aggregate(run_records: list[dict[str, Any]], primary_case_ids: set[str]) -> dict[str, Any]:
    """Aggregate only a complete 1,200-case, two-arm primary result table.

    Unlike the Task 9 preflight helper, this refuses any extra diagnostic arm
    or non-primary case instead of silently filtering it.
    """
    if len(primary_case_ids) != PRIMARY_CASES:
        raise RuntimeError(
            f"Primary aggregation refused: frozen primary manifest has {len(primary_case_ids)}/{PRIMARY_CASES} cases"
        )
    expected_arms = set(PRIMARY_ARMS)
    contamination = [
        (record.get("case_id"), record.get("arm"))
        for record in run_records
        if record.get("case_id") not in primary_case_ids or record.get("arm") not in expected_arms
    ]
    if contamination:
        raise RuntimeError(
            f"Primary aggregation refused: diagnostic/non-primary contamination {contamination[:3]}"
        )
    expected = PRIMARY_CASES * len(expected_arms)
    if len(run_records) != expected:
        completed_cases = len(
            {
                record.get("case_id")
                for record in run_records
                if record.get("status") == "COMPLETED"
                and record.get("case_id") in primary_case_ids
                and record.get("arm") in expected_arms
            }
        )
        raise RuntimeError(
            f"Primary aggregation refused: completed_primary_cases={completed_cases}/{PRIMARY_CASES}; "
            f"records={len(run_records)}/{expected}"
        )
    keys = [(str(record["case_id"]), str(record["arm"])) for record in run_records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Primary aggregation refused: duplicate case/arm record")
    if any(record.get("status") != "COMPLETED" for record in run_records):
        raise RuntimeError("Primary aggregation refused: FAILED/INVALID/incomplete primary record")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        by_case[str(record["case_id"])].append(record)
    if set(by_case) != primary_case_ids or any(
        {str(record["arm"]) for record in records} != expected_arms for records in by_case.values()
    ):
        raise RuntimeError("Primary aggregation refused: exact paired-arm coverage violated")

    base = np.asarray(
        [
            next(record for record in by_case[case_id] if record["arm"] == "base10")["success"]
            for case_id in sorted(by_case)
        ],
        dtype=np.int8,
    )
    snap = np.asarray(
        [
            next(record for record in by_case[case_id] if record["arm"] == "snap1_20k")["success"]
            for case_id in sorted(by_case)
        ],
        dtype=np.int8,
    )
    return {
        "status": "PRIMARY1200_AGGREGATED",
        "case_count": PRIMARY_CASES,
        "base10_successes": int(base.sum()),
        "snap1_20k_successes": int(snap.sum()),
        "paired_success_difference": float((snap - base).mean()),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "diagnostic_records_consumed": 0,
    }


def quaternion_rotation_distance(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> float:
    """Shortest geodesic quaternion distance in radians, stable to q/-q."""
    left = np.asarray(left_xyzw, dtype=np.float64)
    right = np.asarray(right_xyzw, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("Orientation quaternions must have shape (4,) in xyzw convention")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0 or not np.isfinite(left_norm + right_norm):
        raise ValueError("Orientation quaternions must be finite and non-zero")
    cosine = abs(float(np.dot(left / left_norm, right / right_norm)))
    return float(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)))


def _available_metric(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "available": value is not None,
        "value": None if value is None else float(value),
        "unavailable_reason": reason,
    }


def transition_divergence(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compute unweighted raw transition-divergence components."""
    q_left = left.get("qpos")
    q_right = right.get("qpos")
    joint = (
        _available_metric(float(np.linalg.norm(np.asarray(q_left) - np.asarray(q_right))))
        if q_left is not None and q_right is not None
        else _available_metric(None, "qpos unavailable")
    )

    eef_left = left.get("end_effector_pose") or {}
    eef_right = right.get("end_effector_pose") or {}
    pos_left = eef_left.get("pos")
    pos_right = eef_right.get("pos")
    position = (
        _available_metric(float(np.linalg.norm(np.asarray(pos_left) - np.asarray(pos_right))))
        if pos_left is not None and pos_right is not None
        else _available_metric(None, "end-effector position unavailable")
    )
    quat_left = eef_left.get("quat")
    quat_right = eef_right.get("quat")
    orientation = (
        _available_metric(quaternion_rotation_distance(np.asarray(quat_left), np.asarray(quat_right)))
        if quat_left is not None and quat_right is not None
        else _available_metric(None, "end-effector orientation unavailable")
    )

    grip_left = left.get("gripper_state")
    grip_right = right.get("gripper_state")
    gripper = (
        _available_metric(float(np.linalg.norm(np.asarray(grip_left) - np.asarray(grip_right))))
        if grip_left is not None and grip_right is not None
        else _available_metric(None, "gripper state unavailable")
    )

    objects_left = left.get("object_states") or {}
    objects_right = right.get("object_states") or {}
    shared_objects = sorted(set(objects_left) & set(objects_right))
    per_object: dict[str, Any] = {}
    for name in shared_objects:
        lstate = objects_left[name]
        rstate = objects_right[name]
        if "pos" not in lstate or "pos" not in rstate:
            per_object[name] = _available_metric(None, "object position unavailable")
            continue
        position_value = float(
            np.linalg.norm(
                np.asarray(lstate["pos"], dtype=np.float64) - np.asarray(rstate["pos"], dtype=np.float64)
            )
        )
        item: dict[str, Any] = {"available": True, "position_l2": position_value}
        if "quat" in lstate and "quat" in rstate:
            item["orientation_radians"] = quaternion_rotation_distance(
                np.asarray(lstate["quat"]), np.asarray(rstate["quat"])
            )
        else:
            item["orientation_radians"] = None
            item["orientation_unavailable_reason"] = "object orientation unavailable"
        per_object[name] = item
    object_metric = {
        "available": bool(per_object),
        "per_object": per_object,
        "unavailable_reason": None if per_object else "no shared task-relevant object state",
    }
    return {
        "joint_state_l2": joint,
        "end_effector_position_l2": position,
        "end_effector_orientation_radians": orientation,
        "gripper_l2": gripper,
        "object_state": object_metric,
        "combined_weighted_scalar": {
            "available": False,
            "value": None,
            "unavailable_reason": "not preregistered; Task 10 logs raw components only",
        },
    }
