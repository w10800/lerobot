#!/usr/bin/env python
"""Pure fail-closed helpers for CRP-VLA Task 11.

This module validates immutable execution/result records and performs the
already frozen paired non-inferiority aggregation.  It cannot load a model,
step a simulator, or start a rollout.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from task9_common import (
    NONINFERIORITY_MARGIN,
    PRIMARY_ARMS,
    SELECTED_MODEL_SHA256,
    bootstrap_interval,
    canonical_json_sha256,
    exact_mcnemar,
    file_sha256,
)
from task10_common import PRIMARY_CASES, validate_confirmation1200_manifest

TASK10_MANIFEST_SHA256 = "01962a5255f4bbc51647a1f81a328838910f440577f4e063f48e5be723d59654"
BASE_ARM = "base10"
SNAP_ARM = "snap1_20k"
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260815
RESULT_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1

REQUIRED_RESULT_FIELDS = {
    "schema_version",
    "status",
    "case_id",
    "task_id",
    "arm",
    "model_id",
    "model_hash",
    "manifest_hash",
    "initial_state_hash",
    "simulator_state_hash",
    "qpos_qvel_hash",
    "evaluator_hash",
    "success",
    "termination_reason",
    "trace_path",
    "trace_hash",
    "attempt_id",
    "result_path",
    "result_hash",
}


def result_payload_hash(record: Mapping[str, Any]) -> str:
    """Hash a result without its self-referential result_hash field."""
    return canonical_json_sha256({key: value for key, value in record.items() if key != "result_hash"})


def write_hashed_json(path: Path, record: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["result_path"] = str(path.resolve())
    payload["result_hash"] = result_payload_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["result_hash"]


def validate_execution_manifest(record: Mapping[str, Any], frozen_manifest: Mapping[str, Any]) -> None:
    validate_confirmation1200_manifest(frozen_manifest)
    if record.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        raise ValueError("Task 11 execution manifest schema mismatch")
    if record.get("status") != "TASK11_EXECUTION_MANIFEST_FROZEN":
        raise ValueError("Task 11 execution manifest is not frozen")
    if record.get("confirmation1200_manifest_sha256") != TASK10_MANIFEST_SHA256:
        raise ValueError("Task 10 manifest hash drift")
    if record.get("selected_model_sha256") != SELECTED_MODEL_SHA256:
        raise ValueError("Selected model hash drift")
    if float(record.get("noninferiority_margin")) != NONINFERIORITY_MARGIN:
        raise ValueError("Non-inferiority margin drift")
    if tuple(record.get("primary_arms", ())) != PRIMARY_ARMS:
        raise ValueError("Primary arm set/order drift")
    if int(record.get("case_count", -1)) != PRIMARY_CASES:
        raise ValueError("Execution manifest case count drift")
    shards = record.get("shards")
    if not isinstance(shards, list) or len(shards) != 40:
        raise ValueError("Task 11 requires exactly 40 task shards")
    shard_cases = [case_id for shard in shards for case_id in shard.get("case_ids", [])]
    frozen_ids = sorted(str(case["case_id"]) for case in frozen_manifest["cases"])
    if sorted(shard_cases) != frozen_ids or len(shard_cases) != len(set(shard_cases)):
        raise ValueError("Execution shard coverage differs from frozen cases")
    if any(int(shard.get("case_count", -1)) != 30 for shard in shards):
        raise ValueError("Every Task 11 shard must contain 30 cases")


def validate_result_record(
    record: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    arm: str,
    base_model_hash: str,
    verify_files: bool = True,
) -> None:
    missing = sorted(REQUIRED_RESULT_FIELDS - set(record))
    if missing:
        raise ValueError(f"Task 11 result missing fields: {missing}")
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Task 11 result schema mismatch")
    if record.get("status") != "COMPLETED":
        raise ValueError("Task 11 result is not completed")
    if record.get("case_id") != case.get("case_id") or record.get("task_id") != case.get("task_id"):
        raise ValueError("Task 11 case identity mismatch")
    if arm not in PRIMARY_ARMS or record.get("arm") != arm:
        raise ValueError("Task 11 arm identity mismatch")
    expected_model = base_model_hash if arm == BASE_ARM else SELECTED_MODEL_SHA256
    if record.get("model_hash") != expected_model:
        raise ValueError("Task 11 model hash mismatch")
    if record.get("manifest_hash") != TASK10_MANIFEST_SHA256:
        raise ValueError("Task 11 result manifest hash mismatch")
    for key in ("initial_state_hash", "simulator_state_hash", "qpos_qvel_hash", "evaluator_hash"):
        if record.get(key) != case.get(key):
            raise ValueError(f"Task 11 result {key} mismatch")
    if not isinstance(record.get("success"), bool):
        raise ValueError("Task 11 success must be boolean")
    if record.get("result_hash") != result_payload_hash(record):
        raise ValueError("Task 11 result payload hash mismatch")
    if verify_files:
        result_path = Path(str(record["result_path"]))
        trace_path = Path(str(record["trace_path"]))
        if not result_path.is_file() or not trace_path.is_file():
            raise ValueError("Task 11 result or trace manifest is missing")
        loaded = json.loads(result_path.read_text())
        if loaded != dict(record):
            raise ValueError("Task 11 result file content mismatch")
        if file_sha256(trace_path) != record.get("trace_hash"):
            raise ValueError("Task 11 trace manifest hash mismatch")


def completion_audit(
    records: list[dict[str, Any]],
    frozen_manifest: Mapping[str, Any],
    *,
    base_model_hash: str,
    diagnostic_case_ids: set[str],
    verify_files: bool = True,
) -> dict[str, Any]:
    validate_confirmation1200_manifest(frozen_manifest)
    cases = {str(case["case_id"]): case for case in frozen_manifest["cases"]}
    expected_keys = {(case_id, arm) for case_id in cases for arm in PRIMARY_ARMS}
    observed_keys = [(str(record.get("case_id")), str(record.get("arm"))) for record in records]
    duplicates = sorted(key for key, count in Counter(observed_keys).items() if count > 1)
    missing = sorted(expected_keys - set(observed_keys))
    extras = sorted(set(observed_keys) - expected_keys)
    if len(records) != PRIMARY_CASES * 2 or duplicates or missing or extras:
        raise RuntimeError(
            "FORMAL_CONFIRMATION_INVALID: cardinality/pairing failure "
            f"records={len(records)}/2400 duplicates={len(duplicates)} missing={len(missing)} extras={len(extras)}"
        )
    result_paths: list[str] = []
    trace_paths: list[str] = []
    result_hashes: list[str] = []
    trace_hashes: list[str] = []
    for record in records:
        case_id = str(record["case_id"])
        arm = str(record["arm"])
        validate_result_record(
            record,
            case=cases[case_id],
            arm=arm,
            base_model_hash=base_model_hash,
            verify_files=verify_files,
        )
        result_paths.append(str(record["result_path"]))
        trace_paths.append(str(record["trace_path"]))
        result_hashes.append(str(record["result_hash"]))
        trace_hashes.append(str(record["trace_hash"]))
    uniqueness = {
        "result_paths": len(result_paths) == len(set(result_paths)),
        "trace_paths": len(trace_paths) == len(set(trace_paths)),
        "result_hashes": len(result_hashes) == len(set(result_hashes)),
        "trace_hashes": len(trace_hashes) == len(set(trace_hashes)),
    }
    if not all(uniqueness.values()):
        raise RuntimeError(f"FORMAL_CONFIRMATION_INVALID: non-unique result/trace evidence {uniqueness}")
    if any(record.get("diagnostic_derived_feature_used", False) for record in records):
        raise RuntimeError("FORMAL_CONFIRMATION_INVALID: diagnostic-derived feature contamination")
    by_case: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_case[str(record["case_id"])].add(str(record["arm"]))
    if set(by_case) != set(cases) or any(arms != set(PRIMARY_ARMS) for arms in by_case.values()):
        raise RuntimeError("FORMAL_CONFIRMATION_INVALID: exact paired coverage failure")
    return {
        "status": "TASK11_COMPLETION_AUDIT_PASSED",
        "records": len(records),
        "base_results": sum(record["arm"] == BASE_ARM for record in records),
        "snap_results": sum(record["arm"] == SNAP_ARM for record in records),
        "exact_pairs": len(by_case),
        "duplicate_case_arm": 0,
        "missing_case_arm": 0,
        "formal100_contamination": sum(bool(cases[cid]["formal100_overlap_check"]) for cid in cases),
        "dev40_contamination": sum(bool(cases[cid]["dev40_overlap_check"]) for cid in cases),
        "diagnostic_derived_feature_contamination": 0,
        "diagnostic_identity_overlap_count": len(set(cases) & diagnostic_case_ids),
        "diagnostic_cases_remain_primary_cases_without_diagnostic_features": True,
        "unique_evidence": uniqueness,
    }


def frozen_primary_statistics(
    records: Iterable[Mapping[str, Any]], frozen_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the frozen paired percentile bootstrap only after completion audit."""
    cases = {str(case["case_id"]): case for case in frozen_manifest["cases"]}
    by_case: dict[str, dict[str, bool]] = defaultdict(dict)
    for record in records:
        by_case[str(record["case_id"])][str(record["arm"])] = bool(record["success"])
    if set(by_case) != set(cases) or any(set(arms) != set(PRIMARY_ARMS) for arms in by_case.values()):
        raise RuntimeError("FORMAL_CONFIRMATION_INVALID: primary aggregation coverage mismatch")
    ordered = sorted(cases)
    base = np.asarray([by_case[case_id][BASE_ARM] for case_id in ordered], dtype=np.int8)
    snap = np.asarray([by_case[case_id][SNAP_ARM] for case_id in ordered], dtype=np.int8)
    differences = snap.astype(np.float64) - base.astype(np.float64)
    interval = bootstrap_interval(differences, BOOTSTRAP_REPEATS, BOOTSTRAP_SEED)
    mcnemar = exact_mcnemar(snap, base)
    n11 = int(np.sum((snap == 1) & (base == 1)))
    n10 = int(np.sum((snap == 1) & (base == 0)))
    n01 = int(np.sum((snap == 0) & (base == 1)))
    n00 = int(np.sum((snap == 0) & (base == 0)))
    if n11 + n10 + n01 + n00 != PRIMARY_CASES:
        raise RuntimeError("FORMAL_CONFIRMATION_INVALID: contingency table cardinality mismatch")
    passed = interval[0] >= NONINFERIORITY_MARGIN
    return {
        "status": "FORMAL_CONFIRMATION1200_PASS" if passed else "FORMAL_CONFIRMATION1200_FAIL",
        "case_count": PRIMARY_CASES,
        "base10_success_count": int(base.sum()),
        "base10_success_rate": float(base.mean()),
        "snap1_20k_success_count": int(snap.sum()),
        "snap1_20k_success_rate": float(snap.mean()),
        "paired_contingency": {"n_11": n11, "n_10": n10, "n_01": n01, "n_00": n00},
        "paired_success_difference": float(differences.mean()),
        "case_paired_percentile_bootstrap_ci95": interval,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_pass": passed,
        "exact_mcnemar_two_sided_p": mcnemar["two_sided_exact_p"],
        "diagnostic_records_consumed": 0,
        "frozen_protocol_modified": False,
    }


def task_breakdown(
    records: Iterable[Mapping[str, Any]], frozen_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cases = {str(case["case_id"]): case for case in frozen_manifest["cases"]}
    by_case: dict[str, dict[str, bool]] = defaultdict(dict)
    for record in records:
        by_case[str(record["case_id"])][str(record["arm"])] = bool(record["success"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, case in cases.items():
        grouped[str(case["task_id"])].append(case_id)
    output = []
    for task_id in sorted(grouped):
        ids = sorted(grouped[task_id])
        base = sum(by_case[case_id][BASE_ARM] for case_id in ids)
        snap = sum(by_case[case_id][SNAP_ARM] for case_id in ids)
        output.append(
            {
                "task_id": task_id,
                "cases": len(ids),
                "base10_successes": base,
                "snap1_20k_successes": snap,
                "paired_difference": (snap - base) / len(ids),
            }
        )
    return output
