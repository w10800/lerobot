from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT_DIR = ROOT / "scripts/research/crp_vla"
SPEC = importlib.util.spec_from_file_location("task9_common", SCRIPT_DIR / "task9_common.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def confirmation_case(case_id: str = "confirm-task-00-state-05") -> dict:
    return {
        "case_id": case_id,
        "task_id": "libero_spatial:0",
        "task_name": "task",
        "initial_state_source": "fixture[5]",
        "initial_state_hash": f"initial-{case_id}",
        "simulator_state_hash": f"sim-{case_id}",
        "qpos_qvel_hash": f"q-{case_id}",
        "observation_hash": f"obs-{case_id}",
        "processor_hash": "processor",
        "evaluator_hash": "evaluator",
        "runtime_metadata": {},
        "noise_schedule_hash": f"noise-{case_id}",
        "formal100_overlap_check": False,
        "dev40_overlap_check": False,
    }


def test_manifest_schema_accepts_complete_disjoint_case():
    MODULE.validate_confirmation_manifest({"schema_version": 1, "cases": [confirmation_case()]})


def test_state_overlap_rejection():
    case = confirmation_case()
    with pytest.raises(ValueError, match="formal100 state overlap"):
        MODULE.assert_no_state_overlap([case], {case["simulator_state_hash"]}, set())


def test_model_hash_mismatch_rejection(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Model SHA-256 mismatch"):
        MODULE.require_model_hash(model)


def test_incomplete_run_aggregation_rejection():
    records = [{"case_id": "c1", "arm": "base10", "status": "COMPLETED", "success": True}]
    with pytest.raises(RuntimeError, match="1/2 records"):
        MODULE.guarded_aggregate(records, {"c1"}, {"base10", "snap1_20k"})


def test_duplicate_trace_path_rejection():
    records = [
        {"trace_manifest": {"path": "same", "sha256": "hash-a"}},
        {"trace_manifest": {"path": "same", "sha256": "hash-b"}},
    ]
    with pytest.raises(ValueError, match="Duplicate trace manifest path"):
        MODULE.detect_duplicate_trace_paths_and_hashes(records)


def test_duplicate_result_hash_detection():
    records = [
        {"trace_manifest": {"path": "a", "sha256": "same"}},
        {"trace_manifest": {"path": "b", "sha256": "same"}},
    ]
    with pytest.raises(ValueError, match="Duplicate trace manifest hash"):
        MODULE.detect_duplicate_trace_paths_and_hashes(records)


def test_arm_input_equality_rejects_noise_mismatch():
    base = {
        "arm": "base10",
        "initial_sim_state_sha256": "state",
        "canonical_input_sha256": "input",
        "noise_schedule_sha256": "noise-a",
        "evaluator_sha256": "eval",
    }
    snap = {**base, "arm": "snap1", "noise_schedule_sha256": "noise-b"}
    with pytest.raises(ValueError, match="noise_schedule_sha256"):
        MODULE.validate_arm_input_equality([base, snap], {"base10", "snap1"})


def test_diagnostic_subset_is_deterministic_and_task_balanced():
    cases = []
    for task in ("suite:0", "suite:1"):
        for index in range(6):
            case = confirmation_case(f"{task}-{index}")
            case["task_id"] = task
            cases.append(case)
    first = MODULE.deterministic_diagnostic_subset(cases, per_task=5, seed=42)
    second = MODULE.deterministic_diagnostic_subset(list(reversed(cases)), per_task=5, seed=42)
    assert first == second
    assert first["case_count"] == 10
