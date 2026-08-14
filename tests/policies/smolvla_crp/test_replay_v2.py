import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_DIR = Path(__file__).parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_module("replay_v2_common")
RUNNER = load_module("run_replay_v2")
SUMMARY = load_module("summarize_replay_v2")


def test_capsule_round_trip_preserves_arrays_tensors_and_bfloat16(tmp_path):
    payload = {
        "rgb": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "batch": {
            "image": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            "features": torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            "task": ["pick object"],
        },
    }
    expected_hash = COMMON.structured_hash(payload["batch"])
    record = COMMON.write_capsule(tmp_path / "capsule", {"schema_version": 2}, payload)
    metadata, restored = COMMON.load_capsule(tmp_path / "capsule")

    assert len(record["sha256"]) == 64
    assert metadata["schema_version"] == 2
    np.testing.assert_array_equal(restored["rgb"], payload["rgb"])
    assert restored["batch"]["features"].dtype == torch.bfloat16
    assert COMMON.structured_hash(restored["batch"]) == expected_hash


def test_capsule_detects_member_tampering(tmp_path):
    COMMON.write_capsule(tmp_path / "capsule", {}, {"state": np.arange(3)})
    metadata = tmp_path / "capsule" / "metadata.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        COMMON.load_capsule(tmp_path / "capsule")


def test_validate_case_records_enforces_all_six_invariants():
    records = [
        {
            "arm": arm,
            "status": "COMPLETED",
            "capsule_sha256": "capsule",
            "initial_sim_state_sha256": "state",
            "canonical_input_sha256": "input",
            "noise_schedule_sha256": "noise",
            "evaluator_sha256": "eval",
        }
        for arm in COMMON.ARMS
    ]
    COMMON.validate_case_records(records)
    records[-1]["canonical_input_sha256"] = "changed"
    with pytest.raises(ValueError, match="invariant mismatch"):
        COMMON.validate_case_records(records)


def test_failure_classifier_is_evidence_backed_and_conservative():
    events = {
        "first_target_contact": {"available": True, "observed": True, "step": 4},
        "first_gripper_close": {"available": True, "observed": True, "step": 5},
        "first_object_lift": {"available": True, "observed": False},
        "collision": {"available": True, "observed": False},
        "object_dropped": {"available": True, "observed": False},
    }
    assert COMMON.classify_failure(events, False)["category"] == "GRASP_FAILURE"
    events["first_target_contact"] = {"available": False, "observed": False}
    assert COMMON.classify_failure(events, False)["category"] == "UNKNOWN"


def test_task_semantics_parses_manipulated_object_and_receptacle(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text(
        "(define (problem x) (:obj_of_interest mug_1 bowl_1) "
        "(:init (On mug_1 table)) (:goal (And (In mug_1 bowl_1_contain_region))))"
    )
    semantics = RUNNER.task_semantics(bddl)
    assert semantics["manipulated_object"] == "mug_1"
    assert semantics["receptacle_or_fixture"] == "bowl_1"
    assert "In mug_1" in semantics["goal_expression"]


def write_numeric_trace(tmp_path: Path, arm: str, actions: np.ndarray) -> dict:
    numeric = tmp_path / f"{arm}.npz"
    np.savez_compressed(numeric, predicted_chunks=np.zeros((1, 50, 32)), executed_actions=actions)
    manifest = tmp_path / f"{arm}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "numeric_actions": {
                    "path": str(numeric),
                    "sha256": COMMON.file_sha256(numeric),
                }
            }
        )
    )
    return {"path": str(manifest), "sha256": COMMON.file_sha256(manifest)}


def test_minimal_six_arm_table_integration(tmp_path):
    results = []
    for index, arm in enumerate(COMMON.ARMS):
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = index / 10
        actions[5:, -1] = 1
        results.append(
            {
                "status": "COMPLETED",
                "suite": "libero_spatial",
                "task_id": 0,
                "task": "pick object",
                "init_state_id": 4,
                "env_seed": 0,
                "arm": arm,
                "success": index % 2 == 0,
                "steps_run": 10,
                "capsule_sha256": "capsule",
                "initial_sim_state_sha256": "state",
                "canonical_input_sha256": "input",
                "noise_schedule_sha256": "noise",
                "evaluator_sha256": "eval",
                "latency_ms_per_replan": [80 + index],
                "events": {
                    "first_target_contact": {"observed": False},
                    "first_gripper_close": {"observed": True, "step": 5},
                    "first_object_lift": {"observed": False},
                    "first_receptacle_entry": {"observed": False},
                    "collision": {"observed": False},
                    "object_dropped": {"observed": False},
                },
                "failure_classification": {
                    "category": None if index % 2 == 0 else "UNKNOWN",
                    "evidence": {"synthetic_fixture": True},
                },
                "trace_manifest": write_numeric_trace(tmp_path, arm, actions),
            }
        )
    rows, events = SUMMARY.build_tables({"results": results})

    assert len(rows) == 1
    assert len(events) == 6
    assert rows[0]["snap1_executed_prefix_mse_vs_base10"] > 0
    assert rows[0]["base10_executed_prefix_mse_vs_base10"] == 0


def test_registered_designs_have_disjoint_complete_coverage():
    repository = Path(__file__).parents[3]
    smoke = json.loads((repository / "configs/crp_vla/replay_v2_smoke.json").read_text())
    development = json.loads((repository / "configs/crp_vla/replay_v2_development.json").read_text())
    assert len(smoke["cases"]) == 8
    assert {case["init_state_id"] for case in smoke["cases"]} == {3}
    assert len(development["cases"]) == 40
    assert {case["init_state_id"] for case in development["cases"]} == {4}
    assert len({(case["suite"], case["task_id"]) for case in development["cases"]}) == 40
