import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/research/crp_vla"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task11_common import (  # noqa: E402
    BASE_ARM,
    SNAP_ARM,
    TASK10_MANIFEST_SHA256,
    completion_audit,
    frozen_primary_statistics,
    result_payload_hash,
)


def _manifest() -> dict:
    cases = []
    for index in range(1200):
        task = index // 30
        cases.append(
            {
                "case_id": f"case-{index:04d}",
                "task_id": f"task-{task:02d}",
                "task_name": f"task-{task:02d}",
                "initial_state_id": index,
                "initial_state_source": "fixture",
                "initial_state_hash": f"initial-{index}",
                "simulator_state_hash": f"sim-{index}",
                "qpos_qvel_hash": f"q-{index}",
                "observation_hash": f"obs-{index}",
                "processor_hash": "processor",
                "evaluator_hash": "evaluator",
                "runtime_metadata": {},
                "noise_schedule_hash": f"noise-{index}",
                "formal100_overlap_check": False,
                "dev40_overlap_check": False,
                "task9_600_membership": index < 600,
            }
        )
    return {
        "schema_version": 2,
        "status": "CONFIRMATION1200_MANIFEST_FROZEN",
        "selected_checkpoint_step": 20_000,
        "selected_model_sha256": "3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53",
        "noninferiority_margin": -0.03,
        "case_count": 1200,
        "task_count": 40,
        "cases_per_task": 30,
        "cases": cases,
    }


def _records(manifest: dict, base_success: int = 1000, snap_success: int = 1000) -> list[dict]:
    records = []
    for index, case in enumerate(manifest["cases"]):
        for arm, success in ((BASE_ARM, index < base_success), (SNAP_ARM, index < snap_success)):
            record = {
                "schema_version": 1,
                "status": "COMPLETED",
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "arm": arm,
                "model_id": arm,
                "model_hash": "basehash" if arm == BASE_ARM else manifest["selected_model_sha256"],
                "manifest_hash": TASK10_MANIFEST_SHA256,
                "initial_state_hash": case["initial_state_hash"],
                "simulator_state_hash": case["simulator_state_hash"],
                "qpos_qvel_hash": case["qpos_qvel_hash"],
                "evaluator_hash": case["evaluator_hash"],
                "success": success,
                "termination_reason": "SUCCESS" if success else "MAX_STEPS",
                "trace_path": f"/trace/{case['case_id']}/{arm}",
                "trace_hash": f"trace-{case['case_id']}-{arm}",
                "attempt_id": "attempt-0001",
                "result_path": f"/result/{case['case_id']}/{arm}",
                "diagnostic_derived_feature_used": False,
            }
            record["result_hash"] = result_payload_hash(record)
            records.append(record)
    return records


def test_completion_audit_requires_exact_2400_pairs() -> None:
    manifest = _manifest()
    records = _records(manifest)
    result = completion_audit(
        records,
        manifest,
        base_model_hash="basehash",
        diagnostic_case_ids=set(),
        verify_files=False,
    )
    assert result["records"] == 2400
    assert result["exact_pairs"] == 1200
    with pytest.raises(RuntimeError, match="cardinality/pairing"):
        completion_audit(
            records[:-1],
            manifest,
            base_model_hash="basehash",
            diagnostic_case_ids=set(),
            verify_files=False,
        )


def test_completion_audit_rejects_duplicate_and_hash_drift() -> None:
    manifest = _manifest()
    records = _records(manifest)
    with pytest.raises(RuntimeError, match="cardinality/pairing"):
        completion_audit(
            records[:-1] + [records[0]],
            manifest,
            base_model_hash="basehash",
            diagnostic_case_ids=set(),
            verify_files=False,
        )
    records[0]["model_hash"] = "drift"
    records[0]["result_hash"] = result_payload_hash(records[0])
    with pytest.raises(ValueError, match="model hash"):
        completion_audit(
            records,
            manifest,
            base_model_hash="basehash",
            diagnostic_case_ids=set(),
            verify_files=False,
        )


def test_frozen_primary_statistics_uses_snap_minus_base_and_margin() -> None:
    manifest = _manifest()
    records = _records(manifest, base_success=1000, snap_success=1000)
    stats = frozen_primary_statistics(records, manifest)
    assert stats["paired_success_difference"] == 0.0
    assert stats["paired_contingency"] == {"n_11": 1000, "n_10": 0, "n_01": 0, "n_00": 200}
    assert stats["noninferiority_pass"] is True


def test_result_hash_is_self_field_independent() -> None:
    record = {"a": 1, "result_hash": "old"}
    assert result_payload_hash(record) == result_payload_hash({"a": 1, "result_hash": "new"})
    assert json.loads(json.dumps(record)) == record
