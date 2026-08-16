from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_task13_recovery import overlap_rows  # noqa: E402
from task13_recovery_common import (  # noqa: E402
    clustering_units,
    reject_attempt001_primary,
    require_complete_attempt002,
    require_registry_hash,
    validate_frozen_cases,
    validate_task12_registry,
)


def registry() -> dict:
    return {
        "status": "USED_STATE_REGISTRY_COMPLETE",
        "task12_authoritative_query_record_count": 2,
        "task12_unique_state_payload_hash_count": 2,
        "authoritative_sources": {"task12_query_shards": {"paths": ["raw.jsonl"]}},
        "entries": [
            {
                "dataset": "task12_mechanism_query",
                "identity_type": "state_blob_sha256",
                "identity_value": "task12-state",
            },
            {
                "dataset": "task13_attempt001_initial",
                "identity_type": "initial_state_hash",
                "identity_value": "attempt1-initial",
            },
        ],
    }


def case(index: int = 0) -> dict:
    return {
        "case_id": f"case-{index}",
        "task_key": f"task:{index // 5}",
        "initial_state_hash": f"initial-{index}",
        "simulator_state_hash": f"sim-{index}",
        "qpos_qvel_hash": f"q-{index}",
        "outcome_accessed": False,
    }


def test_task12_raw_registry_is_nonempty() -> None:
    validate_task12_registry(registry())


def test_expected_authoritative_task12_source_is_required() -> None:
    value = registry()
    value["authoritative_sources"] = {}
    with pytest.raises(ValueError, match="authoritative"):
        validate_task12_registry(value)


def test_zero_checked_states_cannot_pass() -> None:
    value = registry()
    value["task12_unique_state_payload_hash_count"] = 0
    with pytest.raises(ValueError, match="Zero checked"):
        validate_task12_registry(value)


def test_overlap_auditor_rejects_injected_task12_state() -> None:
    value = case()
    value["simulator_state_hash"] = "task12-state"
    assert overlap_rows([value], registry())[0]["dataset"] == "task12_mechanism_or_probe"


def test_overlap_auditor_rejects_attempt001_state() -> None:
    value = case()
    value["initial_state_hash"] = "attempt1-initial"
    assert overlap_rows([value], registry())[0]["dataset"] == "task13_attempt001_initial"


def test_duplicate_dev_b2_candidate_fails() -> None:
    rows = [case(i) for i in range(10)]
    rows[1]["simulator_state_hash"] = rows[0]["simulator_state_hash"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frozen_cases(rows, 10, 5)


def test_manifest_must_be_frozen_before_outcomes() -> None:
    rows = [case(i) for i in range(10)]
    rows[0]["outcome_accessed"] = True
    with pytest.raises(ValueError, match="before outcome"):
        validate_frozen_cases(rows, 10, 5)


def test_registry_hash_drift_fails(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("original")
    expected = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash drift"):
        require_registry_hash(path, expected)


def test_attempt001_cannot_be_primary() -> None:
    with pytest.raises(ValueError, match="Attempt001"):
        reject_attempt001_primary({"primary_source": "attempt001"})


def test_incomplete_attempt002_cannot_be_analyzed() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        require_complete_attempt002({"status": "RUNNING_PARTIAL", "results": []})


def test_replan_analysis_preserves_case_and_task_clustering() -> None:
    rows = [
        {"case_id": "a", "task_id": "t1"},
        {"case_id": "a", "task_id": "t1"},
        {"case_id": "b", "task_id": "t2"},
    ]
    assert clustering_units(rows) == {"state_count": 3, "case_count": 2, "task_count": 2}
