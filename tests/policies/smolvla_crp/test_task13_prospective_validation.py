from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from task13_common import validate_dev_b_manifest  # noqa: E402


def manifest() -> dict:
    cases = []
    for task in range(40):
        for state in range(10):
            index = task * 10 + state
            cases.append(
                {
                    "task_key": f"suite:{task}",
                    "initial_state_hash": f"initial-{index}",
                    "simulator_state_hash": f"sim-{index}",
                    "qpos_qvel_hash": f"q-{index}",
                    "outcome_accessed": False,
                }
            )
    return {
        "status": "TASK13_DEV_B_FROZEN",
        "case_count": 400,
        "registered_arms": ["base10", "snap1"],
        "cases": cases,
    }


def test_dev_b_requires_balanced_unique_outcome_blind_cases() -> None:
    validate_dev_b_manifest(manifest())


def test_dev_b_rejects_duplicate_state_identity() -> None:
    payload = manifest()
    payload["cases"][1]["simulator_state_hash"] = payload["cases"][0]["simulator_state_hash"]
    with pytest.raises(ValueError, match="duplicate identity"):
        validate_dev_b_manifest(payload)
