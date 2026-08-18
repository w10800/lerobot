"""Frozen constants and pure validation helpers for CRP-VLA Task 13."""

from __future__ import annotations

from collections import Counter
from typing import Any

DEV_B_FIRST_STATE = 35
DEV_B_STATES_PER_TASK = 10
DEV_B_CASES = 400
DEV_B_SEED = 20260817
TASK13_BOOTSTRAP_REPEATS = 10_000
TASK13_BOOTSTRAP_SEED = 20260817
TASK13_ARMS = ("base10", "snap1")
TASK13_HORIZONS = (1, 3, 5, 10)
SELECTED_MODEL_SHA256 = "3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53"
STATE_IDENTITY_FIELDS = ("initial_state_hash", "simulator_state_hash", "qpos_qvel_hash")


def validate_dev_b_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "TASK13_DEV_B_FROZEN":
        raise ValueError("Dev-B is not frozen")
    cases = manifest.get("cases", [])
    if len(cases) != DEV_B_CASES or manifest.get("case_count") != DEV_B_CASES:
        raise ValueError("Dev-B must contain exactly 400 cases")
    counts = Counter(str(case["task_key"]) for case in cases)
    if len(counts) != 40 or set(counts.values()) != {DEV_B_STATES_PER_TASK}:
        raise ValueError("Dev-B must contain exactly ten cases for each of 40 tasks")
    if manifest.get("registered_arms") != list(TASK13_ARMS):
        raise ValueError("Dev-B registered arms drift")
    for field in STATE_IDENTITY_FIELDS:
        values = [str(case[field]) for case in cases]
        if len(set(values)) != DEV_B_CASES:
            raise ValueError(f"Dev-B duplicate identity: {field}")
    if any(case.get("outcome_accessed") is not False for case in cases):
        raise ValueError("Dev-B candidate selection was not outcome blind")
