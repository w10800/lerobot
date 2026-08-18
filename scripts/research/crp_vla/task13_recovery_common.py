"""Fail-closed pure gates for Task 13 recovery."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from task9_common import file_sha256


def validate_task12_registry(registry: dict[str, Any]) -> None:
    if registry.get("status") != "USED_STATE_REGISTRY_COMPLETE":
        raise ValueError("Used-state registry is incomplete")
    if int(registry.get("task12_authoritative_query_record_count", 0)) <= 0:
        raise ValueError("Task 12 authoritative raw state source is empty")
    if int(registry.get("task12_unique_state_payload_hash_count", 0)) <= 0:
        raise ValueError("Zero checked Task 12 states cannot pass")
    sources = registry.get("authoritative_sources", {})
    if not sources.get("task12_query_shards", {}).get("paths"):
        raise ValueError("Expected authoritative Task 12 query source was not read")


def require_registry_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError("Used-state registry hash drift")


def validate_frozen_cases(cases: list[dict[str, Any]], expected: int, per_task: int) -> None:
    if len(cases) != expected:
        raise ValueError("Frozen case cardinality mismatch")
    if set(Counter(str(row["task_key"]) for row in cases).values()) != {per_task}:
        raise ValueError("Frozen cases are not task balanced")
    for field in ("initial_state_hash", "simulator_state_hash", "qpos_qvel_hash"):
        values = [str(row[field]) for row in cases]
        if len(set(values)) != expected:
            raise ValueError(f"Duplicate Dev-B2 candidate: {field}")
    if any(row.get("outcome_accessed") is not False for row in cases):
        raise ValueError("Manifest was not frozen before outcome access")


def reject_attempt001_primary(record: dict[str, Any]) -> None:
    if record.get("primary_source") == "attempt001" or record.get("attempt") == "attempt001":
        raise ValueError("Attempt001 cannot be accepted by the primary Task 13 aggregator")


def require_complete_attempt002(record: dict[str, Any]) -> None:
    if record.get("status") != "DEVELOPMENT_COMPLETED" or len(record.get("results", [])) != 400:
        raise ValueError("Attempt002 is incomplete")


def clustering_units(rows: list[dict[str, Any]]) -> dict[str, int]:
    cases = {str(row["case_id"]) for row in rows}
    tasks = {str(row["task_id"]) for row in rows}
    if not cases or not tasks:
        raise ValueError("Case/task clustering units are missing")
    return {"state_count": len(rows), "case_count": len(cases), "task_count": len(tasks)}
