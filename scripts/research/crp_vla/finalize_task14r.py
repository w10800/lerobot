#!/usr/bin/env python
"""Freeze the 400-case Task14R bank and issue its distribution decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import array_sha256
from task14r_common import (
    TASK14R_CAPACITY_REQUIRED_VALID,
    TASK14R_EVIDENCE_LABELS,
    TASK14R_NAME,
    audit_generated_distribution,
    freeze_state_bank,
    terminal_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-run", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def verify_payload(row: dict[str, Any]) -> None:
    path = Path(row["payload"]["path"])
    if not path.is_file() or file_sha256(path) != row["payload"]["sha256"]:
        raise RuntimeError(f"Generated state payload integrity failed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        raw_state = np.asarray(archive["raw_state"])
    if array_sha256(raw_state) != row["raw_state_sha256"]:
        raise RuntimeError(f"Generated raw-state payload hash mismatch: {path}")


def compact_bank_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": row["state_id"],
        "task_key": row["task_key"],
        "suite": row["suite"],
        "task_id": row["task_id"],
        "candidate_index": row["candidate_index"],
        "seed": row["seed"],
        "bank_role": row["bank_role"],
        "hash_sort_rank": row["hash_sort_rank"],
        "raw_state_sha256": row["raw_state_sha256"],
        "raw_state_bytes_sha256": row["raw_state_bytes_sha256"],
        "settled_state_blob_sha256": row["settled_state_blob_sha256"],
        "settled_physical_state_sha256": row["settled_physical_state_sha256"],
        "settled_render_sha256": row["settled_render_sha256"],
        "acceptance_checks": row["acceptance_checks"],
        "policy_query_count": row["policy_query_count"],
        "outcomes_accessed": row["outcomes_accessed"],
        "payload": {
            "path": portable_path(row["payload"]["path"]),
            "sha256": row["payload"]["sha256"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before finalizing Task14R")
    phase_a = json.loads(args.phase_a_run.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    if phase_a.get("evidence_labels") != list(TASK14R_EVIDENCE_LABELS):
        raise RuntimeError("Task14R Phase A evidence labels drift")
    if int(phase_a.get("formal_outcome_rollout_count", -1)) != 0:
        raise RuntimeError("Task14R must not contain formal Base/Snap outcomes")
    if seeds.get("status") != "TASK14R_STATE_SEEDS_FROZEN":
        raise RuntimeError("Task14R seed manifest is not frozen")

    expected_tasks = [str(row["task_key"]) for row in seeds["tasks"]]
    candidates: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    official: list[dict[str, Any]] = []
    shard_records = []
    for task in seeds["tasks"]:
        path = args.shards_root / str(task["suite"]) / f"task{int(task['task_id']):02d}" / "manifest.json"
        shard = json.loads(path.read_text())
        if shard.get("status") != "TASK14R_STATE_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete Task14R state shard: {path}")
        if shard.get("task_key") != task["task_key"]:
            raise RuntimeError(f"Task14R shard task mismatch: {path}")
        if int(shard.get("policy_query_count", -1)) != 0 or bool(shard.get("outcomes_accessed")):
            raise RuntimeError(f"Policy/outcome access entered Task14R state generation: {path}")
        if int(shard.get("formal_outcome_rollout_count", -1)) != 0:
            raise RuntimeError(f"Formal outcome was revealed in Task14R shard: {path}")
        if len(shard["candidates"]) != 20 or len(shard["official_references"]) != 50:
            raise RuntimeError(f"Task14R shard cardinality drift: {path}")
        for row in [*shard["candidates"], *shard["capacity_probes"]]:
            verify_payload(row)
        candidates.extend(shard["candidates"])
        probes.extend(shard["capacity_probes"])
        official.extend(shard["official_references"])
        shard_records.append(
            {
                "task_key": task["task_key"],
                "path": portable_path(path),
                "sha256": file_sha256(path),
                "accepted_candidate_count": shard["accepted_candidate_count"],
                "accepted_capacity_probe_count": shard["accepted_capacity_probe_count"],
            }
        )

    if len(candidates) != 800 or len(official) != 2_000:
        raise RuntimeError("Task14R full-shard cardinality drift")
    all_generated = [*candidates, *probes]
    raw_hashes = [str(row["raw_state_sha256"]) for row in all_generated]
    settled_hashes = [str(row["settled_physical_state_sha256"]) for row in all_generated]
    globally_unique = len(raw_hashes) == len(set(raw_hashes)) and len(settled_hashes) == len(
        set(settled_hashes)
    )
    if not globally_unique:
        raise RuntimeError("Cross-shard generated-state duplication detected")

    bank = freeze_state_bank(candidates, expected_tasks)
    formal_rows = bank["formal_states"]
    formal_reproducible = all(bool(row["acceptance_checks"]["reset_reproducible"]) for row in formal_rows)
    zero_historical_overlap = all(
        bool(row["acceptance_checks"]["zero_historical_overlap"]) for row in formal_rows
    )
    probe_reports = []
    for task_key in seeds["capacity_probe"]["tasks"]:
        rows = [row for row in probes if row["task_key"] == task_key]
        accepted = [row for row in rows if bool(row["accepted"])]
        passed = (
            len(rows) == int(seeds["capacity_probe"]["candidate_states_per_probe_task"])
            and len(accepted) >= TASK14R_CAPACITY_REQUIRED_VALID
            and len({row["raw_state_sha256"] for row in accepted}) == len(accepted)
            and all(row["acceptance_checks"]["reset_reproducible"] for row in accepted)
        )
        probe_reports.append(
            {
                "task_key": task_key,
                "candidate_count": len(rows),
                "accepted_unique_reproducible_count": len(accepted),
                "required_count": TASK14R_CAPACITY_REQUIRED_VALID,
                "passed": passed,
            }
        )
    capacity_passed = len(probe_reports) == 4 and all(row["passed"] for row in probe_reports)

    if bank["status"] == "FRESH_STATE_BANK_400_FROZEN":
        distribution = audit_generated_distribution(official, formal_rows, expected_tasks)
    else:
        distribution = {
            "status": "DISTRIBUTION_AUDIT_NOT_RUN_INSUFFICIENT_BANK",
            "reason": bank["status"],
        }
    recovery_status = terminal_status(
        bank_status=bank["status"],
        distribution_status=distribution["status"],
        capacity_probe_passed=capacity_passed,
        zero_historical_overlap=zero_historical_overlap,
        all_formal_reproducible=formal_reproducible,
    )
    phase_a_passed = phase_a.get("status") == "TASK14R_PHASE_A_ENGINEERING_GATES_PASSED"
    status = recovery_status if phase_a_passed else "TASK14R_PHASE_A_ENGINEERING_GATES_FAILED"

    args.output_root.mkdir(parents=True, exist_ok=False)
    phase_a_summary = {
        "schema_version": "task14r.phase_a.summary.v1",
        "task_name": TASK14R_NAME,
        "status": phase_a["status"],
        "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "source": {"path": portable_path(args.phase_a_run), "sha256": file_sha256(args.phase_a_run)},
        "case_count": phase_a["case_count"],
        "rollout_count": phase_a["rollout_count"],
        "passed_identity_case_count": phase_a["passed_identity_case_count"],
        "failed_identity_case_count": phase_a["failed_identity_case_count"],
        "cases": [
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "noop_closed_loop_identity": row["noop_vs_snap_repeat"]["closed_loop_identity"],
                "full_swap_closed_loop_identity": row["full_swap_vs_base_repeat"]["closed_loop_identity"],
                "diagnostics": row["diagnostics"],
            }
            for row in phase_a["case_audits"]
        ],
    }
    bank_manifest = {
        "schema_version": "task14r.state_bank.v1",
        "task_name": TASK14R_NAME,
        "status": bank["status"],
        "distribution_name": "simulator-generated evaluation distribution",
        "official_libero_test_states": False,
        "formal_case_count": bank["formal_case_count"],
        "reserve_case_count": bank["reserve_case_count"],
        "selection_rule": bank["selection_rule"],
        "adaptive_reserve_activation": False,
        "policy_query_count": 0,
        "outcomes_accessed": False,
        "formal_outcome_rollout_count": 0,
        "tasks": bank["tasks"],
        "formal_states": [compact_bank_row(row) for row in bank["formal_states"]],
        "reserve_states": [compact_bank_row(row) for row in bank["reserve_states"]],
        "shards": shard_records,
    }
    decision = {
        "schema_version": "task14r.final_decision.v1",
        "task_name": TASK14R_NAME,
        "status": status,
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty_before_finalize": False,
        "generated_unix_ns": time.time_ns(),
        "phase_a_engineering_gates_passed": phase_a_passed,
        "phase_a_evidence_labels": list(TASK14R_EVIDENCE_LABELS),
        "capacity_probe_passed": capacity_passed,
        "capacity_probe_tasks": probe_reports,
        "state_bank_status": bank["status"],
        "formal_case_count": bank["formal_case_count"],
        "global_generated_payload_uniqueness": globally_unique,
        "zero_historical_overlap": zero_historical_overlap,
        "all_formal_states_reproducible": formal_reproducible,
        "distribution_audit_status": distribution["status"],
        "formal_base_snap_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "next_task_if_ready": "TASK14C_PROSPECTIVE_POSITION_CAUSAL_AUDIT"
        if status == "READY_FOR_TASK14_CAUSAL_RELAUNCH"
        else None,
    }
    outputs = {
        "TASK14R_PHASE_A_SUMMARY.json": phase_a_summary,
        "TASK14R_STATE_BANK_MANIFEST.json": bank_manifest,
        "TASK14R_DISTRIBUTION_AUDIT.json": distribution,
        "TASK14R_FINAL_DECISION.json": decision,
    }
    for name, payload in outputs.items():
        (args.output_root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "formal_cases": bank["formal_case_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
