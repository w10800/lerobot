#!/usr/bin/env python
"""Freeze the outcome-conditioned Attempt002 engineering dry-run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from task14r_common import TASK14R_EVIDENCE_LABELS, TASK14R_NAME, TASK14R_PHASE_A_ARMS

EXPECTED_TASK14_COMMIT = "54fee829ba0b180c53a85b7a9f56b24fbe5ae768"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--dev-b2-manifest", type=Path, required=True)
    parser.add_argument("--attempt002-run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before freezing the Task14R Phase A manifest")
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_TASK14_COMMIT, "HEAD"], check=True)
    paired = load_json(args.paired_results)
    design = load_json(args.dev_b2_manifest)
    run = load_json(args.attempt002_run_manifest)
    if paired.get("status") != "TASK13_ATTEMPT002_PAIRED_RESULTS_COMPLETE":
        raise RuntimeError("Attempt002 paired results are not complete")
    if design.get("status") != "TASK13_ATTEMPT002_DEV_B2_FROZEN":
        raise RuntimeError("Attempt002 Dev-B2 design is not frozen")
    if run.get("status") != "DEVELOPMENT_COMPLETED" or int(run.get("rollout_count", 0)) != 400:
        raise RuntimeError("Attempt002 execution manifest is not complete")

    design_index = {str(row["case_id"]): row for row in design["cases"]}
    result_index = {
        (
            str(row["suite"]),
            int(row["task_id"]),
            int(row["init_state_id"]),
            int(row["env_seed"]),
            str(row["arm"]),
        ): row
        for row in run["results"]
    }
    harmful = sorted(
        (row for row in paired["pairs"] if row.get("outcome_label") == "harmful"),
        key=lambda row: str(row["case_id"]),
    )
    if len(harmful) != 15:
        raise RuntimeError(f"Task14R requires exactly 15 Attempt002 harmful cases, got {len(harmful)}")

    cases = []
    for pair in harmful:
        design_case_id = str(pair["design_case_id"])
        case = design_index[design_case_id]
        key_base = (
            str(case["suite"]),
            int(case["task_id"]),
            int(case["init_state_id"]),
            int(case["env_seed"]),
            "base10",
        )
        key_snap = (*key_base[:-1], "snap1")
        base = result_index[key_base]
        snap = result_index[key_snap]
        if not bool(base["success"]) or bool(snap["success"]):
            raise RuntimeError(f"Attempt002 harmful label drift: {pair['case_id']}")
        cases.append(
            {
                **case,
                "attempt002_case_id": pair["case_id"],
                "attempt002_outcome_label": "harmful",
                "attempt002_base_success": True,
                "attempt002_snap_success": False,
                "attempt002_base_action_stream_sha256": base["action_stream_sha256"],
                "attempt002_snap_action_stream_sha256": snap["action_stream_sha256"],
                "attempt002_base_trace_manifest": base["trace_manifest"],
                "attempt002_snap_trace_manifest": snap["trace_manifest"],
                "capsule_path": base["capsule_path"],
                "capsule_sha256": base["capsule_sha256"],
                "noise_schedule_sha256": base["noise_schedule_sha256"],
            }
        )

    payload = {
        "schema_version": "task14r.phase_a.engineering.v1",
        "task_name": TASK14R_NAME,
        "phase": "PHASE_A_ATTEMPT002_ENGINEERING_DRY_RUN",
        "status": "TASK14R_PHASE_A_ENGINEERING_MANIFEST_FROZEN",
        "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
        "outcome_conditioned": True,
        "primary_evidence": False,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "registered_arms": list(TASK14R_PHASE_A_ARMS),
        "case_count": len(cases),
        "planned_rollout_count": len(cases) * len(TASK14R_PHASE_A_ARMS),
        "source_sha256": {
            "paired_results": file_sha256(args.paired_results),
            "dev_b2_manifest": file_sha256(args.dev_b2_manifest),
            "attempt002_run_manifest": file_sha256(args.attempt002_run_manifest),
        },
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "generated_unix_ns": time.time_ns(),
        "online_query_rule": (
            "At every replan both Base-10 and Snap-1 query the current observation actually visited "
            "by that arm with the same frozen noise tensor; cached original actions are forbidden"
        ),
        "identity_gates": {
            "noop": "stepwise action/physical-state/queue/termination identity with snap_repeat",
            "full_swap": "stepwise action/physical-state/queue/termination identity with base_repeat",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
