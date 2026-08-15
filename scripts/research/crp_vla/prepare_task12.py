#!/usr/bin/env python
"""Freeze the Task 12 PASS-branch mechanism protocol before trace inspection."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from task9_common import file_sha256, load_json
from task12_common import (
    COUNTERFACTUAL_HORIZONS,
    TASK12_BOOTSTRAP_REPEATS,
    TASK12_BOOTSTRAP_SEED,
    outcome_stratum,
    validate_task11_pass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task11-root", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task11 = args.task11_root.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stats_path = task11 / "CONFIRMATION1200_PRIMARY_STATISTICS.json"
    pairs_path = task11 / "CONFIRMATION1200_PAIRED_RESULTS.json"
    trace_path = task11 / "CONFIRMATION1200_TRACE_MANIFEST.json"
    audit_path = task11 / "CONFIRMATION1200_COMPLETION_AUDIT.md"
    readiness_path = task11 / "MECHANISM_DATA_READINESS.md"
    stats = load_json(stats_path)
    pairs = load_json(pairs_path)
    diagnostic = load_json(args.diagnostic_subset)
    validate_task11_pass(stats)
    if "TASK11_COMPLETION_AUDIT_PASSED" not in audit_path.read_text():
        raise RuntimeError("Task 11 completion audit is not passed")
    if "MECHANISM_DATA_READY" not in readiness_path.read_text():
        raise RuntimeError("Task 11 mechanism data is not ready")
    if pairs.get("case_count") != 1200 or len(pairs.get("pairs", [])) != 1200:
        raise RuntimeError("Task 11 paired result cardinality mismatch")
    diagnostic_ids = set(diagnostic["case_ids"])
    if len(diagnostic_ids) != 200:
        raise RuntimeError("Task 12 requires the frozen outcome-blind 200-case subset")
    strata = Counter(
        outcome_stratum(item["base10_success"], item["snap1_20k_success"])
        for item in pairs["pairs"]
        if item["case_id"] in diagnostic_ids
    )
    protocol = {
        "schema_version": 1,
        "status": "TASK12_PASS_BRANCH_PROTOCOL_FROZEN",
        "frozen_unix_ns": time.time_ns(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task11_status": stats["status"],
        "task11_primary_statistics_sha256": file_sha256(stats_path),
        "task11_paired_results_sha256": file_sha256(pairs_path),
        "task11_trace_manifest_sha256": file_sha256(trace_path),
        "diagnostic_subset_sha256": file_sha256(args.diagnostic_subset),
        "phase_a": {
            "population": "all 1200 frozen Task 11 paired cases",
            "data": "archived factual Base-10 and Snap-1 trajectories only",
            "primary_contrast": "base_only_success minus concordant_success",
            "metrics": [
                "initial_same-state_predicted-action prefix L2 at h=1,3,5,10",
                "time-aligned executed-action translation/rotation/gripper divergence",
                "time-aligned end-effector position divergence",
                "time-aligned task-object position divergence",
                "time-aligned contact-set Jaccard distance",
                "episode-length absolute difference",
            ],
            "combined_weighted_physical_scalar": False,
        },
        "phase_b": {
            "population": "frozen outcome-blind 200-case diagnostic subset",
            "case_count": 200,
            "diagnostic_outcome_strata_disclosed_after_task11": dict(sorted(strata.items())),
            "primary_origin": "Snap-1 student-visited replan states",
            "secondary_origin": "Base-10 visited replan states",
            "queries": "Base-10 and Snap-1 on the identical archived state/input/noise",
            "isolated_branch_horizons": list(COUNTERFACTUAL_HORIZONS),
            "self_replay_required_before_cross_query": True,
        },
        "descriptive_bootstrap": {
            "repeats": TASK12_BOOTSTRAP_REPEATS,
            "seed": TASK12_BOOTSTRAP_SEED,
            "confirmatory_gate": False,
        },
        "training_boundary": {
            "phase_a_can_authorize_training": False,
            "phase_b_required": True,
            "allowed_next_step": "one scoped student-state preservation probe if same-state/branch evidence localizes a repairable mechanism",
            "hyperparameter_sweep": False,
            "formal_claim_from_training": False,
        },
        "trace_features_accessed_before_freeze": False,
        "new_model_trained_before_freeze": False,
    }
    protocol_path = root / "TASK12_MECHANISM_PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    report = [
        "# CRP-VLA Task 12 Mechanism Protocol",
        "",
        "**Status: TASK12_PASS_BRANCH_PROTOCOL_FROZEN**",
        "",
        "Task 11 passed the frozen ordinary-LIBERO non-inferiority gate. Task 12 therefore asks why local closed-loop behavior can differ even when aggregate success is preserved.",
        "",
        "Phase A uses all 1,200 archived pairs for outcome-stratified, time-aligned trajectory description. Phase B uses the already frozen outcome-blind 200-case subset for same-state policy queries and isolated 1/3/5/10-step branches from student-visited states.",
        "",
        "No weighted physical scalar will be introduced. Raw action, end-effector, object, gripper, contact, and duration components remain separate. Phase A cannot authorize training by itself; any training probe requires Phase B localization and is exploratory only.",
        "",
        f"Protocol manifest SHA-256: `{file_sha256(protocol_path)}`.",
    ]
    (root / "TASK12_MECHANISM_PROTOCOL.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": protocol["status"], "diagnostic_strata": dict(strata)}, sort_keys=True))


if __name__ == "__main__":
    main()
