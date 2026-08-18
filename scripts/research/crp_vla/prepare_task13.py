#!/usr/bin/env python
"""Freeze Task 13 hypotheses, metrics, controls, and decision rules before Dev-B outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from task9_common import file_sha256, load_json
from task13_common import (
    DEV_B_CASES,
    DEV_B_FIRST_STATE,
    DEV_B_STATES_PER_TASK,
    SELECTED_MODEL_SHA256,
    TASK13_ARMS,
    TASK13_BOOTSTRAP_REPEATS,
    TASK13_BOOTSTRAP_SEED,
    TASK13_HORIZONS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task11-statistics", type=Path, required=True)
    parser.add_argument("--task12-query-summary", type=Path, required=True)
    parser.add_argument("--task12-branch-summary", type=Path, required=True)
    parser.add_argument("--task12-final", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    task11 = load_json(args.task11_statistics)
    query = load_json(args.task12_query_summary)
    branch = load_json(args.task12_branch_summary)
    if task11.get("status") != "FORMAL_CONFIRMATION1200_PASS":
        raise RuntimeError("Task 11 confirmed baseline is unavailable")
    if query.get("status") != "TASK12_PHASE_B_SAME_STATE_QUERIES_COMPLETE":
        raise RuntimeError("Task 12 query evidence is incomplete")
    if branch.get("status") != "TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE":
        raise RuntimeError("Task 12 branch evidence is incomplete")
    if "TASK12_EXPLORATORY_PROBE_NOT_PROMOTED" not in args.task12_final.read_text():
        raise RuntimeError("Task 12 final decision drift")

    metrics = {
        "schema_version": 1,
        "status": "TASK13_METRICS_FROZEN",
        "labels": {
            "harmful": "base10_success == true and snap1_success == false",
            "preserved": "base10_success == true and snap1_success == true",
            "secondary_only": ["student_only_success", "both_fail"],
        },
        "state_sampling": "all replan states from both primary arms; aggregate state metrics to case means before primary contrasts",
        "raw_action_primary": "denormalized Base-10 versus Snap-1 first-10-action prefix L2",
        "transition_primary_families": [
            "joint_position_l2",
            "end_effector_position_l2",
            "end_effector_orientation_distance",
        ],
        "transition_secondary": ["object_position_l2", "gripper_state_l2"],
        "horizons": list(TASK13_HORIZONS),
        "predictor_a": "case-mean raw_action_primary",
        "predictor_b": "case-mean end_effector_position_l2 at h=5",
        "cross_validation": "leave-one-task-out",
        "prediction_metrics": ["AUROC", "AUPRC", "Brier"],
        "bootstrap": {
            "case_repeats": TASK13_BOOTSTRAP_REPEATS,
            "task_cluster_repeats": TASK13_BOOTSTRAP_REPEATS,
            "seed": TASK13_BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "negative_controls": {
            "base_visited": "same metrics and sampling on Base-10 visited states",
            "random_student_state": "one state per case selected by minimum SHA256(case_id:replan_index:seed)",
            "raw_student_action_norm": "denormalized Snap-1 first-10 prefix L2 norm",
            "horizon_zero": "all restored physical divergence components must equal exactly zero",
        },
        "combined_weighted_score": False,
        "metric_modification_after_outcomes": False,
    }
    (root / "TASK13_FROZEN_METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n")
    decision = """# CRP-VLA Task 13 Decision Rule

**Status: TASK13_DECISION_RULE_FROZEN**

Infrastructure or provenance failure yields `TASK13_INVALID` and no scientific conclusion.

H1 is supported when the Snap-visited harmful-minus-preserved raw-action case-bootstrap interval is positive and its point effect exceeds the identically computed Base-visited point effect. H2 is supported when at least two of joint, EEF-position, and EEF-orientation families have positive case-bootstrap lower bounds at three or more of four frozen horizons, positive task-cluster lower bounds at two or more horizons, and leave-one-task-out signs are not controlled by one task.

H3 uses only EEF-position divergence at h=5 versus raw action distance. “More informative” requires positive task-bootstrap lower bounds for both leave-one-task-out AUROC and AUPRC differences; Brier is reported without changing this gate.

- H1 + H2, with stable Snap-over-Base specificity: `CLOSED_LOOP_IMPACT_REPLICATED`.
- H2 but H3 not supported: `CLOSED_LOOP_AMPLIFICATION_REPLICATED_BUT_NO_PREDICTIVE_ADVANTAGE`.
- Some preregistered family/horizon effects replicate but H2 is not met: `PARTIAL_MECHANISM_REPLICATION`.
- No preregistered mechanism family replicates: `MECHANISM_NOT_REPLICATED`.

The optional tag `IMPACT_MORE_INFORMATIVE_THAN_RAW_ACTION_ERROR` is emitted only if H3 passes. No result authorizes training in Task 13.
"""
    (root / "TASK13_DECISION_RULE.md").write_text(decision)
    prereg = f"""# CRP-VLA Task 13 Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Verification Status: PREREGISTERED / OUTCOMES UNSEEN
- Frozen unix ns: `{time.time_ns()}`

**Status: TASK13_PREREGISTERED**

Task 13 prospectively tests whether Task 12's student-visited same-state action signal and isolated robot-trajectory amplification replicate on a new disjoint Dev-B. It does not train, tune, or select a model.

## Frozen design

- Dev-B: {DEV_B_CASES} cases = 40 tasks x {DEV_B_STATES_PER_TASK} fixed states.
- Candidate state indices: {DEV_B_FIRST_STATE}–{DEV_B_FIRST_STATE + DEV_B_STATES_PER_TASK - 1} for every task.
- Arms: `{TASK13_ARMS[0]}` and `{TASK13_ARMS[1]}` only.
- Models: Base-10 teacher and original confirmed 20k Snap-1 student (`{SELECTED_MODEL_SHA256}`).
- Same state, condition, evaluator, normalization, execution horizon, termination, and matched noise.
- All replan states are retained and queried; isolated branches use h=1/3/5/10.

## Frozen hypotheses

- H1: harmful-versus-preserved action discrepancy is stronger on Snap-visited than Base-visited states.
- H2: harmful Snap-visited states show larger future joint/EEF position/orientation divergence.
- H3: frozen EEF-position h=5 transition impact outperforms raw action distance for harmful-state discrimination.

All labels, metrics, controls, bootstrap procedures, prediction protocol, and decisions are frozen in the companion files before any Dev-B outcome is observed. Task 11 is locked and the Task 12 probe is excluded from primary models.

Repository commit at protocol generation: `{subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}`.
"""
    (root / "TASK13_PREREGISTRATION.md").write_text(prereg)
    inputs = {
        "task11_statistics": file_sha256(args.task11_statistics),
        "task12_query_summary": file_sha256(args.task12_query_summary),
        "task12_branch_summary": file_sha256(args.task12_branch_summary),
        "task12_final": file_sha256(args.task12_final),
    }
    (root / "TASK13_PREREGISTRATION_INPUTS.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(json.dumps({"status": "TASK13_PREREGISTERED", "outcomes_accessed": False}))


if __name__ == "__main__":
    main()
