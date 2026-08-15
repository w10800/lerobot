#!/usr/bin/env python
"""Freeze the single Task 12 student-state preservation probe.

This command only reads already-completed Task 11/12 artifacts. It cannot load
a policy, step an environment, or train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from task9_common import file_sha256, load_json

STATUS = "TASK12_STUDENT_STATE_PRESERVATION_PROBE_FROZEN"
SPLIT_SEED = 20260816
REQUIRED_POSITIVE_METRICS = (
    "mean_joint_state_l2_h1",
    "mean_joint_state_l2_h10",
    "mean_end_effector_position_l2_h1",
    "mean_end_effector_position_l2_h10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--task11-paired-results", type=Path, required=True)
    parser.add_argument("--phase-b-query-summary", type=Path, required=True)
    parser.add_argument("--phase-b-branch-summary", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def balanced_outcome_blind_split(
    shards: list[dict[str, Any]], diagnostic_case_ids: set[str], seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """Choose one held-out diagnostic case per shard using identifiers only."""
    train: list[str] = []
    heldout: list[str] = []
    seen: set[str] = set()
    for shard in sorted(shards, key=lambda row: row["shard_id"]):
        cases = sorted(set(shard["case_ids"]) & diagnostic_case_ids)
        if len(cases) != 5:
            raise ValueError(f"Expected five diagnostic cases in {shard['shard_id']}, got {len(cases)}")
        if seen & set(cases):
            raise ValueError("A diagnostic case appears in more than one shard")
        ranked = sorted(
            cases,
            key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(),
        )
        heldout.append(ranked[0])
        train.extend(ranked[1:])
        seen.update(cases)
    if seen != diagnostic_case_ids:
        raise ValueError("Execution shards do not exactly cover the diagnostic subset")
    return sorted(train), sorted(heldout)


def require_localized_mechanism(branch: dict[str, Any]) -> dict[str, Any]:
    if branch.get("status") != "TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE":
        raise ValueError("Task 12 Phase B isolated branches are incomplete")
    contrast = branch.get("harmful_minus_preserved", {})
    evidence: dict[str, Any] = {}
    for metric in REQUIRED_POSITIVE_METRICS:
        row = contrast.get(metric)
        if not row or not row.get("available") or float(row["ci95"][0]) <= 0:
            raise ValueError(f"Required localized mechanism evidence is absent: {metric}")
        evidence[metric] = row
    return evidence


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {args.output}")
    execution = load_json(args.execution_manifest)
    diagnostic = load_json(args.diagnostic_subset)
    paired = load_json(args.task11_paired_results)
    query = load_json(args.phase_b_query_summary)
    branch = load_json(args.phase_b_branch_summary)
    if paired.get("status") != "FORMAL_CONFIRMATION1200_PASS":
        raise ValueError("Task 11 did not pass the frozen confirmation gate")
    if query.get("status") != "TASK12_PHASE_B_SAME_STATE_QUERIES_COMPLETE":
        raise ValueError("Task 12 Phase B same-state queries are incomplete")
    mechanism = require_localized_mechanism(branch)
    diagnostic_ids = set(diagnostic["case_ids"])
    if len(diagnostic_ids) != 200:
        raise ValueError("Expected the frozen 200-case diagnostic subset")
    train_ids, heldout_ids = balanced_outcome_blind_split(execution["shards"], diagnostic_ids)
    if len(train_ids) != 160 or len(heldout_ids) != 40 or set(train_ids) & set(heldout_ids):
        raise ValueError("Unexpected Task 12 train/held-out split")
    checkpoint = args.selected_checkpoint.resolve()
    protocol = {
        "schema_version": 1,
        "status": STATUS,
        "decision": "D-019",
        "evidence_boundary": {
            "classification": "EXPLORATORY",
            "task11_primary_result_remains_frozen": True,
            "task11_cases_used_for_training_are_not_future_confirmation_support": True,
            "heldout_results_are_not_confirmatory": True,
            "libero_cf_training": False,
            "hyperparameter_sweep": False,
            "checkpoint_selection": False,
        },
        "input_sha256": {
            "execution_manifest": file_sha256(args.execution_manifest),
            "diagnostic_subset": file_sha256(args.diagnostic_subset),
            "task11_paired_results": file_sha256(args.task11_paired_results),
            "phase_b_query_summary": file_sha256(args.phase_b_query_summary),
            "phase_b_branch_summary": file_sha256(args.phase_b_branch_summary),
        },
        "selected_checkpoint": {
            "path": str(checkpoint),
            "model_sha256": file_sha256(checkpoint / "model.safetensors"),
            "expected_step": 20000,
        },
        "localized_mechanism_evidence": mechanism,
        "split": {
            "method": "one minimum SHA256(seed:case_id) per execution shard; no outcome fields",
            "seed": SPLIT_SEED,
            "train_case_count": len(train_ids),
            "heldout_case_count": len(heldout_ids),
            "train_case_ids": train_ids,
            "heldout_case_ids": heldout_ids,
        },
        "training": {
            "seed": SPLIT_SEED,
            "steps": 500,
            "optimizer": "AdamW",
            "learning_rate": 1e-6,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": "bf16",
            "execution_horizon": 10,
            "batch_composition": {
                "snap_visited_base10_target": 1,
                "base_visited_original_snap1_anchor": 1,
            },
            "loss": "MSE over normalized first-10 action prefix; equal preservation and anchor weights",
            "trainable_scope": "existing 20k trainable expert/state projection parameters",
            "saved_checkpoints": [500],
        },
        "preflight": {
            "same_state_self_replay_samples": 64,
            "maximum_raw_action_roundtrip_abs_error": 1e-5,
            "zero_learning_rate_step_must_preserve_trainable_tensors_exactly": True,
            "finite_loss_and_gradient_required": True,
        },
        "evaluation": {
            "offline_heldout_case_count": 40,
            "offline_use_all_archived_replans": True,
            "closed_loop_support": "frozen replay-v2 dev40 only",
            "closed_loop_arms": ["snap10", "snap2", "snap1"],
            "no_task11_reconfirmation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": STATUS, "train": len(train_ids), "heldout": len(heldout_ids)}))


if __name__ == "__main__":
    main()
