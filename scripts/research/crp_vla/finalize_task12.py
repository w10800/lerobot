#!/usr/bin/env python
"""Finalize Task 12 mechanism evidence and the single exploratory probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from task9_common import file_sha256, load_json, paired_flip_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-summary", type=Path, required=True)
    parser.add_argument("--phase-b-query-summary", type=Path, required=True)
    parser.add_argument("--phase-b-branch-summary", type=Path, required=True)
    parser.add_argument("--probe-protocol", type=Path, required=True)
    parser.add_argument("--probe-training-result", type=Path, required=True)
    parser.add_argument("--probe-replay", type=Path, required=True)
    parser.add_argument("--original-20k-replay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def result_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        row["suite"],
        int(row["task_id"]),
        int(row["init_state_id"]),
        int(row["env_seed"]),
        row["arm"],
    )


def compare_replays(probe: dict[str, Any], original: dict[str, Any], repeats: int = 10_000) -> dict[str, Any]:
    if probe.get("status") != "MATURATION_REPLAY_COMPLETED" or len(probe.get("results", [])) != 120:
        raise ValueError("Probe dev40 replay is incomplete")
    if original.get("status") != "MATURATION_REPLAY_COMPLETED" or len(original.get("results", [])) != 120:
        raise ValueError("Original 20k dev40 replay is incomplete")
    probe_by_key = {result_key(row): row for row in probe["results"]}
    original_by_key = {result_key(row): row for row in original["results"]}
    if len(probe_by_key) != 120 or set(probe_by_key) != set(original_by_key):
        raise ValueError("Probe/original dev40 identities do not align exactly")
    comparisons: dict[str, Any] = {}
    for arm_index, arm in enumerate(("snap10", "snap2", "snap1")):
        keys = sorted(key for key in probe_by_key if key[-1] == arm)
        probe_rows = [probe_by_key[key] for key in keys]
        original_rows = [original_by_key[key] for key in keys]
        for new, old in zip(probe_rows, original_rows, strict=True):
            invariant_fields = (
                "nfe",
                "initial_sim_state_sha256",
                "canonical_input_sha256",
                "noise_schedule_sha256",
                "evaluator_sha256",
            )
            if any(new[field] != old[field] for field in invariant_fields):
                raise ValueError(f"Probe/original invariant mismatch: {result_key(new)}")
        new_success = np.asarray([row["success"] for row in probe_rows], dtype=np.int8)
        old_success = np.asarray([row["success"] for row in original_rows], dtype=np.int8)
        task_ids = [f"{row['suite']}:{row['task_id']}" for row in probe_rows]
        paired = paired_flip_summary(
            new_success,
            old_success,
            task_ids,
            repeats=repeats,
            seed=20261201 + arm_index,
        )
        common_success_steps = [
            int(new["steps_run"]) - int(old["steps_run"])
            for new, old in zip(probe_rows, original_rows, strict=True)
            if new["success"] and old["success"]
        ]
        comparisons[arm] = {
            "probe_successes": int(new_success.sum()),
            "original_20k_successes": int(old_success.sum()),
            "probe_success_rate": float(new_success.mean()),
            "original_20k_success_rate": float(old_success.mean()),
            "probe_minus_original": paired,
            "common_success_count": len(common_success_steps),
            "common_success_step_difference_mean": (
                float(np.mean(common_success_steps)) if common_success_steps else None
            ),
            "common_success_step_difference_median": (
                float(np.median(common_success_steps)) if common_success_steps else None
            ),
        }
    return comparisons


def verify_probe_traces(probe: dict[str, Any]) -> dict[str, Any]:
    paths: list[str] = []
    hashes: list[str] = []
    for row in probe["results"]:
        trace = row.get("trace_manifest")
        if not isinstance(trace, dict) or set(trace) != {"path", "sha256"}:
            raise ValueError("Probe trace provenance is incomplete")
        path = Path(trace["path"])
        observed = file_sha256(path)
        if observed != trace["sha256"]:
            raise ValueError(f"Probe trace hash mismatch: {path}")
        paths.append(str(path.resolve()))
        hashes.append(observed)
    if len(set(paths)) != 120 or len(set(hashes)) != 120:
        raise ValueError("Probe trace paths or hashes are not unique")
    return {
        "trace_manifest_count": len(paths),
        "unique_trace_path_count": len(set(paths)),
        "unique_trace_hash_count": len(set(hashes)),
        "recomputed_trace_hash_count": len(hashes),
    }


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    comparison_path = root / "TASK12_PROBE_COMPARISON.json"
    report_path = root / "TASK12_FINAL_REPORT.md"
    checksum_path = root / "TASK12_FINAL_CHECKSUMS.txt"
    if any(path.exists() for path in (comparison_path, report_path, checksum_path)):
        raise FileExistsError("Refusing to overwrite a Task 12 final artifact")
    phase_a = load_json(args.phase_a_summary)
    query = load_json(args.phase_b_query_summary)
    branch = load_json(args.phase_b_branch_summary)
    protocol = load_json(args.probe_protocol)
    training = load_json(args.probe_training_result)
    probe = load_json(args.probe_replay)
    original = load_json(args.original_20k_replay)
    required_statuses = {
        "phase_a": (phase_a.get("status"), "TASK12_PHASE_A_COMPLETE"),
        "query": (query.get("status"), "TASK12_PHASE_B_SAME_STATE_QUERIES_COMPLETE"),
        "branch": (branch.get("status"), "TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE"),
        "protocol": (
            protocol.get("status"),
            "TASK12_STUDENT_STATE_PRESERVATION_PROBE_FROZEN",
        ),
        "training": (
            training.get("status"),
            "TASK12_STUDENT_STATE_PRESERVATION_PROBE_COMPLETE",
        ),
    }
    mismatched = {name: values for name, values in required_statuses.items() if values[0] != values[1]}
    if mismatched:
        raise ValueError(f"Task 12 prerequisite status mismatch: {mismatched}")
    if training["protocol_sha256"] != file_sha256(args.probe_protocol):
        raise ValueError("Task 12 training/protocol hash mismatch")
    if probe["checkpoint_sha256"] != training["trained_model_sha256"]:
        raise ValueError("Task 12 replay/trained checkpoint hash mismatch")
    trace_audit = verify_probe_traces(probe)
    comparisons = compare_replays(probe, original)
    snap1 = comparisons["snap1"]
    promoted = snap1["probe_successes"] > snap1["original_20k_successes"]
    decision = "TASK12_EXPLORATORY_PROBE_PROMOTED" if promoted else "TASK12_EXPLORATORY_PROBE_NOT_PROMOTED"
    summary = {
        "schema_version": 1,
        "status": "TASK12_COMPLETE",
        "decision": decision,
        "classification": "EXPLORATORY",
        "input_sha256": {
            "phase_a_summary": file_sha256(args.phase_a_summary),
            "phase_b_query_summary": file_sha256(args.phase_b_query_summary),
            "phase_b_branch_summary": file_sha256(args.phase_b_branch_summary),
            "probe_protocol": file_sha256(args.probe_protocol),
            "probe_training_result": file_sha256(args.probe_training_result),
            "probe_replay": file_sha256(args.probe_replay),
            "original_20k_replay": file_sha256(args.original_20k_replay),
        },
        "trained_model_sha256": training["trained_model_sha256"],
        "offline_heldout": {
            "before": training["heldout_before"],
            "after": training["heldout_after"],
        },
        "dev40": comparisons,
        "trace_audit": trace_audit,
        "training_attempts_with_admitted_updates": 1,
        "hyperparameter_sweep": False,
        "task11_reconfirmation": False,
        "libero_cf_training": False,
        "next_step": (
            "Stop this probe line. Retain the confirmed 20k Snap-1 checkpoint and design a future "
            "disjoint closed-loop preservation study before any further training."
        ),
    }
    comparison_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    before = training["heldout_before"]["preservation"]["target_mse_case_mean"]
    after = training["heldout_after"]["preservation"]["target_mse_case_mean"]
    improvement = (before - after) / before
    report_path.write_text(
        "# CRP-VLA Task 12 Final Report\n\n"
        f"**Status: TASK12_COMPLETE / {decision}**\n\n"
        "## Verified mechanism evidence\n\n"
        f"- Phase A pairs: `{phase_a['case_count']}`.\n"
        f"- Same-state policy queries: `{query['policy_query_count']}`; exact self-replays: "
        f"`{query['self_replay_exact_count']}`.\n"
        f"- Snap-visited isolated states: `{branch['student_replan_state_count']}`; branches: "
        f"`{branch['isolated_branch_count']}`; identical restorations: "
        f"`{branch['identical_restoration_count']}`.\n"
        "- Harmful cases show positive joint/end-effector physical divergence contrasts at h1/h3/h5/h10; "
        "object-position contrasts cross zero.\n\n"
        "## Exploratory fixed probe\n\n"
        f"- Held-out Snap-visited Base-target MSE: `{before:.8f}` -> `{after:.8f}` "
        f"(`{improvement:.1%}` relative reduction).\n"
        f"- Held-out Base-visited original-Snap anchor drift MSE: "
        f"`{training['heldout_after']['anchor']['target_mse_case_mean']:.8f}`.\n"
        f"- Trained model SHA-256: `{training['trained_model_sha256']}`.\n\n"
        "## Frozen replay-v2 dev40\n\n"
        f"- Snap-10: `{comparisons['snap10']['original_20k_successes']}/40` -> "
        f"`{comparisons['snap10']['probe_successes']}/40`.\n"
        f"- Snap-2: `{comparisons['snap2']['original_20k_successes']}/40` -> "
        f"`{comparisons['snap2']['probe_successes']}/40`.\n"
        f"- Snap-1: `{comparisons['snap1']['original_20k_successes']}/40` -> "
        f"`{comparisons['snap1']['probe_successes']}/40`; paired delta "
        f"`{comparisons['snap1']['probe_minus_original']['paired_success_difference']:+.3f}`.\n"
        f"- Probe trace manifests recomputed: `{trace_audit['recomputed_trace_hash_count']}/120`, "
        "all paths and content hashes unique.\n\n"
        "## Decision\n\n"
        "The fixed preservation loss improves its held-out offline target but does not improve the one-step "
        "closed-loop endpoint on dev40. Do not promote this checkpoint and do not tune on dev40. Retain the "
        "formally confirmed 20k Snap-1 model as the project baseline. Any next training study requires a new, "
        "disjoint protocol.\n\n"
        "This Task 12 probe is exploratory and does not modify the Task 11 confirmation result.\n"
    )
    checksum_targets = (
        args.phase_a_summary,
        args.phase_b_query_summary,
        args.phase_b_branch_summary,
        args.probe_protocol,
        comparison_path,
        report_path,
    )
    checksum_path.write_text("".join(f"{file_sha256(path)}  {path.resolve()}\n" for path in checksum_targets))
    print(json.dumps({"status": summary["status"], "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
