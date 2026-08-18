#!/usr/bin/env python
"""Aggregate frozen Task 12 same-state query shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from task9_common import file_sha256, load_json
from task12_common import bootstrap_median_difference, finite_summary, outcome_stratum, progress_bin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task11-paired-results", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    protocol = load_json(args.protocol)
    if protocol.get("status") != "TASK12_PASS_BRANCH_PROTOCOL_FROZEN":
        raise RuntimeError("Task 12 protocol is not frozen")
    execution = load_json(args.execution_manifest)
    paired = load_json(args.task11_paired_results)
    outcomes = {item["case_id"]: item for item in paired["pairs"]}
    diagnostic_ids = set(load_json(args.diagnostic_subset)["case_ids"])
    statuses = []
    rows = []
    for shard in execution["shards"]:
        shard_root = root / "phase_b_queries" / shard["shard_id"]
        status = load_json(shard_root / "status.json")
        if status.get("status") != "TASK12_PHASE_B_QUERY_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete Task 12 query shard: {shard['shard_id']}")
        if file_sha256(shard_root / "same_state_metrics.jsonl") != status["metrics_sha256"]:
            raise RuntimeError(f"Task 12 query metric hash mismatch: {shard['shard_id']}")
        if file_sha256(shard_root / "branch_actions.npz") != status["actions_sha256"]:
            raise RuntimeError(f"Task 12 query action hash mismatch: {shard['shard_id']}")
        statuses.append(status)
        with (shard_root / "same_state_metrics.jsonl").open() as stream:
            rows.extend(json.loads(line) for line in stream)
    if {row["case_id"] for row in rows} != diagnostic_ids:
        raise RuntimeError("Task 12 same-state query case coverage mismatch")
    if any(not row["self_replay_exact"] for row in rows):
        raise RuntimeError("Task 12 self replay is not exact")
    by_case_origin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_origin[(row["case_id"], row["origin_arm"])].append(row)
    case_rows = []
    for case_id in sorted(diagnostic_ids):
        outcome = outcomes[case_id]
        for origin in ("base10", "snap1_20k"):
            selected = sorted(by_case_origin[(case_id, origin)], key=lambda item: item["replan_index"])
            if not selected:
                raise RuntimeError(f"Missing Task 12 same-state replans: {case_id}/{origin}")
            case_rows.append(
                {
                    "case_id": case_id,
                    "task_id": selected[0]["task_id"],
                    "origin_arm": origin,
                    "outcome_stratum": outcome_stratum(
                        outcome["base10_success"], outcome["snap1_20k_success"]
                    ),
                    "replans": len(selected),
                    "initial_denorm_prefix_l2_h10": selected[0]["denorm_prefix_l2_h10"],
                    "mean_denorm_prefix_l2_h10": float(
                        np.mean([item["denorm_prefix_l2_h10"] for item in selected])
                    ),
                    "late_denorm_prefix_l2_h10": float(
                        np.mean(
                            [
                                item["denorm_prefix_l2_h10"]
                                for index, item in enumerate(selected)
                                if progress_bin(index, len(selected)) >= 5
                            ]
                        )
                    ),
                    "mean_translation_l2_h10": float(
                        np.mean([item["denorm_translation_l2_h10"] for item in selected])
                    ),
                    "mean_rotation_l2_h10": float(
                        np.mean([item["denorm_rotation_l2_h10"] for item in selected])
                    ),
                    "mean_gripper_l2_h10": float(
                        np.mean([item["denorm_gripper_l2_h10"] for item in selected])
                    ),
                }
            )
    with (root / "TASK12_PHASE_B_QUERY_CASE_METRICS.jsonl").open("w") as stream:
        for row in case_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    metrics = (
        "initial_denorm_prefix_l2_h10",
        "mean_denorm_prefix_l2_h10",
        "late_denorm_prefix_l2_h10",
        "mean_translation_l2_h10",
        "mean_rotation_l2_h10",
        "mean_gripper_l2_h10",
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "TASK12_PHASE_B_SAME_STATE_QUERIES_COMPLETE",
        "case_count": len(diagnostic_ids),
        "replan_state_count": len(rows),
        "policy_query_count": sum(item["policy_query_count"] for item in statuses),
        "self_replay_exact_count": len(rows),
        "origins": {},
        "new_rollout": False,
        "new_training": False,
    }
    for origin in ("snap1_20k", "base10"):
        origin_rows = [row for row in case_rows if row["origin_arm"] == origin]
        summary["origins"][origin] = {"strata": {}, "harmful_minus_preserved": {}}
        for stratum in sorted({row["outcome_stratum"] for row in origin_rows}):
            selected = [row for row in origin_rows if row["outcome_stratum"] == stratum]
            summary["origins"][origin]["strata"][stratum] = {
                "case_count": len(selected),
                "metrics": {
                    metric: finite_summary(row[metric] for row in selected) for metric in metrics
                },
            }
        harmful = [row for row in origin_rows if row["outcome_stratum"] == "base_only_success"]
        preserved = [row for row in origin_rows if row["outcome_stratum"] == "concordant_success"]
        for offset, metric in enumerate(metrics):
            summary["origins"][origin]["harmful_minus_preserved"][metric] = (
                bootstrap_median_difference(
                    (row[metric] for row in harmful),
                    (row[metric] for row in preserved),
                    seed=20260901 + offset + (0 if origin == "snap1_20k" else 100),
                )
            )
    summary_path = root / "TASK12_PHASE_B_QUERY_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "TASK12_PHASE_B_QUERY_REPORT.md").write_text(
        "# CRP-VLA Task 12 Phase B Same-State Queries\n\n"
        "**Status: TASK12_PHASE_B_SAME_STATE_QUERIES_COMPLETE**\n\n"
        f"- Diagnostic cases: `{len(diagnostic_ids)}`\n"
        f"- Archived replan states: `{len(rows)}`\n"
        f"- Policy queries: `{summary['policy_query_count']}`\n"
        f"- Exact self replays: `{len(rows)}/{len(rows)}`\n"
        "- New rollout/training: `NO / NO`\n\n"
        "The primary origin is the Snap-1 student-visited state distribution. Harmful-minus-preserved intervals are descriptive and do not form a new formal gate.\n\n"
        f"Summary SHA-256: `{file_sha256(summary_path)}`.\n"
    )
    print(json.dumps({"status": summary["status"], "states": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
