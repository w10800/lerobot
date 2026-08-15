#!/usr/bin/env python
"""Aggregate isolated branches from Snap-1 student-visited states."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from task9_common import file_sha256, load_json
from task12_common import bootstrap_median_difference, finite_summary, outcome_stratum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task11-paired-results", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def available_value(item: dict[str, Any]) -> float:
    return float(item["value"]) if item.get("available") else float("nan")


def object_position_mean(item: dict[str, Any]) -> float:
    if not item.get("available"):
        return float("nan")
    values = [float(value["position_l2"]) for value in item["per_object"].values() if value.get("available")]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    if load_json(args.protocol).get("status") != "TASK12_PASS_BRANCH_PROTOCOL_FROZEN":
        raise RuntimeError("Task 12 protocol is not frozen")
    execution = load_json(args.execution_manifest)
    outcomes = {
        item["case_id"]: item for item in load_json(args.task11_paired_results)["pairs"]
    }
    diagnostic_ids = set(load_json(args.diagnostic_subset)["case_ids"])
    rows = []
    statuses = []
    for shard in execution["shards"]:
        shard_root = root / "phase_b_branches" / shard["shard_id"]
        status = load_json(shard_root / "status.json")
        if status.get("status") != "TASK12_PHASE_B_BRANCH_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete Task 12 branch shard: {shard['shard_id']}")
        metrics_path = shard_root / "branch_transition_metrics.jsonl"
        if file_sha256(metrics_path) != status["metrics_sha256"]:
            raise RuntimeError(f"Task 12 branch metric hash mismatch: {shard['shard_id']}")
        statuses.append(status)
        with metrics_path.open() as stream:
            rows.extend(json.loads(line) for line in stream)
    if {row["case_id"] for row in rows} != diagnostic_ids:
        raise RuntimeError("Task 12 branch case coverage mismatch")
    if any(not row["identical_initial_state"] for row in rows):
        raise RuntimeError("Task 12 branch restoration mismatch")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    metrics = (
        "joint_state_l2",
        "end_effector_position_l2",
        "end_effector_orientation_radians",
        "gripper_l2",
        "object_position_l2",
    )
    case_rows = []
    for case_id in sorted(diagnostic_ids):
        selected = by_case[case_id]
        outcome = outcomes[case_id]
        case_row: dict[str, Any] = {
            "case_id": case_id,
            "task_id": selected[0]["task_id"],
            "outcome_stratum": outcome_stratum(
                outcome["base10_success"], outcome["snap1_20k_success"]
            ),
            "student_replans": len(selected),
        }
        for horizon in (1, 3, 5, 10):
            values: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                comparison = row["transition_divergence"]["base10_vs_snap1_20k"][str(horizon)]
                for metric in metrics[:-1]:
                    values[metric].append(available_value(comparison[metric]))
                values["object_position_l2"].append(object_position_mean(comparison["object_state"]))
                if comparison["combined_weighted_scalar"]["available"]:
                    raise RuntimeError("Weighted physical scalar appeared in Task 12 branch evidence")
            for metric in metrics:
                case_row[f"mean_{metric}_h{horizon}"] = float(np.nanmean(values[metric]))
        case_rows.append(case_row)
    with (root / "TASK12_PHASE_B_BRANCH_CASE_METRICS.jsonl").open("w") as stream:
        for row in case_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE",
        "case_count": len(case_rows),
        "student_replan_state_count": len(rows),
        "isolated_branch_count": sum(item["isolated_branch_count"] for item in statuses),
        "identical_restoration_count": sum(
            item["identical_restoration_count"] for item in statuses
        ),
        "strata": {},
        "harmful_minus_preserved": {},
        "combined_weighted_scalar_used": False,
        "new_training": False,
    }
    all_metric_names = [f"mean_{metric}_h{horizon}" for horizon in (1, 3, 5, 10) for metric in metrics]
    for stratum in sorted({row["outcome_stratum"] for row in case_rows}):
        selected = [row for row in case_rows if row["outcome_stratum"] == stratum]
        summary["strata"][stratum] = {
            "case_count": len(selected),
            "metrics": {
                metric: finite_summary(row[metric] for row in selected) for metric in all_metric_names
            },
        }
    harmful = [row for row in case_rows if row["outcome_stratum"] == "base_only_success"]
    preserved = [row for row in case_rows if row["outcome_stratum"] == "concordant_success"]
    for offset, metric in enumerate(all_metric_names):
        summary["harmful_minus_preserved"][metric] = bootstrap_median_difference(
            (row[metric] for row in harmful),
            (row[metric] for row in preserved),
            seed=20261101 + offset,
        )
    summary_path = root / "TASK12_PHASE_B_BRANCH_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "TASK12_PHASE_B_BRANCH_REPORT.md").write_text(
        "# CRP-VLA Task 12 Phase B Isolated Branches\n\n"
        "**Status: TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE**\n\n"
        f"- Diagnostic cases: `{len(case_rows)}`\n"
        f"- Snap-1 student-visited replan states: `{len(rows)}`\n"
        f"- Isolated Base/Snap branches: `{summary['isolated_branch_count']}`\n"
        f"- Identical restorations: `{summary['identical_restoration_count']}`\n"
        "- Horizons: `1 / 3 / 5 / 10`\n"
        "- Combined weighted scalar: `NO`\n"
        "- New training: `NO`\n\n"
        "Raw transition components are descriptive mechanism evidence, not a formal causal gate.\n\n"
        f"Summary SHA-256: `{file_sha256(summary_path)}`.\n"
    )
    print(json.dumps({"status": summary["status"], "states": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
