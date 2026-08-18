#!/usr/bin/env python
"""Analyze archived paired Task 11 trajectories without new policy queries."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from task9_common import file_sha256, load_json
from task12_common import (
    COUNTERFACTUAL_HORIZONS,
    bootstrap_median_difference,
    contact_set,
    finite_summary,
    jaccard_distance,
    outcome_stratum,
    progress_bin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task11-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def object_position_l2(left: dict[str, Any], right: dict[str, Any]) -> float:
    shared = sorted(set(left) & set(right))
    values = []
    for name in shared:
        if "pos" in left[name] and "pos" in right[name]:
            values.append(
                np.linalg.norm(
                    np.asarray(left[name]["pos"], dtype=np.float64)
                    - np.asarray(right[name]["pos"], dtype=np.float64)
                )
            )
    return float(np.mean(values)) if values else float("nan")


def load_trace(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    top_path = Path(record["trace_path"])
    if file_sha256(top_path) != record["trace_hash"]:
        raise RuntimeError(f"Task 11 trace hash mismatch: {top_path}")
    top = load_json(top_path)
    numeric_path = Path(top["numeric_actions"]["path"])
    rows_path = Path(top["trace_jsonl_gz"]["path"])
    if file_sha256(numeric_path) != top["numeric_actions"]["sha256"]:
        raise RuntimeError(f"Numeric action hash mismatch: {numeric_path}")
    if file_sha256(rows_path) != top["trace_jsonl_gz"]["sha256"]:
        raise RuntimeError(f"Trace row hash mismatch: {rows_path}")
    with np.load(numeric_path) as archive:
        numeric = {key: archive[key].copy() for key in archive.files}
    return top, numeric, read_rows(rows_path)


def pair_metrics(
    case_id: str,
    outcome: dict[str, Any],
    base_record: dict[str, Any],
    snap_record: dict[str, Any],
    diagnostic_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, base_numeric, base_rows = load_trace(base_record)
    _, snap_numeric, snap_rows = load_trace(snap_record)
    base_chunks = base_numeric["predicted_chunks"]
    snap_chunks = snap_numeric["predicted_chunks"]
    replan_count = min(len(base_chunks), len(snap_chunks))
    predicted = {}
    initial_predicted = {}
    for horizon in COUNTERFACTUAL_HORIZONS:
        values = np.linalg.norm(
            base_chunks[:replan_count, :horizon] - snap_chunks[:replan_count, :horizon], axis=(1, 2)
        )
        predicted[str(horizon)] = float(values.mean()) if len(values) else float("nan")
        initial_predicted[str(horizon)] = (
            float(np.linalg.norm(base_chunks[0, :horizon] - snap_chunks[0, :horizon]))
            if replan_count
            else float("nan")
        )
    aligned = min(len(base_rows), len(snap_rows))
    step_metrics: list[dict[str, Any]] = []
    for index in range(aligned):
        base = base_rows[index]
        snap = snap_rows[index]
        base_action = np.asarray(base["actually_executed_action"], dtype=np.float64)
        snap_action = np.asarray(snap["actually_executed_action"], dtype=np.float64)
        base_pos = np.asarray(base["end_effector_pose"]["pos"], dtype=np.float64)
        snap_pos = np.asarray(snap["end_effector_pose"]["pos"], dtype=np.float64)
        step_metrics.append(
            {
                "progress_bin": progress_bin(index, aligned),
                "action_l2": float(np.linalg.norm(base_action - snap_action)),
                "translation_l2": float(np.linalg.norm(base_action[:3] - snap_action[:3])),
                "rotation_l2": float(np.linalg.norm(base_action[3:6] - snap_action[3:6])),
                "gripper_abs": float(abs(base_action[6] - snap_action[6])),
                "eef_position_l2": float(np.linalg.norm(base_pos - snap_pos)),
                "object_position_l2": object_position_l2(base["object_poses"], snap["object_poses"]),
                "contact_jaccard": jaccard_distance(
                    contact_set(base["contact_pairs"]), contact_set(snap["contact_pairs"])
                ),
            }
        )
    row = {
        "case_id": case_id,
        "task_id": outcome["task_id"],
        "diagnostic_subset": case_id in diagnostic_ids,
        "base10_success": bool(outcome["base10_success"]),
        "snap1_20k_success": bool(outcome["snap1_20k_success"]),
        "outcome_stratum": outcome_stratum(outcome["base10_success"], outcome["snap1_20k_success"]),
        "base_steps": len(base_rows),
        "snap_steps": len(snap_rows),
        "aligned_steps": aligned,
        "episode_length_abs_difference": abs(len(base_rows) - len(snap_rows)),
        "initial_predicted_chunk_l2": initial_predicted,
        "mean_aligned_predicted_chunk_l2": predicted,
    }
    for key in (
        "action_l2",
        "translation_l2",
        "rotation_l2",
        "gripper_abs",
        "eef_position_l2",
        "object_position_l2",
        "contact_jaccard",
    ):
        values = [item[key] for item in step_metrics]
        row[f"mean_{key}"] = float(np.nanmean(values)) if values else float("nan")
        row[f"max_{key}"] = float(np.nanmax(values)) if values else float("nan")
    curve_rows = []
    for bin_index in range(10):
        selected = [item for item in step_metrics if item["progress_bin"] == bin_index]
        curve_rows.append(
            {
                "case_id": case_id,
                "outcome_stratum": row["outcome_stratum"],
                "progress_bin": bin_index,
                **{
                    f"mean_{key}": float(np.nanmean([item[key] for item in selected]))
                    if selected
                    else None
                    for key in (
                        "action_l2",
                        "translation_l2",
                        "rotation_l2",
                        "gripper_abs",
                        "eef_position_l2",
                        "object_position_l2",
                        "contact_jaccard",
                    )
                },
            }
        )
    return row, curve_rows


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_json(args.protocol)
    if protocol.get("status") != "TASK12_PASS_BRANCH_PROTOCOL_FROZEN":
        raise RuntimeError("Task 12 protocol is not frozen")
    task11 = args.task11_root.resolve()
    pairs = load_json(task11 / "CONFIRMATION1200_PAIRED_RESULTS.json")
    trace_manifest = load_json(task11 / "CONFIRMATION1200_TRACE_MANIFEST.json")
    diagnostic_ids = set(load_json(args.diagnostic_subset)["case_ids"])
    outcomes = {item["case_id"]: item for item in pairs["pairs"]}
    traces: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in trace_manifest["records"]:
        traces[item["case_id"]][item["arm"]] = item
    if set(traces) != set(outcomes) or any(set(arms) != {"base10", "snap1_20k"} for arms in traces.values()):
        raise RuntimeError("Task 12 trace coverage mismatch")
    case_rows = []
    curve_rows = []
    for index, case_id in enumerate(sorted(outcomes), start=1):
        row, curves = pair_metrics(
            case_id,
            outcomes[case_id],
            traces[case_id]["base10"],
            traces[case_id]["snap1_20k"],
            diagnostic_ids,
        )
        case_rows.append(row)
        curve_rows.extend(curves)
        if index % 50 == 0:
            print(f"phase_a_pairs={index}/1200", flush=True)
    with (root / "TASK12_PHASE_A_CASE_METRICS.jsonl").open("w") as stream:
        for row in case_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (root / "TASK12_PHASE_A_PROGRESS_CURVES.jsonl").open("w") as stream:
        for row in curve_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    metric_names = [
        "initial_predicted_chunk_l2.10",
        "mean_action_l2",
        "mean_translation_l2",
        "mean_rotation_l2",
        "mean_gripper_abs",
        "mean_eef_position_l2",
        "mean_object_position_l2",
        "mean_contact_jaccard",
        "episode_length_abs_difference",
    ]

    def value(row: dict[str, Any], metric: str) -> float:
        if metric.startswith("initial_predicted_chunk_l2."):
            return float(row["initial_predicted_chunk_l2"][metric.rsplit(".", 1)[1]])
        return float(row[metric])

    strata = sorted({row["outcome_stratum"] for row in case_rows})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "TASK12_PHASE_A_COMPLETE",
        "case_count": len(case_rows),
        "diagnostic_case_count": sum(row["diagnostic_subset"] for row in case_rows),
        "protocol_sha256": file_sha256(args.protocol),
        "new_policy_queries": 0,
        "new_rollouts": 0,
        "new_training": False,
        "strata": {},
        "primary_descriptive_contrast": {},
    }
    for stratum in strata:
        selected = [row for row in case_rows if row["outcome_stratum"] == stratum]
        summary["strata"][stratum] = {
            "case_count": len(selected),
            "metrics": {metric: finite_summary(value(row, metric) for row in selected) for metric in metric_names},
        }
    harmful = [row for row in case_rows if row["outcome_stratum"] == "base_only_success"]
    preserved = [row for row in case_rows if row["outcome_stratum"] == "concordant_success"]
    for offset, metric in enumerate(metric_names):
        summary["primary_descriptive_contrast"][metric] = bootstrap_median_difference(
            (value(row, metric) for row in harmful),
            (value(row, metric) for row in preserved),
            seed=20260817 + offset,
        )
    summary_path = root / "TASK12_PHASE_A_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# CRP-VLA Task 12 Phase A",
        "",
        "**Status: TASK12_PHASE_A_COMPLETE**",
        "",
        f"Analyzed `{len(case_rows)}` frozen Task 11 pairs; no new policy query, rollout, or training was performed.",
        "",
        "| Outcome stratum | Cases |",
        "|---|---:|",
    ]
    for stratum in strata:
        lines.append(f"| {stratum} | {summary['strata'][stratum]['case_count']} |")
    lines += [
        "",
        "The prespecified descriptive contrast is Base-only success minus concordant success. Raw components remain separate; these intervals are exploratory mechanism evidence, not a new formal gate.",
        "",
        f"Summary SHA-256: `{file_sha256(summary_path)}`.",
    ]
    (root / "TASK12_PHASE_A_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": summary["status"], "cases": len(case_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
