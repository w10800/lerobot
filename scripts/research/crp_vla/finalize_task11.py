#!/usr/bin/env python
"""Completion audit, frozen aggregation, and final Task 11 reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from task9_common import file_sha256, load_json
from task11_common import (
    BASE_ARM,
    SNAP_ARM,
    TASK10_MANIFEST_SHA256,
    completion_audit,
    frozen_primary_statistics,
    task_breakdown,
    validate_execution_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_capsule_record(record: dict[str, Any]) -> int:
    root = Path(record["path"])
    manifest = root / "sha256_manifest.txt"
    if not manifest.is_file() or file_sha256(manifest) != record["sha256"]:
        raise RuntimeError(f"Replan capsule manifest hash mismatch: {root}")
    members = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", maxsplit=1)
        member = root / name
        if not member.is_file() or file_sha256(member) != expected:
            raise RuntimeError(f"Replan capsule member hash mismatch: {member}")
        members += 1
    return members


def verify_trace_tree(record: dict[str, Any]) -> dict[str, int]:
    manifest_path = Path(record["trace_path"])
    if file_sha256(manifest_path) != record["trace_hash"]:
        raise RuntimeError(f"Top trace manifest hash mismatch: {manifest_path}")
    manifest = load_json(manifest_path)
    for key in ("trace_jsonl_gz", "numeric_actions"):
        member = Path(manifest[key]["path"])
        if not member.is_file() or file_sha256(member) != manifest[key]["sha256"]:
            raise RuntimeError(f"Trace member hash mismatch: {member}")
    capsule_members = 0
    for replan in manifest.get("replan_records", []):
        capsule_members += verify_capsule_record(replan)
    return {"replans": len(manifest.get("replan_records", [])), "capsule_members": capsule_members}


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Cases | Base-10 | Snap-1 20k | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['cases']} | {row['base10_successes']} | "
            f"{row['snap1_20k_successes']} | {row['paired_difference']:+.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    execution = load_json(args.execution_manifest)
    manifest = load_json(args.confirmation1200_manifest)
    diagnostic = load_json(args.diagnostic_subset)
    validate_execution_manifest(execution, manifest)
    if file_sha256(args.confirmation1200_manifest) != TASK10_MANIFEST_SHA256:
        raise RuntimeError("Task 10 manifest changed before finalization")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != execution["repository_commit"]
    ):
        raise RuntimeError("Repository HEAD drift before Task 11 finalization")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Repository must remain clean before Task 11 finalization")

    statuses = []
    for shard in execution["shards"]:
        path = root / "execution" / "shard_status" / f"{shard['shard_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"FORMAL_CONFIRMATION_INVALID: missing shard status {shard['shard_id']}")
        status = load_json(path)
        if (
            status.get("status") != "TASK11_SHARD_COMPLETED"
            or status.get("planned_results") != 60
            or status.get("completed_or_integrity_validated_results") != 60
            or status.get("shard_manifest_sha256") != shard["shard_manifest_sha256"]
        ):
            raise RuntimeError(f"FORMAL_CONFIRMATION_INVALID: invalid shard status {shard['shard_id']}")
        statuses.append(status)

    result_paths = sorted((root / "execution" / "results").glob("*/*/attempt-*/result.json"))
    records = [load_json(path) for path in result_paths]
    audit = completion_audit(
        records,
        manifest,
        base_model_hash=execution["base_model_sha256"],
        diagnostic_case_ids=set(diagnostic["case_ids"]),
        verify_files=True,
    )
    trace_replans = 0
    trace_capsule_members = 0
    for index, record in enumerate(records, start=1):
        checked = verify_trace_tree(record)
        trace_replans += checked["replans"]
        trace_capsule_members += checked["capsule_members"]
        if index % 100 == 0:
            print(f"trace_integrity={index}/2400", flush=True)
    audit["trace_manifests_verified"] = len(records)
    audit["replan_archives_verified"] = trace_replans
    audit["replan_capsule_members_verified"] = trace_capsule_members
    audit["shards_completed"] = len(statuses)

    raw_path = root / "CONFIRMATION1200_RAW_RESULTS.jsonl"
    with raw_path.open("w") as stream:
        for record in sorted(records, key=lambda item: (item["case_id"], item["arm"])):
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[record["case_id"]][record["arm"]] = record
    paired = {
        "schema_version": 1,
        "status": "TASK11_EXACT_1200_PAIRS",
        "manifest_sha256": TASK10_MANIFEST_SHA256,
        "case_count": 1200,
        "pairs": [
            {
                "case_id": case_id,
                "task_id": by_case[case_id][BASE_ARM]["task_id"],
                "base10_result_hash": by_case[case_id][BASE_ARM]["result_hash"],
                "snap1_20k_result_hash": by_case[case_id][SNAP_ARM]["result_hash"],
                "base10_success": by_case[case_id][BASE_ARM]["success"],
                "snap1_20k_success": by_case[case_id][SNAP_ARM]["success"],
            }
            for case_id in sorted(by_case)
        ],
    }
    write_json(root / "CONFIRMATION1200_PAIRED_RESULTS.json", paired)

    audit_md = [
        "# Confirmation1200 Completion Audit",
        "",
        "**Status: TASK11_COMPLETION_AUDIT_PASSED**",
        "",
        "This audit completed before the frozen statistical aggregator was invoked.",
        "",
        f"- Base-10 results: `{audit['base_results']}/1200`",
        f"- 20k Snap-1 results: `{audit['snap_results']}/1200`",
        f"- Exact paired cases: `{audit['exact_pairs']}/1200`",
        "- Duplicate/missing case-arm records: `0 / 0`",
        "- formal100/dev40 contamination: `0 / 0`",
        "- Diagnostic-derived feature contamination: `0`",
        f"- Unique result/trace paths and hashes: `{audit['unique_evidence']}`",
        f"- Top trace manifests verified: `{audit['trace_manifests_verified']}`",
        f"- Replan archives verified: `{audit['replan_archives_verified']}`",
        f"- Replan capsule members verified: `{audit['replan_capsule_members_verified']}`",
        "- Model/evaluator/manifest drift: `NO`",
        "",
        "```text",
        "TASK11_COMPLETION_AUDIT_PASSED",
        "```",
    ]
    (root / "CONFIRMATION1200_COMPLETION_AUDIT.md").write_text("\n".join(audit_md) + "\n")

    # The formal aggregator is intentionally called only after the audit file exists.
    stats = frozen_primary_statistics(records, manifest)
    write_json(root / "CONFIRMATION1200_PRIMARY_STATISTICS.json", stats)
    breakdown = task_breakdown(records, manifest)
    (root / "CONFIRMATION1200_TASK_BREAKDOWN.md").write_text(
        "# Confirmation1200 Task Breakdown\n\n"
        "Secondary descriptive evidence; it does not modify the frozen primary gate.\n\n"
        + markdown_table(breakdown)
        + "\n"
    )
    c = stats["paired_contingency"]
    primary_report = [
        "# Confirmation1200 Primary Report",
        "",
        f"**Formal status: {stats['status']}**",
        "",
        "## Frozen paired result",
        "",
        f"- Base-10: `{stats['base10_success_count']}/1200` (`{stats['base10_success_rate']:.6f}`)",
        f"- 20k Snap-1: `{stats['snap1_20k_success_count']}/1200` (`{stats['snap1_20k_success_rate']:.6f}`)",
        f"- Paired difference Snap-1 minus Base-10: `{stats['paired_success_difference']:+.6f}`",
        f"- Frozen paired percentile-bootstrap 95% interval: `{stats['case_paired_percentile_bootstrap_ci95']}`",
        f"- Non-inferiority margin: `{stats['noninferiority_margin']}`",
        f"- Frozen non-inferiority decision: `{'PASS' if stats['noninferiority_pass'] else 'FAIL'}`",
        "",
        "| | Base success | Base failure |",
        "|---|---:|---:|",
        f"| Snap success | {c['n_11']} | {c['n_10']} |",
        f"| Snap failure | {c['n_01']} | {c['n_00']} |",
        "",
        f"Secondary exact McNemar two-sided p-value: `{stats['exact_mcnemar_two_sided_p']}`. "
        "This does not replace the non-inferiority gate.",
        "",
        "Mechanism remains unverified; no new model was trained.",
        "",
        "```text",
        stats["status"],
        "```",
    ]
    (root / "CONFIRMATION1200_PRIMARY_REPORT.md").write_text("\n".join(primary_report) + "\n")

    trace_manifest = {
        "schema_version": 1,
        "status": "CONFIRMATION1200_TRACES_ARCHIVED",
        "rollout_trace_count": len(records),
        "replan_archive_count": trace_replans,
        "records": [
            {
                "case_id": record["case_id"],
                "arm": record["arm"],
                "trace_path": record["trace_path"],
                "trace_hash": record["trace_hash"],
                "result_hash": record["result_hash"],
            }
            for record in sorted(records, key=lambda item: (item["case_id"], item["arm"]))
        ],
    }
    write_json(root / "CONFIRMATION1200_TRACE_MANIFEST.json", trace_manifest)
    diagnostic_ids = set(diagnostic["case_ids"])
    diagnostic_inventory = {
        "schema_version": 1,
        "status": "CONFIRMATION1200_DIAGNOSTIC_TRACES_ARCHIVED",
        "diagnostic_case_count": len(diagnostic_ids),
        "diagnostic_rollout_trace_count": sum(record["case_id"] in diagnostic_ids for record in records),
        "primary_aggregator_consumed_diagnostic_features": False,
        "records": [item for item in trace_manifest["records"] if item["case_id"] in diagnostic_ids],
    }
    write_json(root / "CONFIRMATION1200_DIAGNOSTIC_TRACE_INVENTORY.json", diagnostic_inventory)
    mechanism_ready = trace_replans > 0 and diagnostic_inventory["diagnostic_rollout_trace_count"] == 400
    (root / "MECHANISM_DATA_READINESS.md").write_text(
        "# Mechanism Data Readiness\n\n"
        f"**Status: {'MECHANISM_DATA_READY' if mechanism_ready else 'MECHANISM_DATA_NOT_READY'}**\n\n"
        f"- Full primary rollout traces: `{len(records)}`\n"
        f"- Full replan archives: `{trace_replans}`\n"
        f"- Frozen diagnostic-case arm traces: `{diagnostic_inventory['diagnostic_rollout_trace_count']}/400`\n"
        "- Student-visited observations and simulator states: archived per replan.\n"
        "- Mechanism analysis performed in Task 11: `NO`\n"
        "- Transition metric tuned in Task 11: `NO`\n\n"
        "This status permits a separately authorized future mechanism audit; it does not establish a mechanism.\n"
    )

    failures = sorted((root / "execution" / "results").glob("*/*/attempt-*/failure.json"))
    retry_results = [record for record in records if record["attempt_id"] != "attempt-0001"]
    retry_compliant = all(record.get("retry_lineage") for record in retry_results)
    final_answers = {
        "1_formal_arm_results_2400": len(records) == 2400,
        "2_exact_paired_cases_1200": len(by_case) == 1200,
        "3_base10_success_count_rate": [stats["base10_success_count"], stats["base10_success_rate"]],
        "4_snap1_20k_success_count_rate": [
            stats["snap1_20k_success_count"],
            stats["snap1_20k_success_rate"],
        ],
        "5_contingency_n11_n10_n01_n00": [c["n_11"], c["n_10"], c["n_01"], c["n_00"]],
        "6_paired_difference": stats["paired_success_difference"],
        "7_frozen_confidence_interval": stats["case_paired_percentile_bootstrap_ci95"],
        "8_frozen_noninferiority_gate_pass": stats["noninferiority_pass"],
        "9_model_hash_evaluator_manifest_drift": False,
        "10_formal100_dev40_contamination": [0, 0],
        "11_duplicate_missing_case": [0, 0],
        "12_infrastructure_failures_retries": [len(failures), len(retry_results)],
        "13_retries_frozen_protocol_compliant": retry_compliant,
        "14_diagnostic_subset_isolated": True,
        "15_trajectory_evidence_complete": True,
        "16_mechanism_analysis_data_ready": mechanism_ready,
        "17_new_model_trained": False,
        "18_checkpoint_modified": False,
        "19_frozen_statistical_protocol_modified": False,
    }
    final_report = [
        "# CRP-VLA Task 11 Final Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Verification Status: VERIFIED",
        "- Version Label: crp_vla_task11_final_v1",
        "",
        f"**Final status: {stats['status']}**",
        "",
        "## Required answers",
        "",
        "```json",
        json.dumps(final_answers, indent=2, sort_keys=True),
        "```",
        "",
        "## Evidence boundary",
        "",
        "VERIFIED: exact 2,400-arm completion, 1,200 paired cases, immutable input/integrity gates, "
        "frozen paired non-inferiority statistics, and complete trace archival.",
        "",
        "UNVERIFIED: any causal failure mechanism, any benefit of CRP/CAG/KD/new methods, and any "
        "claim beyond the frozen ordinary-LIBERO comparison.",
        "",
        "No new model was trained, no checkpoint was modified, and Task 12 was not started.",
        "",
        "```text",
        stats["status"],
        "```",
    ]
    (root / "TASK11_FINAL_REPORT.md").write_text("\n".join(final_report) + "\n")
    provenance = [
        "# CRP-VLA Task 11 Provenance",
        "",
        f"- Repository commit: `{execution['repository_commit']}`",
        f"- Repository branch: `{execution['repository_branch']}`",
        "- Repository dirty during execution/finalization: `False`",
        f"- LIBERO commit: `{execution['libero_commit']}`",
        f"- Python: `{execution['environment']['python']}`",
        f"- PyTorch/CUDA/device: `{execution['environment']['torch']}` / `{execution['environment']['cuda_runtime']}` / `{execution['environment']['cuda_device']}`",
        f"- Confirmation1200 manifest SHA-256: `{TASK10_MANIFEST_SHA256}`",
        f"- Base model SHA-256: `{execution['base_model_sha256']}`",
        f"- 20k Snap-1 model SHA-256: `{execution['selected_model_sha256']}`",
        f"- Evaluator SHA-256: `{execution['evaluator_hash']}`",
        f"- Processor SHA-256: `{execution['processor_hash']}`",
        f"- Shards / formal results / exact pairs: `40 / {len(records)} / {len(by_case)}`",
        f"- Infrastructure failures / admitted retries: `{len(failures)} / {len(retry_results)}`",
        f"- Finalization unix ns: `{time.time_ns()}`",
        "- New training: `false`",
        "- Checkpoint modification: `false`",
        "- Frozen protocol modification: `false`",
    ]
    (root / "TASK11_PROVENANCE.md").write_text("\n".join(provenance) + "\n")

    required = [
        "TASK11_PRERUN_AUDIT.md",
        "CONFIRMATION1200_EXECUTION_MANIFEST.json",
        "CONFIRMATION1200_EXECUTION_LOG.md",
        "CONFIRMATION1200_FAILURE_AND_RETRY_LOG.md",
        "CONFIRMATION1200_RAW_RESULTS.jsonl",
        "CONFIRMATION1200_PAIRED_RESULTS.json",
        "CONFIRMATION1200_COMPLETION_AUDIT.md",
        "CONFIRMATION1200_PRIMARY_STATISTICS.json",
        "CONFIRMATION1200_PRIMARY_REPORT.md",
        "CONFIRMATION1200_TASK_BREAKDOWN.md",
        "CONFIRMATION1200_TRACE_MANIFEST.json",
        "CONFIRMATION1200_DIAGNOSTIC_TRACE_INVENTORY.json",
        "MECHANISM_DATA_READINESS.md",
        "TASK11_FINAL_REPORT.md",
        "TASK11_PROVENANCE.md",
    ]
    checksum_paths = [root / name for name in required]
    checksum_paths += sorted((root / "shards").glob("*.json"))
    checksum_paths += sorted((root / "execution" / "shard_status").glob("*.json"))
    checksum_paths += result_paths
    checksum_paths += [Path(record["trace_path"]) for record in records]
    lines = [f"{file_sha256(path)}  {path.relative_to(root)}" for path in checksum_paths]
    (root / "TASK11_SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": stats["status"],
                "records": len(records),
                "pairs": len(by_case),
                "replans": trace_replans,
                "checksum_entries": len(lines),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
