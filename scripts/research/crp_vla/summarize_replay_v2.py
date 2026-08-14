#!/usr/bin/env python
"""Validate and summarize a replay-v2 smoke or development run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import (
    ARMS,
    PAIRWISE_COMPARISONS,
    file_sha256,
    paired_comparison,
    validate_case_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def case_key(result: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        result["suite"],
        int(result["task_id"]),
        int(result["init_state_id"]),
        int(result["env_seed"]),
    )


def first_close(actions: np.ndarray) -> int | None:
    indices = np.flatnonzero(actions[:, -1] > 0)
    return int(indices[0]) if len(indices) else None


def initial_prefix_diagnostics(by_arm: dict[str, dict[str, Any]], execution_horizon: int) -> dict[str, Any]:
    arrays = {}
    for arm, result in by_arm.items():
        manifest = json.loads(Path(result["trace_manifest"]["path"]).read_text())
        numeric_path = Path(manifest["numeric_actions"]["path"])
        if file_sha256(numeric_path) != manifest["numeric_actions"]["sha256"]:
            raise ValueError(f"Numeric trace hash mismatch for {arm}")
        with np.load(numeric_path, allow_pickle=False) as archive:
            arrays[arm] = archive["executed_actions"][:execution_horizon].astype(np.float64)
    teacher = arrays["base10"]
    output = {}
    for arm in ARMS:
        candidate = arrays[arm]
        common = min(len(teacher), len(candidate))
        if common == 0:
            raise ValueError(f"No executed-prefix support for {arm}")
        delta = candidate[:common] - teacher[:common]
        teacher_close = first_close(teacher[:common])
        candidate_close = first_close(candidate[:common])
        output[arm] = {
            "executed_prefix_mse_vs_base10": float(np.mean(delta**2)),
            "translation_mse_vs_base10": float(np.mean(delta[:, :3] ** 2)),
            "rotation_mse_vs_base10": float(np.mean(delta[:, 3:6] ** 2)),
            "gripper_class_agreement_vs_base10": float(
                np.mean((candidate[:common, -1] > 0) == (teacher[:common, -1] > 0))
            ),
            "gripper_class_disagreement_vs_base10": float(
                np.mean((candidate[:common, -1] > 0) != (teacher[:common, -1] > 0))
            ),
            "first_close_error_vs_base10": (
                abs(candidate_close - teacher_close)
                if teacher_close is not None and candidate_close is not None
                else None
            ),
        }
    return output


def build_tables(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in record["results"]:
        grouped[case_key(result)][result["arm"]] = result
    rows = []
    events = []
    for key in sorted(grouped):
        by_arm = grouped[key]
        validate_case_records(list(by_arm.values()))
        prefix = initial_prefix_diagnostics(by_arm, int(record["results"][0].get("execution_horizon", 10)))
        reference = by_arm["base10"]
        row = {
            "suite": key[0],
            "task_id": key[1],
            "init_state_id": key[2],
            "env_seed": key[3],
            "task": reference["task"],
            "capsule_sha256": reference["capsule_sha256"],
            "canonical_input_sha256": reference["canonical_input_sha256"],
            "initial_sim_state_sha256": reference["initial_sim_state_sha256"],
            "noise_schedule_sha256": reference["noise_schedule_sha256"],
        }
        for arm in ARMS:
            result = by_arm[arm]
            row[f"{arm}_success"] = int(result["success"])
            row[f"{arm}_steps_run"] = int(result["steps_run"])
            latencies = np.asarray(result["latency_ms_per_replan"], dtype=np.float64)
            row[f"{arm}_latency_median_ms"] = float(np.median(latencies))
            for metric, value in prefix[arm].items():
                row[f"{arm}_{metric}"] = value
            events.append(
                {
                    "suite": key[0],
                    "task_id": key[1],
                    "init_state_id": key[2],
                    "env_seed": key[3],
                    "arm": arm,
                    "success": bool(result["success"]),
                    "failure_category": result["failure_classification"]["category"],
                    "failure_evidence_json": json.dumps(
                        result["failure_classification"]["evidence"], sort_keys=True
                    ),
                    "first_target_contact_step": result["events"]["first_target_contact"].get("step"),
                    "first_gripper_close_step": result["events"]["first_gripper_close"].get("step"),
                    "first_object_lift_step": result["events"]["first_object_lift"].get("step"),
                    "first_receptacle_entry_step": result["events"]["first_receptacle_entry"].get("step"),
                    "collision_observed": bool(result["events"]["collision"].get("observed")),
                    "object_dropped": bool(result["events"]["object_dropped"].get("observed")),
                    "trace_manifest_sha256": result["trace_manifest"]["sha256"],
                }
            )
        rows.append(row)
    return rows, events


def arm_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        arm: {
            "successes": int(sum(row[f"{arm}_success"] for row in rows)),
            "cases": len(rows),
            "success_rate": float(np.mean([row[f"{arm}_success"] for row in rows])),
            "latency_median_ms": float(np.median([row[f"{arm}_latency_median_ms"] for row in rows])),
        }
        for arm in ARMS
    }


def point_biserial(rows: list[dict[str, Any]], arm: str, metric: str) -> dict[str, Any]:
    raw_values = [row[f"{arm}_{metric}"] for row in rows]
    values = np.asarray([np.nan if value is None else value for value in raw_values], dtype=np.float64)
    failures = 1 - np.asarray([row[f"{arm}_success"] for row in rows], dtype=np.float64)
    supported = np.isfinite(values)
    if supported.sum() < 3 or len(np.unique(failures[supported])) < 2 or np.std(values[supported]) == 0:
        return {"estimate": None, "supported_cases": int(supported.sum()), "reason": "insufficient variation"}
    return {
        "estimate": float(np.corrcoef(values[supported], failures[supported])[0, 1]),
        "supported_cases": int(supported.sum()),
        "interpretation": "descriptive association; not causal and not a selection criterion",
    }


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def build_summary(
    record: dict[str, Any], rows: list[dict[str, Any]], events: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    expected_status = "INFRASTRUCTURE_SMOKE_PASSED" if args.phase == "smoke" else "DEVELOPMENT_COMPLETED"
    if record.get("status") != expected_status or record.get("phase") != args.phase:
        raise ValueError("Manifest is not a completed run of the selected phase")
    output = {
        "schema_version": 2,
        "status": "SMOKE_ADMITTED" if args.phase == "smoke" else "DEVELOPMENT_ANALYZED",
        "phase": args.phase,
        "formal_gate_unchanged": True,
        "crp_training_hold": True,
        "manifest_sha256": file_sha256(args.manifest),
        "case_count": len(rows),
        "rollout_count": len(events),
        "admission": {
            "all_capsules_verified": True,
            "all_six_arm_invariants_verified": True,
            "infrastructure_exception_count": 0,
        },
    }
    if args.phase == "smoke":
        return output
    output["overall"] = {"arms": arm_rates(rows)}
    output["by_suite"] = {
        suite: {"case_count": len(subset), "arms": arm_rates(subset)}
        for suite in sorted({row["suite"] for row in rows})
        if (subset := [row for row in rows if row["suite"] == suite])
    }
    output["comparisons"] = {
        f"{left}_vs_{right}": paired_comparison(
            rows, left, right, args.bootstrap_repeats, args.bootstrap_seed + index
        )
        for index, (left, right) in enumerate(PAIRWISE_COMPARISONS)
    }
    output["failure_categories"] = {
        category: sum(event["failure_category"] == category for event in events)
        for category in sorted({event["failure_category"] for event in events if event["failure_category"]})
    }
    output["offline_closed_loop_associations"] = {
        arm: {
            "executed_prefix_mse_vs_failure": point_biserial(rows, arm, "executed_prefix_mse_vs_base10"),
            "gripper_class_disagreement_vs_failure": point_biserial(
                rows, arm, "gripper_class_disagreement_vs_base10"
            ),
            "first_close_timing_error_vs_failure": point_biserial(rows, arm, "first_close_error_vs_base10"),
        }
        for arm in ("base2", "base1", "snap2", "snap1")
    }
    return output


def write_report(summary: dict[str, Any], path: Path) -> None:
    rates = summary["overall"]["arms"]
    comparisons = summary["comparisons"]
    base_recovery = comparisons["base2_vs_base1"]["success_rate_difference"]
    snap_recovery = comparisons["snap2_vs_snap1"]["success_rate_difference"]
    base_10_to_2 = comparisons["base10_vs_base2"]["success_rate_difference"]
    snap_10_to_2 = comparisons["snap10_vs_snap2"]["success_rate_difference"]
    base_cost = rates["base2"]["latency_median_ms"] - rates["base1"]["latency_median_ms"]
    snap_cost = rates["snap2"]["latency_median_ms"] - rates["snap1"]["latency_median_ms"]
    lines = [
        "# Replay-v2 Development Results",
        "",
        "**Status: DIAGNOSTIC/DEVELOPMENT. The original formal gate remains FAIL; CRP remains HOLD.**",
        "",
        "## Six-arm success and latency",
        "",
        "| Arm | Success | Rate | Median policy latency |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = rates[arm]
        lines.append(
            f"| {arm} | {item['successes']}/{item['cases']} | {item['success_rate']:.1%} | "
            f"{item['latency_median_ms']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. 2 NFE recovery over 1 NFE: Base {base_recovery:+.1%}; Snap {snap_recovery:+.1%}.",
            f"2. 10→2 versus 2→1 success loss: Base {base_10_to_2:+.1%} versus {base_recovery:+.1%}; "
            f"Snap {snap_10_to_2:+.1%} versus {snap_recovery:+.1%}.",
            "3. Executed-prefix error association with failure is reported descriptively in `summary.json`; "
            "it is not interpreted causally.",
            "4. Gripper class-disagreement association is reported on the same cases; unavailable transition timing "
            "support is not imputed.",
            "5. Base and Snap NFE patterns are shown separately above; no equality is assumed.",
            f"6. Observed 2-NFE latency increment: Base {base_cost:+.2f} ms; Snap {snap_cost:+.2f} ms, "
            f"for success changes {base_recovery:+.1%}/{snap_recovery:+.1%} respectively.",
            "",
            "## Evidence boundary",
            "",
            "All statistics use the frozen 40-case development set. Failure labels are emitted only when registered "
            "trajectory/contact evidence supports them; ambiguous cases remain UNKNOWN. This set may be used for the "
            "pre-registered mature-baseline checkpoint selection, but it is not a confirmation set.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_sha_manifest(root: Path, files: list[Path]) -> None:
    (root / "sha256_manifest.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(files))
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    record = json.loads(args.manifest.read_text())
    rows, events = build_tables(record)
    expected = 8 if args.phase == "smoke" else 40
    if len(rows) != expected or len(events) != expected * len(ARMS):
        raise ValueError("Completed manifest has the wrong case or rollout count")
    summary = build_summary(record, rows, events, args)
    phase_root = args.output_root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    summary_path = phase_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    files = [summary_path]
    if args.phase == "smoke":
        report_path = phase_root / "SMOKE_RESULTS.md"
        report_path.write_text(
            "# Replay-v2 Phase A Smoke\n\n"
            "VERIFIED: 8 capsules and 48 same-process rollouts passed all replay-v2 admission invariants. "
            "This phase is infrastructure-only and is not included in scientific analysis.\n"
        )
        files.append(report_path)
    else:
        per_case_path = args.output_root / "per_case_results.parquet"
        event_path = args.output_root / "event_traces.parquet"
        write_parquet(rows, per_case_path)
        write_parquet(events, event_path)
        report_path = args.output_root / "DEVELOPMENT_RESULTS.md"
        write_report(summary, report_path)
        files.extend((per_case_path, event_path, report_path))
    excluded_parts = {"capsules", "traces", "libero_standard_config"}
    files.extend(
        path
        for path in args.output_root.rglob("*")
        if path.is_file()
        and path.name != "sha256_manifest.txt"
        and not excluded_parts.intersection(path.relative_to(args.output_root).parts)
    )
    files = sorted(set(files))
    write_sha_manifest(args.output_root, files)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
