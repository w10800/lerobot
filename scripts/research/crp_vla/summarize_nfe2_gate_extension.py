#!/usr/bin/env python
"""Summarize the frozen four-arm gate plus its two-arm NFE=2 diagnostic."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FORMAL_ARMS = ("base10", "base1", "snap10", "snap1")
DIAGNOSTIC_ARMS = ("base2", "snap2")
ALL_ARMS = ("base10", "base2", "base1", "snap10", "snap2", "snap1")
COMPARISONS = (
    ("base2", "base10"),
    ("base2", "base1"),
    ("snap2", "snap10"),
    ("snap2", "snap1"),
    ("snap2", "base10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--nfe2-manifest", type=Path, required=True)
    parser.add_argument("--base-latency", type=Path, required=True)
    parser.add_argument("--snap-latency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(result: dict) -> tuple[str, int, int, int]:
    return (
        result["suite"],
        int(result["task_id"]),
        int(result["init_state_id"]),
        int(result["env_seed"]),
    )


def group_results(results: list[dict], expected_arms: tuple[str, ...]) -> dict[tuple, dict]:
    grouped: dict[tuple, dict] = defaultdict(dict)
    for result in results:
        grouped[case_key(result)][result["arm"]] = result
    if any(set(arms) != set(expected_arms) for arms in grouped.values()):
        raise ValueError(f"Every case must contain exactly {expected_arms}")
    return dict(grouped)


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    left_win = int(np.sum((left == 1) & (right == 0)))
    right_win = int(np.sum((left == 0) & (right == 1)))
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_win, right_win) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_success_right_failure": left_win,
        "left_failure_right_success": right_win,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repeats, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def task_cluster_bootstrap_ci(
    rows: list[dict], left: str, right: str, repeats: int, seed: int
) -> list[float]:
    task_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        task_values[(row["suite"], row["task_id"])].append(
            float(row[f"{left}_success"] - row[f"{right}_success"])
        )
    task_means = np.asarray([np.mean(values) for values in task_values.values()])
    return bootstrap_ci(task_means, repeats, seed)


def comparison_summary(rows: list[dict], left: str, right: str, repeats: int, seed: int) -> dict:
    left_values = np.asarray([row[f"{left}_success"] for row in rows], dtype=np.int8)
    right_values = np.asarray([row[f"{right}_success"] for row in rows], dtype=np.int8)
    differences = left_values.astype(np.float64) - right_values.astype(np.float64)
    return {
        "left": left,
        "right": right,
        "success_rate_difference": float(differences.mean()),
        "case_paired_bootstrap_ci95": bootstrap_ci(differences, repeats, seed),
        "task_cluster_bootstrap_ci95": task_cluster_bootstrap_ci(rows, left, right, repeats, seed + 10_000),
        "mcnemar": exact_mcnemar(left_values, right_values),
    }


def arm_rates(rows: list[dict]) -> dict:
    return {
        arm: {
            "successes": int(sum(row[f"{arm}_success"] for row in rows)),
            "cases": len(rows),
            "success_rate": float(np.mean([row[f"{arm}_success"] for row in rows])),
        }
        for arm in ALL_ARMS
    }


def latency_summary(path: Path) -> dict:
    record = json.loads(path.read_text())
    by_nfe = {int(item["num_steps"]): item for item in record["results"]}
    if not {1, 2}.issubset(by_nfe):
        raise ValueError(f"Latency manifest {path} must contain 1 and 2 NFE")
    one = float(by_nfe[1]["latency_ms_median"])
    two = float(by_nfe[2]["latency_ms_median"])
    return {
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "nfe1_median_ms": one,
        "nfe2_median_ms": two,
        "nfe1_savings_vs_nfe2_ms": two - one,
        "warmups_per_nfe": record.get("warmups_per_nfe"),
        "latency_repeats_per_nfe": record.get("latency_repeats_per_nfe"),
    }


def build_rows(formal: dict, diagnostic: dict) -> list[dict]:
    formal_cases = group_results(formal["results"], FORMAL_ARMS)
    diagnostic_cases = group_results(diagnostic["results"], DIAGNOSTIC_ARMS)
    if set(formal_cases) != set(diagnostic_cases):
        raise ValueError("Formal and NFE=2 diagnostic case sets differ")
    rows = []
    for key in sorted(formal_cases):
        all_results = {**formal_cases[key], **diagnostic_cases[key]}
        reference = all_results["base10"]
        for arm, result in all_results.items():
            if result["initial_sim_state_sha256"] != reference["initial_sim_state_sha256"]:
                raise ValueError(f"Simulator-state mismatch at {key} {arm}")
            if result["initial_observation_hashes"] != reference["initial_observation_hashes"]:
                raise ValueError(f"Canonical-input mismatch at {key} {arm}")
        for arm in DIAGNOSTIC_ARMS:
            common = min(
                len(all_results[arm]["noise_sha256_per_replan"]),
                len(reference["noise_sha256_per_replan"]),
            )
            if (
                all_results[arm]["noise_sha256_per_replan"][:common]
                != reference["noise_sha256_per_replan"][:common]
            ):
                raise ValueError(f"Noise-prefix mismatch at {key} {arm}")
        row = {
            "suite": key[0],
            "task_id": key[1],
            "init_state_id": key[2],
            "env_seed": key[3],
            "task": reference["task"],
            "simulator_state_sha256": reference["initial_sim_state_sha256"],
            "canonical_input_sha256": all_results["base2"]["canonical_input_sha256"],
        }
        for arm in ALL_ARMS:
            result = all_results[arm]
            row[f"{arm}_success"] = int(result["success"])
            row[f"{arm}_steps_run"] = int(result["steps_run"])
            row[f"{arm}_success_timestep"] = (
                int(result.get("success_timestep", result["steps_run"])) if result["success"] else None
            )
            row[f"{arm}_action_stream_sha256"] = result["action_stream_sha256"]
        rows.append(row)
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def write_report(summary: dict, path: Path) -> None:
    rates = summary["overall"]["arms"]
    comparisons = summary["comparisons"]
    base_monotonic = (
        rates["base10"]["success_rate"] >= rates["base2"]["success_rate"] >= rates["base1"]["success_rate"]
    )
    snap_monotonic = (
        rates["snap10"]["success_rate"] >= rates["snap2"]["success_rate"] >= rates["snap1"]["success_rate"]
    )
    base_recovery = comparisons["base2_vs_base1"]["success_rate_difference"]
    snap_recovery = comparisons["snap2_vs_snap1"]["success_rate_difference"]
    latency = summary["latency"]
    lines = [
        "# NFE=2 Diagnostic Extension Summary",
        "",
        "**Status: DIAGNOSTIC — does not replace the frozen formal gate and does not authorize training.**",
        "",
        "## Results",
        "",
        "| Arm | Successes | Rate |",
        "|---|---:|---:|",
    ]
    for arm in ALL_ARMS:
        item = rates[arm]
        lines.append(f"| {arm} | {item['successes']}/{item['cases']} | {item['success_rate']:.1%} |")
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            f"- 2 NFE recovery over 1 NFE: Base {base_recovery:+.1%}; Snap {snap_recovery:+.1%}. This is diagnostic, not a new non-inferiority decision.",
            f"- Monotonic 10→2→1 degradation: Base **{base_monotonic}**; Snap **{snap_monotonic}**.",
            f"- Warmed median 1-NFE savings relative to 2 NFE: Base {latency['base']['nfe1_savings_vs_nfe2_ms']:.2f} ms; Snap {latency['snap']['nfe1_savings_vs_nfe2_ms']:.2f} ms. Whether that saving is worthwhile must be judged against the paired success differences above; this report does not change the frozen margin.",
            "- 2 NFE can be retained as a **candidate diagnostic upper bound / correction teacher**, but this evidence does not approve conditional-correction training.",
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Difference | Case bootstrap 95% CI | Task-cluster 95% CI | Discordant (L+/R−, L−/R+) | Exact McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in comparisons.items():
        case_ci = item["case_paired_bootstrap_ci95"]
        task_ci = item["task_cluster_bootstrap_ci95"]
        mc = item["mcnemar"]
        lines.append(
            f"| {name} | {item['success_rate_difference']:+.1%} | "
            f"[{case_ci[0]:+.1%}, {case_ci[1]:+.1%}] | "
            f"[{task_ci[0]:+.1%}, {task_ci[1]:+.1%}] | "
            f"{mc['left_success_right_failure']}, {mc['left_failure_right_success']} | "
            f"{mc['two_sided_exact_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "All 200 added rollouts reuse the frozen paired cases. The original 400-rollout result remains the only formal gate. No -3% decision margin is applied here.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    formal = json.loads(args.formal_manifest.read_text())
    diagnostic = json.loads(args.nfe2_manifest.read_text())
    if formal.get("rollout_count") != 400 or diagnostic.get("rollout_count") != 200:
        raise ValueError("Expected the frozen 400 rollouts and exactly 200 diagnostic rollouts")
    if diagnostic.get("run_kind") != "nfe2_diagnostic" or diagnostic.get("status") != "DIAGNOSTIC_ANALYZED":
        raise ValueError("Input is not a completed NFE=2 diagnostic")
    if diagnostic.get("reference_manifest_sha256") != file_sha256(args.formal_manifest):
        raise ValueError("Diagnostic does not reference the supplied frozen formal manifest")
    rows = build_rows(formal, diagnostic)
    comparisons = {
        f"{left}_vs_{right}": comparison_summary(
            rows, left, right, args.bootstrap_repeats, args.bootstrap_seed + index
        )
        for index, (left, right) in enumerate(COMPARISONS)
    }
    suites = sorted({row["suite"] for row in rows})
    summary = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ANALYZED",
        "formal_gate_unchanged": True,
        "formal_manifest": str(args.formal_manifest.resolve()),
        "formal_manifest_sha256": file_sha256(args.formal_manifest),
        "nfe2_manifest": str(args.nfe2_manifest.resolve()),
        "nfe2_manifest_sha256": file_sha256(args.nfe2_manifest),
        "case_count": len(rows),
        "added_rollout_count": len(rows) * len(DIAGNOSTIC_ARMS),
        "bootstrap_unit_primary": "paired_case",
        "bootstrap_unit_secondary": "task_cluster",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "overall": {"arms": arm_rates(rows)},
        "by_suite": {
            suite: {"case_count": len(subset), "arms": arm_rates(subset)}
            for suite in suites
            if (subset := [row for row in rows if row["suite"] == suite])
        },
        "comparisons": comparisons,
        "latency": {
            "base": latency_summary(args.base_latency),
            "snap": latency_summary(args.snap_latency),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, args.output_dir / "per_case_results.parquet")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary, args.output_dir / "NFE2_DIAGNOSTIC_SUMMARY.md")
    provenance_rows = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in (
            args.formal_manifest,
            args.nfe2_manifest,
            args.base_latency,
            args.snap_latency,
        )
    ]
    with (args.output_dir / "source_provenance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256"))
        writer.writeheader()
        writer.writerows(provenance_rows)
    output_files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "sha256_manifest.txt"
    )
    (args.output_dir / "sha256_manifest.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in output_files)
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
