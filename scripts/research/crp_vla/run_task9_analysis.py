#!/usr/bin/env python
"""Audit CRP-VLA Task 9 inputs and generate the non-rollout analyses.

This command is deliberately incapable of training or stepping a simulator.  It
reads frozen Task 0--8 artifacts, computes Task 9 Parts A/B/C1, and writes only
under the new Task 9 output root.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from task9_common import (
    NONINFERIORITY_MARGIN,
    REGISTERED_STEPS,
    SELECTED_MODEL_SHA256,
    SELECTED_STEP,
    canonical_json_sha256,
    detect_duplicate_trace_paths_and_hashes,
    file_sha256,
    leave_one_task_out_predictions,
    load_json,
    paired_flip_summary,
    prediction_metrics,
    selection_bootstrap,
    spearman,
    standardized_mean_difference,
)

BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260815
TASK9_STATUS_BLOCKED = "BLOCKED_WITH_ADMITTED_FAILURE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/crp_vla/task9_baseline_validation"),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--sample-trials", type=int, default=500)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("None of the candidate inputs exists: " + ", ".join(map(str, paths)))


def verify_text_sha_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", maxsplit=1)
        member = path.parent / name
        actual = file_sha256(member) if member.exists() else None
        records.append(
            {
                "path": str(member.resolve()),
                "expected_sha256": digest,
                "actual_sha256": actual,
                "passed": actual == digest,
            }
        )
    return records


def result_case_id(result: dict[str, Any]) -> str:
    return (
        f"{result['suite']}-task{int(result['task_id']):02d}-"
        f"init{int(result['init_state_id']):02d}-seed{int(result['env_seed'])}"
    )


def task_cluster_id(result: dict[str, Any]) -> str:
    return f"{result['suite']}:{int(result['task_id'])}"


def trace_first_row(path: Path) -> dict[str, Any]:
    trace_path = path.with_name(path.name.replace(".sha256.json", ".trace.jsonl.gz"))
    with gzip.open(trace_path, "rt") as stream:
        return json.loads(next(stream))


def audit_inputs(repo: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    metric_staged = output_root / "source_inputs" / "task6_metric_validity"
    paths = {
        "maturation_manifest": repo / "artifacts/crp_vla/maturation/checkpoint_manifest.json",
        "maturation_sha_manifest": repo / "artifacts/crp_vla/maturation/sha256_manifest.txt",
        "replay_dev_manifest": repo / "artifacts/crp_vla/replay_v2/development/run_manifest.json",
        "replay_dev_summary": repo / "artifacts/crp_vla/replay_v2/development/summary.json",
        "replay_dev_cases": repo / "artifacts/crp_vla/replay_v2/per_case_results.parquet",
        "replay_dev_events": repo / "artifacts/crp_vla/replay_v2/event_traces.parquet",
        "formal100": first_existing(
            [
                repo / "artifacts/crp_vla/followup/libero_paired_final_v1.json",
                repo / "artifacts/crp_vla/followup_20260813/libero_paired_final_v1.json",
            ]
        ),
        "metric_validity_statistics": first_existing(
            [
                repo / "artifacts/crp_vla/followup_20260814/metric_validity/statistics.json",
                metric_staged / "statistics.json",
            ]
        ),
        "metric_validity_records": first_existing(
            [
                repo / "artifacts/crp_vla/followup_20260814/metric_validity/per_record_metrics.parquet",
                metric_staged / "per_record_metrics.parquet",
            ]
        ),
        "metric_validity_sha_manifest": first_existing(
            [
                repo / "artifacts/crp_vla/followup_20260814/metric_validity/sha256_manifest.txt",
                metric_staged / "sha256_manifest.txt",
            ]
        ),
    }
    for step in REGISTERED_STEPS:
        paths[f"factual_{step}"] = repo / f"artifacts/crp_vla/maturation/factual/step_{step:06d}.json"
        paths[f"replay_{step}"] = repo / f"artifacts/crp_vla/maturation/replay/step_{step:06d}.json"

    errors: list[str] = []
    warnings: list[str] = []
    sha_records: list[dict[str, Any]] = []
    for label, path in paths.items():
        if not path.exists():
            errors.append(f"Missing required input {label}: {path}")
            continue
        sha_records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
        )

    maturation = load_json(paths["maturation_manifest"])
    if maturation.get("selected_step") != SELECTED_STEP:
        errors.append("Maturation selected step is not frozen 20k")
    if maturation.get("selected", {}).get("checkpoint_sha256") != SELECTED_MODEL_SHA256:
        errors.append("Maturation selected model SHA-256 mismatch")
    checkpoint_rows = maturation.get("checkpoints", [])
    if [row.get("step") for row in checkpoint_rows] != list(REGISTERED_STEPS):
        errors.append("Maturation checkpoint coverage/order mismatch")

    all_maturation_results: list[dict[str, Any]] = []
    for step in REGISTERED_STEPS:
        factual = load_json(paths[f"factual_{step}"])
        replay = load_json(paths[f"replay_{step}"])
        if factual.get("checkpoint_step") != step or factual.get("status") != "FACTUAL_CHECKPOINT_EVALUATED":
            errors.append(f"Factual checkpoint {step} is incomplete or mismatched")
        if factual.get("task_count") != 40 or factual.get("record_count") != 360:
            errors.append(f"Factual checkpoint {step} coverage mismatch")
        if replay.get("checkpoint_step") != step or replay.get("status") != "MATURATION_REPLAY_COMPLETED":
            errors.append(f"Maturation replay {step} is incomplete or mismatched")
        results = replay.get("results", [])
        if len(results) != 120 or any(result.get("status") != "COMPLETED" for result in results):
            errors.append(f"Maturation replay {step} does not contain 120 COMPLETED records")
        if any(result.get("checkpoint_step") != step for result in results):
            errors.append(f"Maturation replay {step} result checkpoint mismatch")
        if any(result.get("nfe") not in (10, 2, 1) for result in results):
            errors.append(f"Maturation replay {step} contains invalid NFE")
        all_maturation_results.extend(results)
    if len(all_maturation_results) != 720:
        errors.append(f"Expected 720 maturation rollouts, found {len(all_maturation_results)}")
    try:
        detect_duplicate_trace_paths_and_hashes(all_maturation_results)
    except ValueError as error:
        errors.append(str(error))
    trace_recomputed = 0
    for result in all_maturation_results:
        trace = result.get("trace_manifest", {})
        trace_path = Path(trace.get("path", ""))
        if not trace_path.exists():
            errors.append(f"Missing maturation trace manifest: {trace_path}")
            continue
        actual = file_sha256(trace_path)
        if actual != trace.get("sha256"):
            errors.append(f"Maturation trace hash mismatch: {trace_path}")
        else:
            trace_recomputed += 1

    dev = load_json(paths["replay_dev_manifest"])
    dev_results = dev.get("results", [])
    if dev.get("status") != "DEVELOPMENT_COMPLETED" or len(dev_results) != 240:
        errors.append("Replay-v2 development manifest is incomplete")
    if any(result.get("status") != "COMPLETED" for result in dev_results):
        errors.append("Replay-v2 development contains non-COMPLETED record")
    try:
        detect_duplicate_trace_paths_and_hashes(dev_results)
    except ValueError as error:
        errors.append(f"Replay-v2: {error}")

    formal = load_json(paths["formal100"])
    formal_results = formal.get("results", [])
    formal_cases = {
        (r["suite"], int(r["task_id"]), int(r["init_state_id"]), int(r["env_seed"])) for r in formal_results
    }
    dev_cases = {
        (r["suite"], int(r["task_id"]), int(r["init_state_id"]), int(r["env_seed"])) for r in dev_results
    }
    if formal.get("case_count") != 100 or len(formal_results) != 400 or len(formal_cases) != 100:
        errors.append("Formal100 coverage is not 100 paired cases / 400 rollouts")
    if len(dev_cases) != 40:
        errors.append("Development coverage is not 40 cases")
    if formal_cases & dev_cases:
        errors.append("formal100 and dev40 case identities overlap")
    formal_state_hashes = {r.get("initial_sim_state_sha256") for r in formal_results}
    dev_state_hashes = {r.get("initial_sim_state_sha256") for r in dev_results}
    if None in formal_state_hashes or None in dev_state_hashes:
        errors.append("formal100/dev40 state hashes are not identifiable")
    if formal_state_hashes & dev_state_hashes:
        errors.append("formal100 and dev40 simulator-state hashes overlap")

    metric_sha = verify_text_sha_manifest(paths["metric_validity_sha_manifest"])
    if any(not row["passed"] for row in metric_sha):
        errors.append("Task 6 metric-validity SHA manifest failed")
    for row in metric_sha:
        sha_records.append({"label": "metric_validity_member", **row})

    maturation_sha = verify_text_sha_manifest(paths["maturation_sha_manifest"])
    if any(not row["passed"] for row in maturation_sha):
        errors.append("Maturation report SHA manifest failed")

    integrity = {
        "schema_version": 1,
        "status": "INPUT_INTEGRITY_PASSED" if not errors else TASK9_STATUS_BLOCKED,
        "repository_revision_before_task9": git_value(repo, "rev-parse", "HEAD"),
        "repository_status_before_task9": git_value(repo, "status", "--short"),
        "python": platform.python_version(),
        "selected_checkpoint_step": SELECTED_STEP,
        "selected_model_sha256_expected": SELECTED_MODEL_SHA256,
        "selected_model_sha256_observed": maturation.get("selected", {}).get("checkpoint_sha256"),
        "maturation_rollouts_completed": len(all_maturation_results),
        "maturation_trace_hashes_recomputed": trace_recomputed,
        "maturation_unique_trace_paths": len({r["trace_manifest"]["path"] for r in all_maturation_results}),
        "maturation_unique_trace_hashes": len(
            {r["trace_manifest"]["sha256"] for r in all_maturation_results}
        ),
        "formal100_case_count": len(formal_cases),
        "dev40_case_count": len(dev_cases),
        "formal100_dev40_case_overlap_count": len(formal_cases & dev_cases),
        "formal100_dev40_state_overlap_count": len(formal_state_hashes & dev_state_hashes),
        "resolved_inputs": {label: str(path.resolve()) for label, path in paths.items()},
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "input_integrity.json", integrity)
    write_json(output_root / "input_sha256_manifest.json", {"files": sha_records})
    report = [
        "# Task 9 Input Integrity Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        f"- Verification Status: {'VERIFIED' if not errors else 'UNVERIFIED'}",
        "- Version Label: crp_vla_task9_input_v1",
        "",
        f"**Status: {integrity['status']}**",
        "",
        f"- Frozen checkpoint: `{SELECTED_STEP}`",
        f"- Frozen model SHA-256 matched: `{integrity['selected_model_sha256_observed'] == SELECTED_MODEL_SHA256}`",
        f"- Maturation rollouts: `{len(all_maturation_results)}/720` COMPLETED",
        f"- Recomputed maturation trace hashes: `{trace_recomputed}/720`",
        f"- formal100/dev40 case overlap: `{len(formal_cases & dev_cases)}`",
        f"- formal100/dev40 state-hash overlap: `{len(formal_state_hashes & dev_state_hashes)}`",
        "",
        "## Resolved inputs",
        "",
        *[f"- `{label}`: `{path.resolve()}`" for label, path in paths.items()],
        "",
        "## Errors",
        "",
        *(errors or ["None."]),
    ]
    (output_root / "INPUT_INTEGRITY_REPORT.md").write_text("\n".join(report) + "\n")
    return integrity, paths


def build_matrix(paths: dict[str, Path], output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    maturation = load_json(paths["maturation_manifest"])
    eligibility = {int(row["step"]): bool(row["eligible"]) for row in maturation["checkpoints"]}
    records: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {"case_order": None, "snap": {}, "base": {}}
    by_step: dict[int, dict[tuple[str, int, int, int], dict[int, dict[str, Any]]]] = {}
    for step in REGISTERED_STEPS:
        replay = load_json(paths[f"replay_{step}"])
        grouped: dict[tuple[str, int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
        for result in replay["results"]:
            key = (
                result["suite"],
                int(result["task_id"]),
                int(result["init_state_id"]),
                int(result["env_seed"]),
            )
            grouped[key][int(result["nfe"])] = result
            records.append(
                {
                    "case_id": result_case_id(result),
                    "task_id": task_cluster_id(result),
                    "checkpoint_step": step,
                    "checkpoint_label": f"snap_{step}",
                    "eligible_by_factual_gate": eligibility[step],
                    "nfe": int(result["nfe"]),
                    "success": int(bool(result["success"])),
                    "failure_label": result.get("failure_classification", {}).get("category"),
                    "trace_path": result["trace_manifest"]["path"],
                    "result_sha256": result["trace_manifest"]["sha256"],
                }
            )
        if any(set(arms) != {10, 2, 1} for arms in grouped.values()) or len(grouped) != 40:
            raise ValueError(f"Checkpoint {step} matrix is incomplete")
        by_step[step] = dict(grouped)

    dev = load_json(paths["replay_dev_manifest"])
    base_grouped: dict[tuple[str, int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for result in dev["results"]:
        if result["arm"] not in {"base10", "base2", "base1"}:
            continue
        key = (result["suite"], int(result["task_id"]), int(result["init_state_id"]), int(result["env_seed"]))
        base_grouped[key][int(result["nfe"])] = result
        records.append(
            {
                "case_id": result_case_id(result),
                "task_id": task_cluster_id(result),
                "checkpoint_step": 0,
                "checkpoint_label": "base",
                "eligible_by_factual_gate": True,
                "nfe": int(result["nfe"]),
                "success": int(bool(result["success"])),
                "failure_label": result.get("failure_classification", {}).get("category"),
                "trace_path": result["trace_manifest"]["path"],
                "result_sha256": result["trace_manifest"]["sha256"],
            }
        )
    case_keys = sorted(by_step[SELECTED_STEP])
    if set(case_keys) != set(base_grouped):
        raise ValueError("Maturation and Base development cases are not aligned")
    vectors["case_order"] = [result_case_id(by_step[SELECTED_STEP][key][1]) for key in case_keys]
    vectors["task_order"] = [task_cluster_id(by_step[SELECTED_STEP][key][1]) for key in case_keys]
    for step in REGISTERED_STEPS:
        vectors["snap"][str(step)] = {
            str(nfe): [int(bool(by_step[step][key][nfe]["success"])) for key in case_keys]
            for nfe in (10, 2, 1)
        }
    vectors["base"] = {
        str(nfe): [int(bool(base_grouped[key][nfe]["success"])) for key in case_keys] for nfe in (10, 2, 1)
    }
    frame = pd.DataFrame(records).sort_values(["case_id", "checkpoint_step", "nfe"])
    frame.to_csv(output_root / "per_case_checkpoint_nfe_matrix.csv", index=False)
    frame.to_parquet(output_root / "per_case_checkpoint_nfe_matrix.parquet", index=False)
    write_json(output_root / "checkpoint_success_vectors.json", vectors)
    return frame, vectors


def analyze_part_a(paths: dict[str, Path], output_root: Path) -> dict[str, Any]:
    frame, vectors = build_matrix(paths, output_root)
    tasks = vectors["task_order"]
    selected = np.asarray(vectors["snap"][str(SELECTED_STEP)]["1"], dtype=np.int8)
    comparisons = {
        "1k_snap1": np.asarray(vectors["snap"]["1000"]["1"]),
        "3k_snap1": np.asarray(vectors["snap"]["3000"]["1"]),
        "5k_snap1": np.asarray(vectors["snap"]["5000"]["1"]),
        "30k_snap1": np.asarray(vectors["snap"]["30000"]["1"]),
        "base10": np.asarray(vectors["base"]["10"]),
        "base1": np.asarray(vectors["base"]["1"]),
        "20k_snap2": np.asarray(vectors["snap"]["20000"]["2"]),
        "20k_snap10": np.asarray(vectors["snap"]["20000"]["10"]),
    }
    pairwise = {
        "schema_version": 1,
        "left": "20k_snap1",
        "comparisons": {
            name: paired_flip_summary(selected, other, tasks, seed=BOOTSTRAP_SEED + index)
            for index, (name, other) in enumerate(comparisons.items())
        },
    }
    write_json(output_root / "checkpoint_pairwise_flips.json", pairwise)
    pair_lines = [
        "# Checkpoint Pairwise Analysis",
        "",
        "**VERIFIED development-only paired analysis; no checkpoint is reselected.**",
        "",
        "| Comparator | Difference | n11 | n00 | n10 | n01 | McNemar p | Case CI | Task CI | Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for name, value in pairwise["comparisons"].items():
        pair_lines.append(
            f"| {name} | {value['paired_success_difference']:+.3f} | {value['n_11']} | {value['n_00']} | "
            f"{value['n_10']} | {value['n_01']} | {value['exact_mcnemar_p']:.4f} | "
            f"{value['case_bootstrap_ci95']} | {value['task_cluster_bootstrap_ci95']} | "
            f"{value['successful_case_jaccard']:.3f} |"
        )
    (output_root / "CHECKPOINT_PAIRWISE_ANALYSIS.md").write_text("\n".join(pair_lines) + "\n")

    maturation = load_json(paths["maturation_manifest"])
    eligible = [int(row["step"]) for row in maturation["checkpoints"] if row["eligible"]]
    success_vectors = {
        step: np.asarray(vectors["snap"][str(step)]["1"], dtype=np.int8) for step in REGISTERED_STEPS
    }
    robustness = selection_bootstrap(success_vectors, eligible, tasks)
    frequencies = sorted(
        ((int(step), frequency) for step, frequency in robustness["case_selection_frequencies"].items()),
        key=lambda item: (-item[1], item[0]),
    )
    robustness["20k_case_selection_frequency"] = robustness["case_selection_frequencies"]["20000"]
    robustness["20k_gap_to_second_highest_frequency"] = frequencies[0][1] - frequencies[1][1]
    write_json(output_root / "checkpoint_selection_bootstrap.json", robustness)

    influence_rows = []
    cases = vectors["case_order"]
    for index, selected_step in enumerate(robustness["leave_one_case_out_selected_steps"]):
        influence_rows.append(
            {
                "unit_type": "case",
                "unit_id": cases[index],
                "selected_step_without_unit": selected_step,
                "20k_retained": selected_step == SELECTED_STEP,
            }
        )
    for task, selected_step in zip(
        robustness["leave_one_task_out_task_ids"],
        robustness["leave_one_task_out_selected_steps"],
        strict=True,
    ):
        influence_rows.append(
            {
                "unit_type": "task",
                "unit_id": task,
                "selected_step_without_unit": selected_step,
                "20k_retained": selected_step == SELECTED_STEP,
            }
        )
    pd.DataFrame(influence_rows).to_csv(output_root / "checkpoint_selection_influence.csv", index=False)
    robust_lines = [
        "# Checkpoint Selection Robustness",
        "",
        "**The D-014 selected checkpoint remains frozen at 20k regardless of this audit.**",
        "",
        f"- Eligible checkpoints: `{eligible}` (10k is descriptive-only and excluded).",
        f"- 20k case-bootstrap selection frequency: `{robustness['20k_case_selection_frequency']:.3f}`.",
        f"- Gap to second-highest case-bootstrap frequency: `{robustness['20k_gap_to_second_highest_frequency']:.3f}`.",
        f"- Leave-one-case-out 20k retention: `{robustness['leave_one_case_out_20k_retention']:.3f}`.",
        f"- Leave-one-task-out 20k retention: `{robustness['leave_one_task_out_20k_retention']:.3f}`.",
        f"- A single task can change the bootstrap selector: `{robustness['single_task_determines_20k']}`.",
    ]
    (output_root / "CHECKPOINT_SELECTION_ROBUSTNESS.md").write_text("\n".join(robust_lines) + "\n")

    losses = [float(row["factual_total_validation_loss"]) for row in maturation["checkpoints"]]
    steps = list(REGISTERED_STEPS)
    relationships: dict[str, Any] = {
        "schema_version": 1,
        "analysis_type": "descriptive",
        "warning": "Only six checkpoints are available; correlations are unstable and non-causal.",
        "factual_loss_vs_success": {},
        "checkpoint_step_vs_success": {},
        "per_checkpoint_nfe_monotonicity": {},
        "case_pattern_counts": {},
    }
    for nfe in (1, 2, 10):
        rates = [np.mean(vectors["snap"][str(step)][str(nfe)]) for step in steps]
        relationships["factual_loss_vs_success"][str(nfe)] = spearman(losses, rates)
        relationships["checkpoint_step_vs_success"][str(nfe)] = spearman(steps, rates)
    for step in steps:
        s10 = np.asarray(vectors["snap"][str(step)]["10"])
        s2 = np.asarray(vectors["snap"][str(step)]["2"])
        s1 = np.asarray(vectors["snap"][str(step)]["1"])
        relationships["per_checkpoint_nfe_monotonicity"][str(step)] = {
            "aggregate_snap10_ge_snap2_ge_snap1": bool(s10.sum() >= s2.sum() >= s1.sum()),
            "case_level_violation_count": int(np.sum(~((s10 >= s2) & (s2 >= s1)))),
        }
        relationships["case_pattern_counts"][str(step)] = {
            "1_succeeds_2_fails": int(np.sum((s1 == 1) & (s2 == 0))),
            "1_fails_2_succeeds": int(np.sum((s1 == 0) & (s2 == 1))),
            "2_succeeds_10_fails": int(np.sum((s2 == 1) & (s10 == 0))),
            "2_fails_10_succeeds": int(np.sum((s2 == 0) & (s10 == 1))),
        }
    write_json(output_root / "metric_success_relationships.json", relationships)
    nfe_lines = [
        "# NFE Non-Monotonicity Audit",
        "",
        "**Descriptive analysis only. Six checkpoints cannot support a stable mechanistic interpretation.**",
        "",
        "| Step | 10≥2≥1 aggregate | Case violations | 1+/2− | 1−/2+ | 2+/10− | 2−/10+ |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for step in steps:
        monotonic = relationships["per_checkpoint_nfe_monotonicity"][str(step)]
        patterns = relationships["case_pattern_counts"][str(step)]
        nfe_lines.append(
            f"| {step} | {monotonic['aggregate_snap10_ge_snap2_ge_snap1']} | {monotonic['case_level_violation_count']} | "
            f"{patterns['1_succeeds_2_fails']} | {patterns['1_fails_2_succeeds']} | "
            f"{patterns['2_succeeds_10_fails']} | {patterns['2_fails_10_succeeds']} |"
        )
    (output_root / "NFE_NON_MONOTONICITY_AUDIT.md").write_text("\n".join(nfe_lines) + "\n")
    return {"frame": frame, "vectors": vectors, "pairwise": pairwise, "robustness": robustness}


def _actions_path(result: dict[str, Any]) -> Path:
    manifest = Path(result["trace_manifest"]["path"])
    return manifest.with_name(manifest.name.replace(".sha256.json", ".actions.npz"))


def _load_initial_chunk(result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = _actions_path(result)
    with np.load(path, allow_pickle=False) as archive:
        chunk = np.asarray(archive["predicted_chunks"][0], dtype=np.float64)
    return chunk, trace_first_row(Path(result["trace_manifest"]["path"]))


def _disagreement(left: np.ndarray, right: np.ndarray, horizon: int) -> dict[str, float]:
    difference = left - right
    prefix = difference[:horizon]
    per_action = np.linalg.norm(prefix, axis=-1)
    return {
        "exec": float(per_action.mean()),
        "full": float(np.linalg.norm(difference, axis=-1).mean()),
        "translation": float(np.linalg.norm(prefix[:, 0:3], axis=-1).mean()),
        "rotation": float(np.linalg.norm(prefix[:, 3:6], axis=-1).mean()),
        "gripper": float(np.abs(prefix[:, 6]).mean()),
        "max_exec": float(per_action.max()),
    }


def _metric_cluster_ci(
    outcomes: np.ndarray, predictions: np.ndarray, tasks: list[str], repeats: int, seed: int
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    clusters = sorted(set(tasks))
    indices = {task: np.flatnonzero(np.asarray(tasks, dtype=object) == task) for task in clusters}
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        draw_tasks = rng.integers(0, len(clusters), size=len(clusters))
        draw = np.concatenate([indices[clusters[index]] for index in draw_tasks])
        metrics = prediction_metrics(outcomes[draw], predictions[draw])
        for key in (
            "auroc",
            "auprc",
            "brier_score",
            "expected_calibration_error_5bin",
            "sensitivity_at_fixed_specificity",
        ):
            value = metrics[key]
            if value is not None and np.isfinite(value):
                samples[key].append(float(value))
    return {
        key: ([float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None)
        for key, values in samples.items()
    }


def _group_difference_bootstraps(
    frame: pd.DataFrame, metric: str, repeats: int, seed: int
) -> tuple[list[float], list[float]]:
    success = frame[frame["outcome"] == 1][metric].to_numpy(dtype=np.float64)
    failure = frame[frame["outcome"] == 0][metric].to_numpy(dtype=np.float64)
    if len(success) == 0 or len(failure) == 0:
        raise ValueError("Both outcome groups are required for disagreement comparison")
    rng = np.random.default_rng(seed)
    success_indices = rng.integers(0, len(success), size=(repeats, len(success)))
    failure_indices = rng.integers(0, len(failure), size=(repeats, len(failure)))
    case_differences = success[success_indices].mean(axis=1) - failure[failure_indices].mean(axis=1)

    tasks = sorted(frame["task_id"].unique().tolist())
    by_task = {task: frame[frame["task_id"] == task] for task in tasks}
    cluster_differences = []
    while len(cluster_differences) < repeats:
        selected = rng.integers(0, len(tasks), size=len(tasks))
        draw = pd.concat([by_task[tasks[index]] for index in selected], ignore_index=True)
        draw_success = draw[draw["outcome"] == 1][metric]
        draw_failure = draw[draw["outcome"] == 0][metric]
        if len(draw_success) and len(draw_failure):
            cluster_differences.append(float(draw_success.mean() - draw_failure.mean()))
    return (
        [float(np.quantile(case_differences, 0.025)), float(np.quantile(case_differences, 0.975))],
        [float(np.quantile(cluster_differences, 0.025)), float(np.quantile(cluster_differences, 0.975))],
    )


def analyze_part_b(paths: dict[str, Path], output_root: Path) -> dict[str, Any]:
    dev = load_json(paths["replay_dev_manifest"])
    snap = load_json(paths[f"replay_{SELECTED_STEP}"])
    by_case: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in dev["results"]:
        if result["arm"] == "base10":
            key = (
                result["suite"],
                int(result["task_id"]),
                int(result["init_state_id"]),
                int(result["env_seed"]),
            )
            by_case[key]["base10"] = result
    for result in snap["results"]:
        key = (result["suite"], int(result["task_id"]), int(result["init_state_id"]), int(result["env_seed"]))
        by_case[key][result["arm"]] = result
    records = []
    for key in sorted(by_case):
        arms = by_case[key]
        if set(arms) != {"base10", "snap10", "snap2", "snap1"}:
            raise ValueError(f"Part B arm coverage incomplete for {key}")
        invariant_keys = (
            "initial_sim_state_sha256",
            "canonical_input_sha256",
            "noise_schedule_sha256",
            "evaluator_sha256",
        )
        for invariant in invariant_keys:
            if len({arms[arm].get(invariant) for arm in arms}) != 1:
                raise ValueError(f"Part B invariant mismatch {invariant} for {key}")
        chunks = {}
        traces = {}
        for arm in arms:
            chunks[arm], traces[arm] = _load_initial_chunk(arms[arm])
        if len({row["processed_policy_input_sha256"] for row in traces.values()}) != 1:
            raise ValueError(f"Processed input mismatch at initial replan for {key}")
        if len({canonical_json_sha256(row["observation_hashes"]) for row in traces.values()}) != 1:
            raise ValueError(f"Observation mismatch at initial replan for {key}")
        d10 = _disagreement(chunks["snap1"], chunks["base10"], 10)
        d12 = _disagreement(chunks["snap1"], chunks["snap2"], 10)
        result = arms["snap1"]
        records.append(
            {
                "case_id": result_case_id(result),
                "task_id": task_cluster_id(result),
                "rollout_arm": "20k_snap1",
                "outcome": int(bool(result["success"])),
                "failure_label": result.get("failure_classification", {}).get("category"),
                "replan_index": 0,
                "environment_step": 0,
                "D_1_10_exec": d10["exec"],
                "D_1_2_exec": d12["exec"],
                "D_1_10_full": d10["full"],
                "D_1_2_full": d12["full"],
                "translation_disagreement": d10["translation"],
                "rotation_disagreement": d10["rotation"],
                "gripper_disagreement": d10["gripper"],
                "max_executed_prefix_discrepancy": d10["max_exec"],
                "noise_variance": None,
                "pre_failure_distance_in_replans": None,
                "pre_contact_distance_in_replans": None,
                "pre_gripper_close_distance_in_replans": None,
                "pre_object_lift_distance_in_replans": None,
                "pre_object_drop_distance_in_replans": None,
                "evidence_availability_flags": json.dumps(
                    {
                        "matched_initial_replan": True,
                        "later_raw_observation_snapshots": False,
                        "multiple_noise_queries": False,
                        "contact_timing_for_cross_arm_query": False,
                    },
                    sort_keys=True,
                ),
            }
        )
    frame = pd.DataFrame(records)
    threshold = float(frame["D_1_10_exec"].quantile(0.9))
    frame["large_discrepancy_90pct_unlabeled"] = frame["D_1_10_exec"] >= threshold
    frame.to_csv(output_root / "closed_loop_discrepancy_records.csv", index=False)
    frame.to_parquet(output_root / "closed_loop_discrepancy_records.parquet", index=False)

    successes = frame[frame["outcome"] == 1]
    failures = frame[frame["outcome"] == 0]
    comparison = {}
    for index, metric in enumerate(
        [
            "D_1_10_exec",
            "D_1_2_exec",
            "translation_disagreement",
            "rotation_disagreement",
            "gripper_disagreement",
        ]
    ):
        left = successes[metric].to_numpy()
        right = failures[metric].to_numpy()
        difference_values = left.mean() - right.mean()
        case_ci, cluster_ci = _group_difference_bootstraps(
            frame, metric, BOOTSTRAP_REPEATS, BOOTSTRAP_SEED + index
        )
        comparison[metric] = {
            "success": {
                "mean": float(left.mean()),
                "median": float(np.median(left)),
                "iqr": [float(np.quantile(left, 0.25)), float(np.quantile(left, 0.75))],
            },
            "failure": {
                "mean": float(right.mean()),
                "median": float(np.median(right)),
                "iqr": [float(np.quantile(right, 0.25)), float(np.quantile(right, 0.75))],
            },
            "mean_difference_success_minus_failure": float(difference_values),
            "case_bootstrap_ci95": case_ci,
            "task_cluster_bootstrap_ci95": cluster_ci,
            "standardized_effect_size": standardized_mean_difference(left, right),
        }

    failure = 1 - frame["outcome"].to_numpy(dtype=np.int8)
    tasks = frame["task_id"].tolist()
    feature_sets = {
        "constant_base_rate": None,
        "D_1_10_only": frame[["D_1_10_exec"]].to_numpy(),
        "D_1_2_only": frame[["D_1_2_exec"]].to_numpy(),
        "combined_disagreement": frame[
            [
                "D_1_10_exec",
                "D_1_2_exec",
                "translation_disagreement",
                "rotation_disagreement",
                "gripper_disagreement",
            ]
        ].to_numpy(),
    }
    metrics_output = {
        "schema_version": 1,
        "analysis_type": "diagnostic_noncausal_held_out_task",
        "evidence_scope": "matched initial replan only; later student-visited observations were not stored",
        "unlabeled_large_discrepancy_threshold": threshold,
        "success_failure_comparison": comparison,
        "models": {},
    }
    prediction_rows = []
    for index, (name, features) in enumerate(feature_sets.items()):
        if features is None:
            predictions = np.empty(len(failure), dtype=np.float64)
            task_array = np.asarray(tasks, dtype=object)
            for task in sorted(set(tasks)):
                train = task_array != task
                predictions[~train] = failure[train].mean()
        else:
            predictions = leave_one_task_out_predictions(features, failure, tasks)
        metric = prediction_metrics(failure, predictions)
        metric["task_cluster_bootstrap_ci95"] = _metric_cluster_ci(
            failure, predictions, tasks, 2_000, BOOTSTRAP_SEED + index * 1000
        )
        metrics_output["models"][name] = metric
        for row_index, probability in enumerate(predictions):
            prediction_rows.append(
                {
                    "case_id": frame.iloc[row_index]["case_id"],
                    "task_id": tasks[row_index],
                    "model": name,
                    "failure": int(failure[row_index]),
                    "predicted_failure_probability": float(probability),
                }
            )
    write_json(output_root / "failure_prediction_metrics.json", metrics_output)
    pd.DataFrame(prediction_rows).to_csv(output_root / "failure_prediction_predictions.csv", index=False)
    report = [
        "# Trajectory Divergence Audit",
        "",
        "## VERIFIED",
        "",
        f"All 40 cases provide a strictly matched initial replan across Base-10 and 20k Snap-10/2/1. The outcome-blind 90th-percentile D_1_10 threshold is `{threshold:.6f}`.",
        "",
        "## INFERRED",
        "",
        "Leave-one-task-out diagnostic prediction was computed from initial-replan disagreement only. Any discrimination is predictive association under this protocol, not a causal mechanism.",
        "",
        "## UNVERIFIED",
        "",
        "Later student-visited replans cannot be queried because Task 7/8 traces retain hashes and action chunks but not raw/normalized observation snapshots after the initial replan. Therefore pre-contact, pre-gripper-close, lift/drop timing and multi-noise variance are unavailable and were not inferred from RGB.",
        "",
        "The statement “disagreement causes grasp failure” is not supported.",
    ]
    (output_root / "TRAJECTORY_DIVERGENCE_AUDIT.md").write_text("\n".join(report) + "\n")
    return metrics_output


def _bootstrap_interval_from_counts(
    positive: int, negative: int, zero: int, repeats: int, seed: int
) -> list[float]:
    n = positive + negative + zero
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [positive / n, negative / n, zero / n], size=repeats)
    means = (draws[:, 0] - draws[:, 1]) / n
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _simulate_design_fast(n: int, delta: float, discordance: float, trials: int, seed: int) -> dict[str, Any]:
    p10 = (discordance + delta) / 2
    p01 = (discordance - delta) / 2
    if min(p10, p01) < 0:
        return {"n": n, "true_delta": delta, "discordance_rate": discordance, "status": "INFEASIBLE"}
    rng = np.random.default_rng(seed)
    widths = []
    passes = []
    estimates = []
    for trial in range(trials):
        positive, negative, zero = rng.multinomial(n, [p10, p01, 1 - discordance])
        ci = _bootstrap_interval_from_counts(positive, negative, zero, BOOTSTRAP_REPEATS, seed + trial + 1)
        widths.append(ci[1] - ci[0])
        passes.append(ci[0] >= NONINFERIORITY_MARGIN)
        estimates.append((positive - negative) / n)
    return {
        "n": n,
        "true_delta": delta,
        "discordance_rate": discordance,
        "status": "SIMULATED",
        "trials": trials,
        "paired_bootstrap_repeats": BOOTSTRAP_REPEATS,
        "expected_ci_width": float(np.mean(widths)),
        "median_ci_width": float(np.median(widths)),
        "pass_probability": float(np.mean(passes)),
        "mean_estimated_delta": float(np.mean(estimates)),
    }


def analyze_sample_size(paths: dict[str, Path], output_root: Path, trials: int) -> dict[str, Any]:
    formal = load_json(paths["formal100"])
    formal_cases: dict[tuple, dict[str, int]] = defaultdict(dict)
    for result in formal["results"]:
        key = (result["suite"], int(result["task_id"]), int(result["init_state_id"]), int(result["env_seed"]))
        if result["arm"] in {"base10", "snap1"}:
            formal_cases[key][result["arm"]] = int(bool(result["success"]))
    formal_discordance = np.mean([arms["base10"] != arms["snap1"] for arms in formal_cases.values()])

    dev = load_json(paths["replay_dev_manifest"])
    snap = load_json(paths[f"replay_{SELECTED_STEP}"])
    dev_pairs: dict[tuple, dict[str, int]] = defaultdict(dict)
    for result in dev["results"]:
        if result["arm"] == "base10":
            key = (
                result["suite"],
                int(result["task_id"]),
                int(result["init_state_id"]),
                int(result["env_seed"]),
            )
            dev_pairs[key]["base10"] = int(bool(result["success"]))
    for result in snap["results"]:
        if result["arm"] == "snap1":
            key = (
                result["suite"],
                int(result["task_id"]),
                int(result["init_state_id"]),
                int(result["env_seed"]),
            )
            dev_pairs[key]["snap1"] = int(bool(result["success"]))
    dev_discordance = np.mean([arms["base10"] != arms["snap1"] for arms in dev_pairs.values()])
    pooled = (formal_discordance * len(formal_cases) + dev_discordance * len(dev_pairs)) / (
        len(formal_cases) + len(dev_pairs)
    )
    discordances = sorted(
        {max(0.05, formal_discordance), max(0.05, dev_discordance), max(0.05, pooled), 0.25}
    )
    rows = []
    for n_index, n in enumerate((200, 400, 600, 800)):
        for delta_index, delta in enumerate((0.0, -0.01, -0.025, -0.05)):
            for rate_index, rate in enumerate(discordances):
                rows.append(
                    _simulate_design_fast(
                        n,
                        delta,
                        rate,
                        trials,
                        BOOTSTRAP_SEED + n_index * 100_000 + delta_index * 10_000 + rate_index * 1000,
                    )
                )
    candidate = [row for row in rows if row["n"] == 600 and abs(row["discordance_rate"] - pooled) < 1e-12]
    target_pass_probability = 0.80
    expanded = []
    recommended = 600
    recommendation_status = "FROZEN_DEFAULT_SUPPORTED"
    candidate_pass = candidate[0].get("pass_probability") if candidate else None
    if candidate_pass is None or candidate_pass < target_pass_probability:
        recommendation_status = "FROZEN_DEFAULT_INSUFFICIENT_RECOMMEND_LARGER_NOT_EXECUTED"
        for index, n in enumerate((1_000, 1_200, 1_600, 2_000, 2_400)):
            item = _simulate_design_fast(
                n,
                0.0,
                pooled,
                trials,
                BOOTSTRAP_SEED + 900_000 + index * 10_000,
            )
            item["design_role"] = "expanded_planning_grid_true_delta_zero_only"
            expanded.append(item)
            if item.get("pass_probability", 0) >= target_pass_probability:
                recommended = n
                break
        else:
            recommended = 2_400
    rows.extend(expanded)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "confirmation_sample_size_results.csv", index=False)
    results = {
        "schema_version": 1,
        "procedure": "prospective paired-Bernoulli Monte Carlo; every simulated dataset uses the frozen formal-gate 10,000-repeat percentile paired-case bootstrap interval",
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "formal100_discordance_rate": float(formal_discordance),
        "dev40_20k_discordance_rate": float(dev_discordance),
        "pooled_development_discordance_rate": float(pooled),
        "discordance_sensitivity_values": discordances,
        "outer_trials": trials,
        "planning_target_pass_probability_at_true_delta_zero": target_pass_probability,
        "results": rows,
        "candidate_n600_at_pooled_discordance": candidate,
        "expanded_planning_grid": expanded,
        "recommended_primary_cases": recommended,
        "recommendation_status": recommendation_status,
    }
    write_json(output_root / "confirmation_sample_size_results.json", results)
    shutil.copy2(
        Path(__file__).with_name("confirmation_sample_size_simulation.py"),
        output_root / "confirmation_sample_size_simulation.py",
    )
    report = [
        "# Confirmation Sample Size Analysis",
        "",
        "**Prospective design only; no confirmation outcome was inspected.**",
        "",
        f"- Formal100 discordance: `{formal_discordance:.3f}`.",
        f"- dev40 (Base-10 vs 20k Snap-1) discordance: `{dev_discordance:.3f}`.",
        f"- Pooled planning discordance: `{pooled:.3f}`.",
        f"- Non-inferiority margin remains `{NONINFERIORITY_MARGIN:+.3f}`.",
        "- The original paired percentile bootstrap procedure was reused exactly with 10,000 resamples per simulated dataset.",
        f"- Planning criterion: at least `{target_pass_probability:.0%}` probability of passing when true delta is 0 at the pooled discordance rate.",
        f"- Recommended primary cases: `{recommended}`. The Task 9 manifest remains the frozen 600-case default and is not automatically enlarged.",
    ]
    (output_root / "CONFIRMATION_SAMPLE_SIZE_ANALYSIS.md").write_text("\n".join(report) + "\n")
    return results


def write_protocols(output_root: Path) -> None:
    protocol = """# Disjoint Confirmation Protocol

## Frozen prospective design

- Primary endpoint: `SR(20k Snap-1) - SR(Base-10)`.
- Non-inferiority margin: `-0.03`; neither formal100 nor dev40 may redefine it.
- Primary arms: Base-10 and frozen 20k Snap-1.
- Diagnostic arms: Base-2, Base-1, 20k Snap-10, 20k Snap-2 on the frozen diagnostic subset only.
- Every paired arm shares the exact initial simulator state, raw and normalized observation, condition, proprioception, processor contract, noise tensor/schedule, action normalization, execution horizon, evaluator, and runtime.
- CRP/CAG/KD/new-method training remains HOLD. No confirmation rollout is authorized by Task 9.

## Failure/invalid rerun rule

Only infrastructure failures documented without outcome access may be rerun. The same case, arm, state, noise schedule, evaluator, and output namespace must be retained; both failed attempt metadata and replacement attempt metadata remain immutable. Scientific failures are never rerun.
"""
    blinding = """# Confirmation Blinding and Aggregation

- Before all primary case/arm records are `COMPLETED` and pass integrity checks, operators may view only `COMPLETED / FAILED / INVALID` infrastructure status.
- No live arm success rate, paired difference, aggregate table, early stopping, or threshold revision is permitted.
- The final aggregator rejects missing, duplicate, failed, invalid, diagnostic-only, or overlapping primary records.
- Diagnostic arms are keyed to the frozen diagnostic subset and excluded from the primary aggregator.
- Aggregation becomes available only after the exact primary manifest cardinality is complete and all provenance/overlap/hash gates pass.
"""
    (output_root / "CONFIRMATION_PROTOCOL.md").write_text(protocol)
    (output_root / "CONFIRMATION_BLINDING_AND_AGGREGATION.md").write_text(blinding)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_root = (
        (repo / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    integrity, paths = audit_inputs(repo, output_root)
    if integrity["status"] != "INPUT_INTEGRITY_PASSED":
        print(json.dumps({"status": TASK9_STATUS_BLOCKED, "errors": integrity["errors"]}, sort_keys=True))
        raise SystemExit(2)
    if args.audit_only:
        print(json.dumps({"status": "INPUT_INTEGRITY_PASSED"}))
        return
    part_a = analyze_part_a(paths, output_root)
    part_b = analyze_part_b(paths, output_root)
    sample_size = analyze_sample_size(paths, output_root, args.sample_trials)
    write_protocols(output_root)
    write_json(
        output_root / "task9_analysis_intermediate.json",
        {
            "status": "TASK9_ANALYSIS_COMPLETED",
            "selected_step_frozen": SELECTED_STEP,
            "selected_model_sha256": SELECTED_MODEL_SHA256,
            "part_a_20k_selection_frequency": part_a["robustness"]["20k_case_selection_frequency"],
            "part_b_models": part_b["models"],
            "recommended_primary_cases": sample_size["recommended_primary_cases"],
        },
    )
    print(json.dumps({"status": "TASK9_ANALYSIS_COMPLETED", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
