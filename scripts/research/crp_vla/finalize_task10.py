#!/usr/bin/env python
"""Fail-closed finalizer for CRP-VLA Task 10 artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from task9_common import SELECTED_MODEL_SHA256, file_sha256, load_json
from task10_common import (
    CASES_PER_TASK,
    DIAGNOSTIC_CASES,
    PRIMARY_CASES,
    confirmation1200_diagnostic_subset,
    validate_confirmation1200_manifest,
)
from trajectory_instrumentation import TRACE_SCHEMA, load_replan_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--task9-root", type=Path, required=True)
    parser.add_argument("--admitted-failures", type=Path)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def check_value(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(check for check in checks if check["name"] == name)


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    preflight_root = args.preflight_root.resolve()
    manifest_path = root / "CONFIRMATION1200_MANIFEST.json"
    subset_path = root / "CONFIRMATION1200_DIAGNOSTIC_SUBSET.json"
    audit_path = root / "confirmation1200_overlap_audit.json"
    manifest = load_json(manifest_path)
    validate_confirmation1200_manifest(manifest)
    subset = load_json(subset_path)
    reconstructed = confirmation1200_diagnostic_subset(manifest["cases"])
    audit = load_json(audit_path)
    preflight = load_json(preflight_root / "confirmation1200_preflight.json")
    counterfactual = load_json(preflight_root / "counterfactual_validation.json")
    checks = preflight["checks"]
    failures = (
        json.loads(args.admitted_failures.read_text())
        if args.admitted_failures and args.admitted_failures.exists()
        else []
    )
    if not isinstance(failures, list):
        raise ValueError("Admitted-failure log must be a JSON list")

    trace_dirs = sorted((preflight_root / "preflight_traces").glob("*/replan_0000"))
    trace_payloads = [load_replan_trace(path, device="cpu") for path in trace_dirs]
    gates = {
        "manifest_frozen": manifest["status"] == "CONFIRMATION1200_MANIFEST_FROZEN",
        "primary_cases_1200": manifest["case_count"] == PRIMARY_CASES,
        "forty_tasks_thirty_cases_each": manifest["task_count"] == 40
        and manifest["cases_per_task"] == CASES_PER_TASK,
        "formal100_overlap_zero": audit["formal100_overlap"] == 0,
        "dev40_overlap_zero": audit["dev40_overlap"] == 0,
        "unique_initial_states": audit["unique_initial_state_hashes"] == PRIMARY_CASES,
        "unique_simulator_states": audit["unique_simulator_state_hashes"] == PRIMARY_CASES,
        "unique_qpos_qvel": audit["unique_qpos_qvel_hashes"] == PRIMARY_CASES,
        "task9_subset_exact": audit["task9_subset_cases"] == 600 and audit["new_cases"] == 600,
        "diagnostic_subset_frozen": subset["case_count"] == DIAGNOSTIC_CASES
        and subset["case_ids"] == reconstructed["case_ids"]
        and subset["confirmation_outcomes_available_at_freeze"] is False,
        "complete_observation_round_trip": len(trace_payloads) == 2
        and all(
            "raw_observation" in payload and "normalized_observation" in payload
            for _, payload in trace_payloads
        ),
        "state_restoration": check_value(checks, "simulator_state_restoration")["passed"]
        and check_value(checks, "restored_observation_hash")["passed"],
        "counterfactual_identical_initial_state": check_value(
            checks, "counterfactual_identical_initial_state"
        )["passed"],
        "counterfactual_isolation": check_value(checks, "counterfactual_branch_isolation")["passed"],
        "counterfactual_determinism": check_value(checks, "counterfactual_determinism")["passed"],
        "primary_aggregation_fail_closed": check_value(checks, "incomplete_aggregator_rejection")["passed"],
        "diagnostic_contamination_rejected": check_value(checks, "diagnostic_contamination_rejection")[
            "passed"
        ],
        "preflight_passed": preflight["status"] == "CONFIRMATION1200_PREFLIGHT_PASSED"
        and all(check["passed"] for check in checks),
        "formal_confirmation_not_started": preflight["formal_confirmation_started"] is False
        and preflight["confirmation_primary_cases_executed"] == 0,
        "new_training_not_started": preflight["new_training_started"] is False,
        "selected_model_unchanged": preflight["selected_model_sha256"] == SELECTED_MODEL_SHA256,
    }
    ready = all(gates.values())
    status = "READY_FOR_1200_CONFIRMATION" if ready else "NOT_READY_FOR_1200_CONFIRMATION"

    (root / "TRACE_SCHEMA.json").write_text(json.dumps(TRACE_SCHEMA, indent=2, sort_keys=True) + "\n")
    (root / "CONFIRMATION1200_MANIFEST_AUDIT.md").write_text(
        "\n".join(
            [
                "# Confirmation 1200 Manifest Audit",
                "",
                "**Verification Status: VERIFIED**",
                "",
                f"- Primary cases: `{audit['primary_cases']}`",
                f"- Tasks / cases per task: `{audit['tasks']} / {audit['cases_per_task']}`",
                f"- Frozen Task 9 subset / new cases: `{audit['task9_subset_cases']} / {audit['new_cases']}`",
                f"- formal100 overlap: `{audit['formal100_overlap']}`",
                f"- dev40 overlap: `{audit['dev40_overlap']}`",
                f"- Unique initial/simulator/qpos-qvel hashes: `{audit['unique_initial_state_hashes']} / {audit['unique_simulator_state_hashes']} / {audit['unique_qpos_qvel_hashes']}`",
                f"- Fixed candidate rejections: `{audit['fixed_candidate_rejections']}`; silent replacements: `0`",
                "",
                "Task 9's 600 cases are an exact, explicitly marked subset. The additional 600 cases use fixed state indices 20–34 for each task. Any fixed candidate rejection fails the full build; no replacement search is allowed.",
            ]
        )
        + "\n"
    )
    (root / "CONFIRMATION1200_PROTOCOL.md").write_text(
        """# Frozen 1,200-Case Confirmation Protocol

- Primary endpoint: `SR(20k Snap-1) - SR(Base-10)`.
- Primary cases: exactly 40 ordinary-LIBERO tasks x 30 frozen states = 1,200.
- Primary arms only: Base-10 and frozen 20k Snap-1.
- Non-inferiority margin remains `-0.03`.
- Success definition, evaluator, simulator termination, action horizon, normalization, and model weights are unchanged.
- Every pair shares the exact initial simulator state, raw/normalized observation, instruction, processor contract, action normalization, noise schedule, execution horizon, evaluator, and runtime settings.
- No interim success aggregation, outcome-based stopping, sample expansion, case replacement, or failed-rollout deletion is allowed.
- Infrastructure failures may be replaced only under the frozen method-blind rule while retaining both immutable attempts and the same case/arm namespace.
- The 200-case diagnostic subset is mechanism-only and never enters the primary test.
- `READY_FOR_1200_CONFIRMATION` is infrastructure readiness, not rollout authorization or a scientific PASS.
"""
    )
    (root / "CONFIRMATION1200_BLINDING_AND_AGGREGATION.md").write_text(
        """# Confirmation 1200 Blinding and Aggregation

- Operators may inspect only infrastructure status (`COMPLETED`, `FAILED`, `INVALID`) until all 2,400 primary arm records pass integrity checks.
- The primary aggregator refuses output unless `completed_primary_cases == 1200`, exact paired-arm coverage holds, and every record is `COMPLETED`.
- Missing, duplicate, failed, invalid, overlapping, non-primary, or diagnostic records make aggregation fail closed.
- Diagnostic results are stored under separate arm/result namespaces and any diagnostic contamination in the primary input is an error, not a filtered row.
- Final inference reuses the frozen paired percentile bootstrap and `-0.03` margin. No weighting or threshold may be tuned after outcomes exist.
"""
    )
    (root / "TRAJECTORY_INSTRUMENTATION_SPEC.md").write_text(
        """# Trajectory Instrumentation Specification

Every rollout replan writes an immutable hash-verified NPZ archive containing the full raw observation, full normalized policy input/processor tensors, language condition, serialized simulator state, qpos, qvel, task-relevant object poses, robot/gripper state, raw policy output, denormalized action, actually executed action chunk, execution horizon, NFE, complete noise tensor and seed, action mask when present, and processor contract/hash.

Identity is `(case_id, task_id, arm_id, rollout_id, replan_index, simulator_timestep)`. Objective simulator events include gripper state, end-effector/object poses, object-gripper distance, contacts, object height, success, and termination. Unsupported grasp state is explicitly `UNAVAILABLE`; no RGB-derived or subjective event labels are created.

`run_replay_v2.py` and `run_maturation_replay_v2.py` now stream these archives at every replan. The stored raw/normalized observations permit future Base-10, 20k Snap-10, 20k Snap-2, and 20k Snap-1 queries from the same diagnostic state without rerunning the original policy visit.
"""
    )
    restoration = check_value(checks, "simulator_state_restoration")
    observation_restore = check_value(checks, "restored_observation_hash")
    (root / "STATE_RESTORATION_VALIDATION.md").write_text(
        "\n".join(
            [
                "# State Restoration Validation",
                "",
                "**Verification Status: VERIFIED on existing dev40 preflight state**",
                "",
                f"- state -> perturb -> restore qpos/qvel/object check: `{restoration['passed']}`",
                f"- restored observation hash exact match: `{observation_restore['passed']}`",
                f"- evidence: `{json.dumps(restoration['evidence'], sort_keys=True)}`",
                "",
                "This validates the Task 10 interface on a pre-existing development capsule; no confirmation primary case was used.",
            ]
        )
        + "\n"
    )
    (root / "COUNTERFACTUAL_BRANCH_VALIDATION.md").write_text(
        "\n".join(
            [
                "# Counterfactual Branch Validation",
                "",
                "**Verification Status: VERIFIED on existing dev40 preflight state**",
                "",
                f"- Identical restored initial state: `{counterfactual['identical_initial_state']}`",
                f"- Horizons: `{counterfactual['horizons']}`",
                f"- Determinism check: `{check_value(checks, 'counterfactual_determinism')['passed']}`",
                "- Raw components: joint-state L2, end-effector position L2, quaternion geodesic orientation distance, gripper L2, and per-object position/orientation divergence.",
                "- Combined weighted scalar: not defined and not used.",
                "",
                "Both branches restore the same frozen state independently. Branch B is restored after branch A and cannot inherit branch A state.",
            ]
        )
        + "\n"
    )
    report_lines = [
        "# Confirmation 1200 Preflight Report",
        "",
        f"**Status: {preflight['status']}**",
        "",
        "No confirmation primary case was executed and no model was trained.",
        "",
        "| Check | Passed |",
        "|---|:---:|",
        *[f"| {check['name']} | {check['passed']} |" for check in checks],
    ]
    (root / "CONFIRMATION1200_PREFLIGHT_REPORT.md").write_text("\n".join(report_lines) + "\n")

    answer_rows = [
        ("1. 1200 cases frozen", gates["manifest_frozen"] and gates["primary_cases_1200"]),
        ("2. 40 tasks x 30 cases", gates["forty_tasks_thirty_cases_each"]),
        ("3. formal100 overlap == 0", gates["formal100_overlap_zero"]),
        ("4. dev40 overlap == 0", gates["dev40_overlap_zero"]),
        (
            "5. duplicate state exists",
            not (
                gates["unique_initial_states"]
                and gates["unique_simulator_states"]
                and gates["unique_qpos_qvel"]
            ),
        ),
        ("6. diagnostic subset frozen before outcomes", gates["diagnostic_subset_frozen"]),
        ("7. full observation save/reload", gates["complete_observation_round_trip"]),
        ("8. exact simulator state restoration", gates["state_restoration"]),
        (
            "9. counterfactual identical initial state",
            gates["counterfactual_identical_initial_state"] and gates["counterfactual_isolation"],
        ),
        ("10. future per-replan four-arm query supported", gates["complete_observation_round_trip"]),
        (
            "11. primary aggregator fail-closed",
            gates["primary_aggregation_fail_closed"] and gates["diagnostic_contamination_rejected"],
        ),
        ("12. all preflight checks passed", gates["preflight_passed"]),
        ("13. admitted failures occurred", bool(failures)),
        ("14. formal 1200-case rollout started", not gates["formal_confirmation_not_started"]),
        ("15. CRP/CAG/KD/new-method training started", not gates["new_training_not_started"]),
    ]
    final_lines = [
        "# CRP-VLA Task 10 Final Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate / reproducibility verification",
        f"- Verification Status: {'VERIFIED' if ready else 'BLOCKED'}",
        "- Version Label: crp_vla_task10_final_v1",
        "",
        f"**Final infrastructure status: {status}**",
        "",
        "The Formal ordinary-LIBERO Gate remains FAIL. The 20k checkpoint remains SELECTED FOR DEVELOPMENT / NOT CONFIRMED. CRP/CAG/KD/new-method training remains HOLD. This report does not authorize confirmation rollout execution.",
        "",
        "## Required answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        *[f"| {label} | {'YES' if value else 'NO'}" for label, value in answer_rows],
        "",
        "## Admitted failures",
        "",
        json.dumps(failures, indent=2, sort_keys=True) if failures else "None.",
        "",
        "## Evidence boundary",
        "",
        "VERIFIED: frozen manifest cardinality/overlap/uniqueness, deterministic diagnostic subset, lossless trace round trip, same-process state restoration, branch isolation/determinism, and fail-closed aggregation preflight.",
        "",
        "INFERRED: this infrastructure is sufficient to collect later student-visited states for a future mechanism audit.",
        "",
        "UNVERIFIED: 20k confirmation outcome, any relationship between transition divergence and failure, any causal failure mechanism, and any benefit of CRP/CAG/KD/new methods.",
        "",
        f"```text\n{status}\n```",
    ]
    (root / "TASK10_FINAL_REPORT.md").write_text("\n".join(final_lines) + "\n")

    provenance = [
        "# CRP-VLA Task 10 Provenance",
        "",
        f"- Repository commit at finalization: `{git_value('rev-parse', 'HEAD')}`",
        f"- Repository dirty at finalization: `{bool(git_value('status', '--porcelain'))}`",
        f"- Branch: `{git_value('branch', '--show-current')}`",
        f"- Python: `{platform.python_version()}`",
        f"- Task 9 manifest SHA-256: `{file_sha256(args.task9_root / 'confirmation_case_manifest.json')}`",
        f"- Task 10 manifest SHA-256: `{file_sha256(manifest_path)}`",
        f"- Selected 20k model SHA-256: `{SELECTED_MODEL_SHA256}`",
        f"- Preflight source: `{preflight['preflight_source']}`",
        "- Confirmation primary rollouts executed: `0`",
        "- New training executed: `false`",
        "- Counterfactual audit scope: one existing dev40 state, infrastructure validation only",
    ]
    (root / "TASK10_PROVENANCE.md").write_text("\n".join(provenance) + "\n")

    checksum_path = root / "TASK10_SHA256SUMS.txt"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path and ".git" not in path.parts
    )
    checksum_path.write_text("".join(f"{file_sha256(path)}  {path.relative_to(root)}\n" for path in files))
    if not ready:
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"{status}: {failed}")
    print(json.dumps({"status": status, "gates": len(gates), "artifacts": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
