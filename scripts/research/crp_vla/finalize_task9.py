#!/usr/bin/env python
"""Fail-closed finalizer for the CRP-VLA Task 9 deliverable."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from task9_common import SELECTED_MODEL_SHA256, file_sha256, load_json

READY = "READY_FOR_DISJOINT_CONFIRMATION"
BLOCKED = "BLOCKED_WITH_ADMITTED_FAILURE"

REQUIRED = (
    "INPUT_INTEGRITY_REPORT.md",
    "input_integrity.json",
    "input_sha256_manifest.json",
    "per_case_checkpoint_nfe_matrix.csv",
    "per_case_checkpoint_nfe_matrix.parquet",
    "checkpoint_success_vectors.json",
    "checkpoint_pairwise_flips.json",
    "checkpoint_selection_bootstrap.json",
    "checkpoint_selection_influence.csv",
    "CHECKPOINT_PAIRWISE_ANALYSIS.md",
    "CHECKPOINT_SELECTION_ROBUSTNESS.md",
    "metric_success_relationships.json",
    "NFE_NON_MONOTONICITY_AUDIT.md",
    "closed_loop_discrepancy_records.csv",
    "closed_loop_discrepancy_records.parquet",
    "failure_prediction_metrics.json",
    "failure_prediction_predictions.csv",
    "TRAJECTORY_DIVERGENCE_AUDIT.md",
    "confirmation_sample_size_simulation.py",
    "confirmation_sample_size_results.csv",
    "confirmation_sample_size_results.json",
    "CONFIRMATION_SAMPLE_SIZE_ANALYSIS.md",
    "confirmation_case_manifest.json",
    "confirmation_diagnostic_subset.json",
    "CONFIRMATION_PROTOCOL.md",
    "CONFIRMATION_BLINDING_AND_AGGREGATION.md",
    "CONFIRMATION_PREFLIGHT_REPORT.md",
    "confirmation_preflight.json",
    "confirmation_overlap_audit.json",
    "confirmation_manifest_sha256.json",
)


def command_value(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = args.output_root.resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    integrity = load_json(root / "input_integrity.json") if (root / "input_integrity.json").exists() else {}
    manifest = (
        load_json(root / "confirmation_case_manifest.json")
        if (root / "confirmation_case_manifest.json").exists()
        else {}
    )
    subset = (
        load_json(root / "confirmation_diagnostic_subset.json")
        if (root / "confirmation_diagnostic_subset.json").exists()
        else {}
    )
    overlap = (
        load_json(root / "confirmation_overlap_audit.json")
        if (root / "confirmation_overlap_audit.json").exists()
        else {}
    )
    preflight = (
        load_json(root / "confirmation_preflight.json")
        if (root / "confirmation_preflight.json").exists()
        else {}
    )
    bootstrap = (
        load_json(root / "checkpoint_selection_bootstrap.json")
        if (root / "checkpoint_selection_bootstrap.json").exists()
        else {}
    )
    flips = (
        load_json(root / "checkpoint_pairwise_flips.json")
        if (root / "checkpoint_pairwise_flips.json").exists()
        else {}
    )
    predictions = (
        load_json(root / "failure_prediction_metrics.json")
        if (root / "failure_prediction_metrics.json").exists()
        else {}
    )
    sample = (
        load_json(root / "confirmation_sample_size_results.json")
        if (root / "confirmation_sample_size_results.json").exists()
        else {}
    )
    checks = {
        "required_files_complete": not missing,
        "input_integrity": integrity.get("status") == "INPUT_INTEGRITY_PASSED",
        "selected_model_hash": integrity.get("selected_model_sha256_observed") == SELECTED_MODEL_SHA256,
        "confirmation_manifest_frozen": manifest.get("status") == "CONFIRMATION_CASE_MANIFEST_FROZEN",
        "confirmation_case_count": manifest.get("case_count") == 600,
        "diagnostic_subset_count": subset.get("case_count") == 200,
        "zero_overlap": overlap.get("status") == "ZERO_OVERLAP_VERIFIED"
        and overlap.get("formal100_case_identity_overlap_count") == 0
        and overlap.get("dev40_case_identity_overlap_count") == 0,
        "preflight": preflight.get("status") == "CONFIRMATION_PREFLIGHT_PASSED",
        "formal_confirmation_not_started": preflight.get("formal_confirmation_started") is False,
        "new_training_not_started": preflight.get("new_training_started") is False,
    }
    status = READY if all(checks.values()) else BLOCKED
    base_pair = flips.get("comparisons", {}).get("base10", {})
    combined = predictions.get("models", {}).get("combined_disagreement", {})
    constant = predictions.get("models", {}).get("constant_base_rate", {})
    report = [
        "# CRP-VLA Task 9 Final Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: VERIFIED" if status == READY else "- Verification Status: UNVERIFIED",
        "- Version Label: crp_vla_task9_final_v1",
        "",
        f"**Final status: {status}**",
        "",
        "## VERIFIED",
        "",
        f"- Frozen selected checkpoint remains 20k with model SHA-256 `{SELECTED_MODEL_SHA256}`.",
        f"- 20k case-bootstrap selection frequency: `{bootstrap.get('20k_case_selection_frequency')}`; leave-one-case retention: `{bootstrap.get('leave_one_case_out_20k_retention')}`; leave-one-task retention: `{bootstrap.get('leave_one_task_out_20k_retention')}`.",
        f"- Development Base-10 vs 20k Snap-1 paired difference: `{base_pair.get('paired_success_difference')}`; case CI `{base_pair.get('case_bootstrap_ci95')}`; task CI `{base_pair.get('task_cluster_bootstrap_ci95')}`; exact McNemar p `{base_pair.get('exact_mcnemar_p')}`.",
        f"- Sample-size analysis recommends `{sample.get('recommended_primary_cases')}` primary cases; the required default manifest remains frozen at `{manifest.get('case_count')}` and was not automatically enlarged.",
        f"- The deterministic diagnostic subset contains `{subset.get('case_count')}` cases.",
        f"- Manifest overlap with formal100/dev40: `{overlap.get('formal100_case_identity_overlap_count')}` / `{overlap.get('dev40_case_identity_overlap_count')}`.",
        "- Formal confirmation rollout has not started; no new training has started.",
        "",
        "## INFERRED",
        "",
        f"- Under leave-one-task-out diagnostic prediction, combined disagreement AUROC/AUPRC are `{combined.get('auroc')}` / `{combined.get('auprc')}` versus constant `{constant.get('auroc')}` / `{constant.get('auprc')}`. This is predictive association, not causation.",
        "",
        "## UNVERIFIED",
        "",
        "- Later student-visited replans were not queryable because raw/normalized observation snapshots after the first replan were not retained. Failure-relative contact/gripper/lift/drop timing and multi-noise variance therefore remain unavailable.",
        "- Condition-specific or grounding-specific degradation remains NOT SUPPORTED; the Formal ordinary-LIBERO gate remains FAIL; 20k remains selected for development and not confirmed.",
        "",
        "## Final gate checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in checks.items()],
        *([f"- Missing files: `{missing}`"] if missing else []),
    ]
    (root / "TASK9_FINAL_REPORT.md").write_text("\n".join(report) + "\n")

    pip_freeze = command_value(["uv", "pip", "freeze", "--python", str(repo / ".venv/bin/python")])
    provenance = [
        "# CRP-VLA Task 9 Provenance",
        "",
        f"- Repository revision before Task 9: `{integrity.get('repository_revision_before_task9')}`",
        f"- Current repository revision before Task 9 commit: `{command_value(['git', '-C', str(repo), 'rev-parse', 'HEAD'])}`",
        f"- Branch: `{command_value(['git', '-C', str(repo), 'branch', '--show-current'])}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- GPU/runtime: `{command_value(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'])}`",
        f"- pip/uv freeze SHA-256: `{__import__('hashlib').sha256(pip_freeze.encode()).hexdigest()}`",
        "- Task 6 staged evidence origin revision: `ad689f200415201af145632cc8d1d146dac1f578`",
        "- Confirmation primary rollout executed: `false`",
        "- New training executed: `false`",
        "",
        "## Verification commands",
        "",
        "- `uv run pytest -q <Task 9 plus replay/manifest/hash regression tests>` → 35 passed",
        "- `.venv/bin/ruff check <Task 9 scripts and test>` → passed",
        "- `run_task9_analysis.py --audit-only` → INPUT_INTEGRITY_PASSED",
        "- `run_task9_analysis.py --sample-trials 500` → TASK9_ANALYSIS_COMPLETED",
        "- `build_task9_confirmation_manifest.py` attempt001 → admitted failure before state generation (byte-level processor manifest mismatch); attempt002 uses verified effective state/action contract",
        "- `run_task9_preflight.py` attempt001 → admitted failure before policy inference (remote Hugging Face metadata request reset)",
        "- `run_task9_preflight.py` attempt002 → admitted failure after model loading (incorrect LIBERO root supplied; no action or rollout executed)",
        "- `run_task9_preflight.py` attempt003 → CONFIRMATION_PREFLIGHT_PASSED using the pinned LIBERO checkout and a Task 9-local offline VLM configuration",
        "",
        "## Frozen dependency list",
        "",
        "```text",
        pip_freeze,
        "```",
    ]
    (root / "TASK9_PROVENANCE.md").write_text("\n".join(provenance) + "\n")

    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "TASK9_SHA256SUMS.txt"
    )
    (root / "TASK9_SHA256SUMS.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(root)}\n" for path in members)
    )
    print(json.dumps({"status": status, "checks": checks, "sha256_entries": len(members)}, sort_keys=True))
    if status != READY:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
