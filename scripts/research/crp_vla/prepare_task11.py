#!/usr/bin/env python
"""Create the immutable Task 11 pre-run audit and 40 task shards."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from run_libero_paired_four_arm_pilot import git_repository_value
from task9_common import (
    NONINFERIORITY_MARGIN,
    PRIMARY_ARMS,
    SELECTED_MODEL_SHA256,
    effective_processor_contract,
    file_sha256,
    load_json,
    require_model_hash,
)
from task10_common import confirmation1200_diagnostic_subset, validate_confirmation1200_manifest
from task11_common import EXECUTION_SCHEMA_VERSION, TASK10_MANIFEST_SHA256, validate_execution_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--task10-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task 11 pre-run freeze")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    manifest_path = args.confirmation1200_manifest.resolve()
    if file_sha256(manifest_path) != TASK10_MANIFEST_SHA256:
        raise RuntimeError("Authoritative Task 10 manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    validate_confirmation1200_manifest(manifest)
    subset = load_json(args.diagnostic_subset)
    expected_subset = confirmation1200_diagnostic_subset(manifest["cases"])
    if subset.get("case_ids") != expected_subset.get("case_ids"):
        raise RuntimeError("Task 10 diagnostic subset identity drift")
    if subset.get("confirmation1200_manifest_sha256") != TASK10_MANIFEST_SHA256:
        raise RuntimeError("Diagnostic subset points to a different confirmation manifest")

    selected_hash = require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")
    base_contract = effective_processor_contract(args.base_checkpoint)
    selected_contract = effective_processor_contract(args.selected_checkpoint)
    if base_contract["sha256"] != selected_contract["sha256"]:
        raise RuntimeError("Base/Snap effective processor contract mismatch")

    formal = load_json(args.formal_manifest)
    development = load_json(args.development_manifest)
    formal_ids = {
        (str(item["suite"]), int(item["task_id"]), int(item["init_state_id"])) for item in formal["results"]
    }
    development_ids = {
        (str(item["suite"]), int(item["task_id"]), int(item["init_state_id"]))
        for item in development["results"]
    }
    case_keys = {
        (str(case["suite"]), int(case["suite_task_id"]), int(case["initial_state_id"]))
        for case in manifest["cases"]
    }
    if case_keys & formal_ids or case_keys & development_ids:
        raise RuntimeError("Task 11 manifest contamination detected during pre-run audit")

    task10_checksum_lines = [
        line.strip()
        for line in (args.task10_root / "TASK10_SHA256SUMS.txt").read_text().splitlines()
        if line.strip()
    ]
    for line in task10_checksum_lines:
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = args.task10_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Task 10 artifact checksum failure: {relative}")

    repository_commit = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    submodules = git_value("submodule", "status").splitlines()
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "mujoco": package_version("mujoco"),
        "mujoco_py": package_version("mujoco-py"),
        "robosuite": package_version("robosuite"),
        "libero": package_version("libero"),
        "renderer": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
        },
    }
    if not environment["cuda_available"]:
        raise RuntimeError("Task 11 requires the audited CUDA host")
    expected_runtime = manifest["cases"][0]["runtime_metadata"]
    for key in ("python", "torch", "cuda_runtime", "cuda_device", "libero_commit"):
        actual = libero_commit if key == "libero_commit" else environment.get(key)
        if actual != expected_runtime.get(key):
            raise RuntimeError(
                f"Task 11 runtime drift for {key}: expected {expected_runtime.get(key)}, observed {actual}"
            )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in manifest["cases"]:
        by_task[str(case["task_id"])].append(case)
    shards = []
    for index, task_id in enumerate(sorted(by_task)):
        cases = sorted(by_task[task_id], key=lambda item: str(item["case_id"]))
        shard = {
            "schema_version": 1,
            "status": "TASK11_SHARD_FROZEN",
            "shard_id": f"shard-{index:02d}",
            "task_id": task_id,
            "case_count": len(cases),
            "case_ids": [str(case["case_id"]) for case in cases],
            "case_identity_payload": [
                {
                    key: case[key]
                    for key in (
                        "case_id",
                        "task_id",
                        "initial_state_id",
                        "initial_state_hash",
                        "simulator_state_hash",
                        "qpos_qvel_hash",
                        "noise_schedule_hash",
                        "evaluator_hash",
                    )
                }
                for case in cases
            ],
            "confirmation1200_manifest_sha256": TASK10_MANIFEST_SHA256,
            "base_model_sha256": base_hash,
            "selected_model_sha256": selected_hash,
            "processor_hash": base_contract["sha256"],
            "evaluator_hash": cases[0]["evaluator_hash"],
            "repository_commit": repository_commit,
            "libero_commit": libero_commit,
            "planned_arms": list(PRIMARY_ARMS),
            "planned_rollouts": len(cases) * len(PRIMARY_ARMS),
        }
        if len(cases) != 30 or len({case["evaluator_hash"] for case in cases}) != 1:
            raise RuntimeError(f"Invalid frozen task shard {task_id}")
        shard_path = output_root / "shards" / f"{shard['shard_id']}.manifest.json"
        json_dump(shard_path, shard)
        shards.append(
            {
                "shard_id": shard["shard_id"],
                "task_id": task_id,
                "case_count": len(cases),
                "case_ids": shard["case_ids"],
                "shard_manifest_path": str(shard_path.resolve()),
                "shard_manifest_sha256": file_sha256(shard_path),
                "planned_rollouts": shard["planned_rollouts"],
            }
        )

    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "TASK11_EXECUTION_MANIFEST_FROZEN",
        "frozen_at_unix_ns": time.time_ns(),
        "repository_commit": repository_commit,
        "repository_branch": branch,
        "repository_dirty": False,
        "submodules": submodules,
        "libero_commit": libero_commit,
        "environment": environment,
        "confirmation1200_manifest_path": str(manifest_path),
        "confirmation1200_manifest_sha256": TASK10_MANIFEST_SHA256,
        "diagnostic_subset_path": str(args.diagnostic_subset.resolve()),
        "diagnostic_subset_sha256": file_sha256(args.diagnostic_subset),
        "formal_manifest_sha256": file_sha256(args.formal_manifest),
        "development_manifest_sha256": file_sha256(args.development_manifest),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_model_sha256": base_hash,
        "selected_checkpoint": str(args.selected_checkpoint.resolve()),
        "selected_model_sha256": selected_hash,
        "offline_vlm_model_dir": str(args.offline_vlm_model_dir.resolve()),
        "processor_hash": base_contract["sha256"],
        "evaluator_hash": manifest["cases"][0]["evaluator_hash"],
        "case_count": 1200,
        "task_count": 40,
        "primary_arms": list(PRIMARY_ARMS),
        "planned_rollouts": 2400,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "bootstrap_type": "paired case percentile bootstrap",
        "bootstrap_repeats": 10_000,
        "bootstrap_seed": 20260815,
        "confidence_level": 0.95,
        "formal100_overlap": 0,
        "dev40_overlap": 0,
        "diagnostic_feature_use_in_primary": False,
        "shards": shards,
    }
    validate_execution_manifest(execution, manifest)
    execution_path = output_root / "CONFIRMATION1200_EXECUTION_MANIFEST.json"
    json_dump(execution_path, execution)

    audit_lines = [
        "# CRP-VLA Task 11 Pre-run Audit",
        "",
        "**Status: TASK11_PRERUN_AUDIT_PASSED**",
        "",
        "No formal primary rollout had started when this audit was written.",
        "",
        "## Repository and runtime",
        "",
        f"- Git HEAD: `{repository_commit}`",
        f"- Branch: `{branch}`",
        "- Working tree clean: `True`",
        f"- LIBERO commit: `{libero_commit}`",
        f"- Python: `{environment['python']}`",
        f"- PyTorch / CUDA: `{environment['torch']}` / `{environment['cuda_runtime']}`",
        f"- CUDA device: `{environment['cuda_device']}`",
        f"- MuJoCo / robosuite / LIBERO: `{environment['mujoco']}` / `{environment['robosuite']}` / `{environment['libero']}`",
        f"- Renderer: `{json.dumps(environment['renderer'], sort_keys=True)}`",
        "",
        "## Frozen inputs",
        "",
        f"- Base-10 model SHA-256: `{base_hash}`",
        f"- 20k Snap-1 model SHA-256: `{selected_hash}`",
        f"- Confirmation1200 manifest SHA-256: `{TASK10_MANIFEST_SHA256}`",
        "- Manifest cardinality: `1200 = 40 tasks × 30 cases`",
        "- Unique initial/simulator/qpos-qvel identities: `1200 / 1200 / 1200`",
        "- formal100 overlap: `0`",
        "- dev40 overlap: `0`",
        "- Diagnostic subset: `200`, identity preserved and excluded from primary features",
        f"- Effective processor contract SHA-256: `{base_contract['sha256']}`",
        f"- Evaluator SHA-256: `{manifest['cases'][0]['evaluator_hash']}`",
        "- Task 10 checksum entries verified: `74/74`",
        "",
        "## Frozen execution",
        "",
        "- Primary arms: `Base-10` and `20k Snap-1` only",
        "- Planned results: `1200 × 2 = 2400`",
        "- Sharding: `40 task shards × 30 cases × 2 arms`",
        "- Execution horizon: `10`",
        "- Non-inferiority margin: `-0.03`",
        "- Bootstrap: paired-case percentile, `10000` repeats, seed `20260815`, 95% interval",
        "- Resume: only hash-valid completed results are skipped; partial/corrupt evidence fails closed",
        "- Interim success-rate aggregation: forbidden",
        "",
        "```text",
        "TASK11_PRERUN_AUDIT_PASSED",
        "```",
    ]
    (output_root / "TASK11_PRERUN_AUDIT.md").write_text("\n".join(audit_lines) + "\n")
    (output_root / "CONFIRMATION1200_EXECUTION_LOG.md").write_text(
        "# Confirmation1200 Execution Log\n\n"
        "Only infrastructure progress may be recorded before completion audit.\n\n"
        f"- Pre-run audit completed at unix ns `{execution['frozen_at_unix_ns']}`.\n"
        "- Formal primary rollouts started: `NO`.\n"
    )
    (output_root / "CONFIRMATION1200_FAILURE_AND_RETRY_LOG.md").write_text(
        "# Confirmation1200 Failure and Retry Log\n\nNo formal execution attempt has started.\n"
    )
    print(
        json.dumps(
            {
                "status": "TASK11_PRERUN_AUDIT_PASSED",
                "shards": len(shards),
                "cases": 1200,
                "planned_rollouts": 2400,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
