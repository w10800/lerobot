#!/usr/bin/env python
"""Run the policy-free Task14R R0 complete-state transaction gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import load_capsule, structured_hash, write_capsule
from run_libero_paired_four_arm_pilot import configure_standard_libero, git_repository_value
from task14r_reset_transaction import (
    TASK14R_R0_EVIDENCE_LABELS,
    TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
    TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT,
    TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT,
    TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0_LIBERO_COMMIT,
    TASK14R_R0_PROBE_ACTIONS,
    TASK14R_R0_RENDERER_BACKEND,
    TASK14R_R0_RENDERER_OFFSAMPLES,
    TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY,
    TASK14R_R0_RESTORE_REPEATS,
    TASK14R_R0_SOURCE_FILES,
    TASK14R_R0_STATUS_FAILED,
    TASK14R_R0_STATUS_PASSED,
    Task14RStageError,
    build_complete_state_capsule,
    capture_probe_step,
    compare_probe_repeats,
    configure_deterministic_renderer,
    mujoco_model_fingerprint,
    restore_complete_state_transaction,
    task14r_r0_terminal_summary,
    validate_task14r_r0_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_files_sha256(repository_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    hashes = {}
    for relative in TASK14R_R0_SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Task14R R0 source file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def validate_pre_simulator_gate(
    protocol: dict[str, Any], repository_root: Path = REPOSITORY_ROOT
) -> dict[str, str]:
    """Validate frozen code/protocol provenance before creating any simulator output."""
    hashes = source_files_sha256(repository_root)
    validate_task14r_r0_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=hashes,
    )
    return hashes


def write_probe_trace(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text,
    ):
        for row in rows:
            text.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "step_count": len(rows)}


def _nvidia_provenance() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip().splitlines()[0]
        gpu_name, driver_version = (part.strip() for part in line.split(",", maxsplit=1))
        return {"gpu_name": gpu_name, "nvidia_driver_version": driver_version}
    except Exception as error:
        return {
            "gpu_name": None,
            "nvidia_driver_version": None,
            "nvidia_query_error": f"{type(error).__name__}: {error}",
        }


def renderer_provenance(*, mujoco_version: str, robosuite_version: str) -> dict[str, Any]:
    return {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **_nvidia_provenance(),
        "mujoco_version": str(mujoco_version),
        "robosuite_version": str(robosuite_version),
        "egl_vendor": None,
        "egl_version": None,
        "egl_query_status": "PENDING_CURRENT_CONTEXT",
        "offsamples": TASK14R_R0_RENDERER_OFFSAMPLES,
    }


def query_current_egl_provenance() -> dict[str, Any]:
    try:
        from OpenGL import EGL

        display = EGL.eglGetCurrentDisplay()
        vendor = EGL.eglQueryString(display, EGL.EGL_VENDOR)
        version = EGL.eglQueryString(display, EGL.EGL_VERSION)
        return {
            "egl_vendor": None if vendor is None else vendor.decode(errors="replace"),
            "egl_version": None if version is None else version.decode(errors="replace"),
            "egl_query_status": "AVAILABLE" if vendor is not None or version is not None else "UNAVAILABLE",
        }
    except Exception as error:
        return {
            "egl_vendor": None,
            "egl_version": None,
            "egl_query_status": f"UNAVAILABLE:{type(error).__name__}:{error}",
        }


def failure_audit(
    *,
    case: dict[str, Any],
    error: Exception,
    failure_stage: str,
    restore_checks: list[dict[str, Any]] | None = None,
    model_checks: list[dict[str, Any]] | None = None,
    trace_manifests: list[dict[str, Any]] | None = None,
    completed_probe_step_count: int = 0,
    renderer_contract: dict[str, Any] | None = None,
    egl_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    restore_checks = [] if restore_checks is None else restore_checks
    model_checks = [] if model_checks is None else model_checks
    trace_manifests = [] if trace_manifests is None else trace_manifests
    return {
        "schema_version": "task14r.r0.task_failure_audit.v2",
        "case": case,
        "evidence_labels": list(TASK14R_R0_EVIDENCE_LABELS),
        "passed": False,
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "completed_restore_transaction_count": len(restore_checks),
        "completed_probe_trajectory_count": sum(
            int(row.get("step_count", 0)) == len(TASK14R_R0_PROBE_ACTIONS) for row in trace_manifests
        ),
        "completed_probe_step_count": int(completed_probe_step_count),
        "partial_restore_checks": restore_checks,
        "partial_model_checks": model_checks,
        "partial_probe_trace_manifests": trace_manifests,
        "renderer_contract": renderer_contract,
        "egl_provenance": egl_provenance,
        "restore_transaction_count": len(restore_checks),
        "probe_trajectory_count": sum(
            int(row.get("step_count", 0)) == len(TASK14R_R0_PROBE_ACTIONS) for row in trace_manifests
        ),
        "probe_step_count": int(completed_probe_step_count),
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
    }


def task_slug(case: dict[str, Any]) -> str:
    return str(case["case_id"])


def make_environment(suite: Any, suite_name: str, task_id: int) -> Any:
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv

    config = LiberoEnvConfig(
        task=suite_name,
        task_ids=[task_id],
        observation_height=256,
        observation_width=256,
    )
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_length=1000,
        camera_name=config.camera_name,
        obs_type=config.obs_type,
        render_mode=config.render_mode,
        observation_width=config.observation_width,
        observation_height=config.observation_height,
        init_states=config.init_states,
        episode_index=0,
        n_envs=1,
        num_steps_wait=10,
        camera_name_mapping=config.camera_name_mapping,
        control_freq=config.fps,
        control_mode=config.control_mode,
        is_libero_plus=config.is_libero_plus,
        hard_reset=True,
    )


def run_task(env: Any, case: dict[str, Any], task_root: Path) -> dict[str, Any]:
    restore_checks: list[dict[str, Any]] = []
    probe_repeats: list[list[dict[str, Any]]] = []
    model_checks: list[dict[str, Any]] = []
    trace_manifests: list[dict[str, Any]] = []
    completed_probe_step_count = 0
    renderer_contract: dict[str, Any] | None = None
    egl_provenance: dict[str, Any] | None = None
    failure_stage = "capsule_creation"
    try:
        # This is the only outer reset in the task. _ensure_env() performs the
        # fixed initial construction; no reset method is called after capture.
        env.init_state_id = int(case["init_state_id"])
        env.reset(seed=int(case["env_seed"]))
        renderer_contract = configure_deterministic_renderer(env)
        egl_provenance = query_current_egl_provenance()
        capsule = build_complete_state_capsule(env, case=case)
        if capsule["initial_success_predicate"]:
            raise Task14RStageError(
                "capsule_creation", f"R0 canonical state is initially successful: {case['case_id']}"
            )

        failure_stage = "capsule_round_trip"
        capsule_path = task_root / "complete_state_capsule"
        capsule_record = write_capsule(
            capsule_path,
            {
                "schema_version": "task14r.r0.capsule_record.v2",
                "record_type": "TASK14R_R0_COMPLETE_STATE_CAPSULE",
                "case": case,
                "evidence_labels": list(TASK14R_R0_EVIDENCE_LABELS),
                "policy_query_count": 0,
                "formal_outcome_rollout_count": 0,
                "training_or_parameter_updates": False,
            },
            capsule,
        )
        _, loaded_capsule = load_capsule(capsule_path, device="cpu")
        if structured_hash(loaded_capsule) != structured_hash(capsule):
            raise Task14RStageError(
                "capsule_round_trip", f"Complete-state capsule round trip failed: {case['case_id']}"
            )

        for repeat_index in range(TASK14R_R0_RESTORE_REPEATS):
            failure_stage = "complete_state_restore"
            observation, checks = restore_complete_state_transaction(env, loaded_capsule)
            restore_checks.append(
                {
                    "repeat_index": repeat_index,
                    **checks,
                    "observation_sha256": structured_hash(observation),
                }
            )
            failure_stage = "probe_trajectory"
            rows: list[dict[str, Any]] = []
            trajectory_error: Exception | None = None
            try:
                for step, action in enumerate(TASK14R_R0_PROBE_ACTIONS):
                    observation, _, terminated, truncated, info = env.step(
                        np.asarray(action, dtype=np.float32)
                    )
                    rows.append(
                        capture_probe_step(
                            env,
                            observation,
                            repeat_index=repeat_index,
                            step=step,
                            action=action,
                            terminated=terminated,
                            truncated=truncated,
                            success=bool(info.get("is_success", False)),
                        )
                    )
                    completed_probe_step_count += 1
                    if terminated or truncated:
                        break
            except Exception as error:
                trajectory_error = error
            finally:
                trace_manifests.append(
                    {
                        "repeat_index": repeat_index,
                        **write_probe_trace(
                            task_root / "probe_traces" / f"repeat{repeat_index:02d}.steps.jsonl.gz",
                            rows,
                        ),
                    }
                )
                probe_repeats.append(rows)
            if trajectory_error is not None:
                raise Task14RStageError("probe_trajectory", str(trajectory_error)) from trajectory_error
            if len(rows) != len(TASK14R_R0_PROBE_ACTIONS):
                raise Task14RStageError(
                    "probe_trajectory",
                    f"Probe terminated early at {len(rows)}/{len(TASK14R_R0_PROBE_ACTIONS)} steps",
                )
            if any(bool(row["success_predicate"]) for row in rows):
                raise Task14RStageError("probe_trajectory", "Probe reached the task success predicate")

            failure_stage = "model_identity"
            model = mujoco_model_fingerprint(env._env.env.sim)
            model_check = {
                "repeat_index": repeat_index,
                "complete_sha256": model["complete_sha256"],
                "identity": model["complete_sha256"] == loaded_capsule["model"]["complete_sha256"],
            }
            model_checks.append(model_check)
            if not model_check["identity"]:
                raise Task14RStageError("model_identity", "Compiled MuJoCo model changed during probe")

        failure_stage = "exact_repeat_comparison"
        comparison = compare_probe_repeats(probe_repeats)
        if not comparison["passed"]:
            raise Task14RStageError(
                "exact_repeat_comparison", f"Exact repeat mismatch: {json.dumps(comparison, sort_keys=True)}"
            )
        return {
            "schema_version": "task14r.r0.task_audit.v2",
            "case": case,
            "evidence_labels": list(TASK14R_R0_EVIDENCE_LABELS),
            "passed": True,
            "capsule": {"path": capsule_record["path"], "sha256": capsule_record["sha256"]},
            "canonical_model_complete_sha256": loaded_capsule["model"]["complete_sha256"],
            "canonical_model_mjb_sha256": loaded_capsule["model"]["mjb_sha256"],
            "canonical_integration_state_sha256": loaded_capsule["integration"]["state_sha256"],
            "canonical_observation_sha256": loaded_capsule["canonical_observation_sha256"],
            "renderer_contract": renderer_contract,
            "egl_provenance": egl_provenance,
            "restore_transaction_count": len(restore_checks),
            "restore_checks": restore_checks,
            "probe_trajectory_count": len(probe_repeats),
            "probe_step_count": completed_probe_step_count,
            "probe_step_counts": [len(rows) for rows in probe_repeats],
            "probe_trace_manifests": trace_manifests,
            "probe_repeat_comparison": comparison,
            "model_checks": model_checks,
            "probe_completed_full_length": True,
            "probe_success_predicate_never_true": True,
            "outer_reset_count_after_capsule": 0,
            "hard_model_rebuild_count_after_capsule": 0,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
    except Exception as error:
        stage = str(getattr(error, "failure_stage", failure_stage))
        return failure_audit(
            case=case,
            error=error,
            failure_stage=stage,
            restore_checks=restore_checks,
            model_checks=model_checks,
            trace_manifests=trace_manifests,
            completed_probe_step_count=completed_probe_step_count,
            renderer_contract=renderer_contract,
            egl_provenance=egl_provenance,
        )


def execute_task_with_audit(
    *,
    case: dict[str, Any],
    task_root: Path,
    environment_factory: Callable[[], Any],
    task_runner: Callable[[Any, dict[str, Any], Path], dict[str, Any]] = run_task,
) -> dict[str, Any]:
    """Run one task and guarantee a task-level success or failure artifact."""
    env = None
    audit: dict[str, Any] | None = None
    try:
        env = environment_factory()
        audit = task_runner(env, case, task_root)
    except Exception as error:
        stage = str(getattr(error, "failure_stage", "environment_creation"))
        audit = failure_audit(case=case, error=error, failure_stage=stage)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if audit is None or bool(audit.get("passed")):
                    audit = failure_audit(
                        case=case,
                        error=error,
                        failure_stage="environment_close",
                        restore_checks=[] if audit is None else list(audit.get("restore_checks", [])),
                        model_checks=[] if audit is None else list(audit.get("model_checks", [])),
                        trace_manifests=[] if audit is None else list(audit.get("probe_trace_manifests", [])),
                        completed_probe_step_count=0
                        if audit is None
                        else int(audit.get("probe_step_count", 0)),
                    )
    if audit is None:
        raise AssertionError("Task14R R0 task audit was not constructed")
    filename = "TASK_AUDIT.json" if bool(audit.get("passed")) else "TASK_AUDIT_FAILURE.json"
    write_json(task_root / filename, audit)
    return audit


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task14R R0")
    protocol = json.loads(args.protocol.read_text())
    verified_source_hashes = validate_pre_simulator_gate(protocol)
    if git_value("rev-parse", "HEAD^") != TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT:
        raise RuntimeError("Task14R R0 implementation parent commit drift")
    if os.environ.get("MUJOCO_GL", "").lower() != TASK14R_R0_RENDERER_BACKEND:
        raise RuntimeError("Task14R R0 requires MUJOCO_GL=egl")
    protocol_sha256 = file_sha256(args.protocol)
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    if libero_commit != TASK14R_R0_LIBERO_COMMIT or libero_commit != protocol["libero_commit"]:
        raise RuntimeError("Pinned LIBERO commit drift")

    args.output_root.mkdir(parents=True, exist_ok=False)
    configure_standard_libero(args.libero_root, args.output_root / "libero_standard_config")

    import libero.libero as libero_module
    import mujoco
    import robosuite
    from libero.libero import benchmark

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    repository_commit = git_value("rev-parse", "HEAD")
    runtime_renderer_provenance = renderer_provenance(
        mujoco_version=mujoco.__version__, robosuite_version=robosuite.__version__
    )
    run_metadata = {
        "schema_version": "task14r.r0.run_metadata.v2",
        "status": "TASK14R_R0_RUNNING",
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "implementation_parent_commit": TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        "source_files_sha256": verified_source_hashes,
        "libero_commit": libero_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "renderer_contract": protocol["renderer_contract"],
        "renderer_provenance": runtime_renderer_provenance,
        "renderer_qualification_boundary": list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY),
        "policy_query_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
    }
    write_json(args.output_root / "RUN_METADATA.json", run_metadata)

    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in sorted({row["suite"] for row in protocol["tasks"]})}
    task_audits = []
    current_case: dict[str, Any] | None = None
    try:
        for current_case in protocol["tasks"]:
            suite_name = str(current_case["suite"])
            task_id = int(current_case["task_id"])
            task_root = args.output_root / "tasks" / task_slug(current_case)
            audit = execute_task_with_audit(
                case=current_case,
                task_root=task_root,
                environment_factory=lambda suite_name=suite_name, task_id=task_id: make_environment(
                    suites[suite_name], suite_name, task_id
                ),
            )
            task_audits.append(audit)
            if audit.get("egl_provenance"):
                runtime_renderer_provenance.update(audit["egl_provenance"])
            partial = {
                **task14r_r0_terminal_summary(task_audits),
                "status": "TASK14R_R0_RUNNING_PARTIAL",
                "repository_commit": repository_commit,
                "protocol_sha256": protocol_sha256,
                "source_files_sha256": verified_source_hashes,
                "renderer_contract": protocol["renderer_contract"],
                "renderer_provenance": runtime_renderer_provenance,
            }
            write_json(args.output_root / "TASK14R_R0_PARTIAL.json", partial)
            if not audit["passed"]:
                raise RuntimeError(f"Task14R R0 task gate failed: {current_case['case_id']}")

        summary = {
            **task14r_r0_terminal_summary(task_audits),
            "repository_commit": repository_commit,
            "repository_dirty": False,
            "protocol_sha256": protocol_sha256,
            "implementation_parent_commit": TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
            "source_files_sha256": verified_source_hashes,
            "libero_commit": libero_commit,
            "expected_restore_transaction_count": TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT,
            "expected_probe_trajectory_count": TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT,
            "expected_probe_step_count": TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
            "renderer_contract": protocol["renderer_contract"],
            "renderer_provenance": runtime_renderer_provenance,
            "renderer_qualification_boundary": list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY),
            "formal_outcomes_revealed": False,
        }
        if summary["status"] != TASK14R_R0_STATUS_PASSED:
            raise RuntimeError("Task14R R0 aggregate gate failed")
        write_json(args.output_root / "TASK14R_R0_FINAL.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as error:
        summary = {
            **task14r_r0_terminal_summary(task_audits),
            "status": TASK14R_R0_STATUS_FAILED,
            "repository_commit": repository_commit,
            "repository_dirty": False,
            "protocol_sha256": protocol_sha256,
            "implementation_parent_commit": TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
            "source_files_sha256": verified_source_hashes,
            "libero_commit": libero_commit,
            "renderer_contract": protocol["renderer_contract"],
            "renderer_provenance": runtime_renderer_provenance,
            "renderer_qualification_boundary": list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY),
            "failed_case": current_case,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "formal_outcomes_revealed": False,
        }
        write_json(args.output_root / "TASK14R_R0_FAILURE.json", summary)
        print(json.dumps(summary, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
