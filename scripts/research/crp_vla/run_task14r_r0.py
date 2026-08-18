#!/usr/bin/env python
"""Run the policy-free Task14R R0 complete-state transaction gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import load_capsule, structured_hash, write_capsule
from run_libero_paired_four_arm_pilot import configure_standard_libero, git_repository_value
from task14r_reset_transaction import (
    TASK14R_R0_EVIDENCE_LABELS,
    TASK14R_R0_LIBERO_COMMIT,
    TASK14R_R0_PROBE_ACTIONS,
    TASK14R_R0_RENDERER_BACKEND,
    TASK14R_R0_RESTORE_REPEATS,
    TASK14R_R0_STATUS_FAILED,
    TASK14R_R0_STATUS_PASSED,
    build_complete_state_capsule,
    capture_probe_step,
    compare_probe_repeats,
    configure_deterministic_renderer,
    mujoco_model_fingerprint,
    restore_complete_state_transaction,
    task14r_r0_terminal_summary,
    validate_task14r_r0_protocol,
)


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
    # This is the only outer reset in the task.  _ensure_env() performs the
    # fixed initial construction; no reset method is called after the capsule.
    env.init_state_id = int(case["init_state_id"])
    env.reset(seed=int(case["env_seed"]))
    renderer_contract = configure_deterministic_renderer(env)
    capsule = build_complete_state_capsule(env, case=case)
    if capsule["initial_success_predicate"]:
        raise RuntimeError(f"R0 canonical state is initially successful: {case['case_id']}")

    capsule_path = task_root / "complete_state_capsule"
    capsule_record = write_capsule(
        capsule_path,
        {
            "schema_version": "task14r.r0.capsule_record.v1",
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
        raise RuntimeError(f"Complete-state capsule round trip failed: {case['case_id']}")

    restore_checks = []
    probe_repeats = []
    model_checks = []
    for repeat_index in range(TASK14R_R0_RESTORE_REPEATS):
        observation, checks = restore_complete_state_transaction(env, loaded_capsule)
        restore_checks.append(
            {
                "repeat_index": repeat_index,
                **checks,
                "observation_sha256": structured_hash(observation),
            }
        )
        rows = []
        for step, action in enumerate(TASK14R_R0_PROBE_ACTIONS):
            observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            rows.append(
                capture_probe_step(
                    env,
                    observation,
                    step=step,
                    action=action,
                    terminated=terminated,
                    truncated=truncated,
                    success=bool(info.get("is_success", False)),
                )
            )
            if terminated or truncated:
                break
        probe_repeats.append(rows)
        model = mujoco_model_fingerprint(env._env.env.sim)
        model_checks.append(
            {
                "repeat_index": repeat_index,
                "complete_sha256": model["complete_sha256"],
                "identity": model["complete_sha256"] == loaded_capsule["model"]["complete_sha256"],
            }
        )

    comparison = compare_probe_repeats(probe_repeats)
    full_length = all(len(rows) == len(TASK14R_R0_PROBE_ACTIONS) for rows in probe_repeats)
    no_probe_success = all(not any(bool(row["success_predicate"]) for row in rows) for rows in probe_repeats)
    passed = (
        all(bool(row["passed"]) for row in restore_checks)
        and all(bool(row["identity"]) for row in model_checks)
        and comparison["passed"]
        and full_length
        and no_probe_success
    )
    return {
        "schema_version": "task14r.r0.task_audit.v1",
        "case": case,
        "evidence_labels": list(TASK14R_R0_EVIDENCE_LABELS),
        "passed": passed,
        "capsule": {"path": capsule_record["path"], "sha256": capsule_record["sha256"]},
        "canonical_model_complete_sha256": loaded_capsule["model"]["complete_sha256"],
        "canonical_model_mjb_sha256": loaded_capsule["model"]["mjb_sha256"],
        "canonical_integration_state_sha256": loaded_capsule["integration"]["state_sha256"],
        "canonical_observation_sha256": loaded_capsule["canonical_observation_sha256"],
        "renderer_contract": renderer_contract,
        "restore_transaction_count": len(restore_checks),
        "restore_checks": restore_checks,
        "probe_trajectory_count": len(probe_repeats),
        "probe_step_counts": [len(rows) for rows in probe_repeats],
        "probe_repeat_comparison": comparison,
        "model_checks": model_checks,
        "probe_completed_full_length": full_length,
        "probe_success_predicate_never_true": no_probe_success,
        "outer_reset_count_after_capsule": 0,
        "hard_model_rebuild_count_after_capsule": 0,
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task14R R0")
    protocol = json.loads(args.protocol.read_text())
    validate_task14r_r0_protocol(protocol)
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
    run_metadata = {
        "schema_version": "task14r.r0.run_metadata.v1",
        "status": "TASK14R_R0_RUNNING",
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "libero_commit": libero_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
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
            env = make_environment(suites[suite_name], suite_name, task_id)
            try:
                audit = run_task(
                    env,
                    current_case,
                    args.output_root / "tasks" / task_slug(current_case),
                )
            finally:
                env.close()
            task_audits.append(audit)
            write_json(
                args.output_root / "tasks" / task_slug(current_case) / "TASK_AUDIT.json",
                audit,
            )
            partial = {
                **task14r_r0_terminal_summary(task_audits),
                "status": "TASK14R_R0_RUNNING_PARTIAL",
                "repository_commit": repository_commit,
                "protocol_sha256": protocol_sha256,
            }
            write_json(args.output_root / "TASK14R_R0_PARTIAL.json", partial)
            if not audit["passed"]:
                raise RuntimeError(f"Task14R R0 task gate failed: {current_case['case_id']}")

        summary = {
            **task14r_r0_terminal_summary(task_audits),
            "repository_commit": repository_commit,
            "repository_dirty": False,
            "protocol_sha256": protocol_sha256,
            "libero_commit": libero_commit,
            "expected_restore_transaction_count": 120,
            "expected_probe_trajectory_count": 120,
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
            "libero_commit": libero_commit,
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
