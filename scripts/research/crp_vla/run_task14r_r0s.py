#!/usr/bin/env python
"""Run the frozen policy-free Task14R R0S renderer-shift audit."""

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import structured_hash
from run_libero_paired_four_arm_pilot import configure_standard_libero, git_repository_value
from task14r_renderer_shift import (
    TASK14R_R0S_CAMERA_KEYS,
    TASK14R_R0S_EVIDENCE_LABELS,
    TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0S_LIBERO_COMMIT,
    TASK14R_R0S_PROBE_STEPS,
    TASK14R_R0S_QUALIFICATION_BOUNDARY,
    TASK14R_R0S_RENDERERS,
    TASK14R_R0S_REPEATS_PER_RENDERER,
    TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH,
    TASK14R_R0S_SOURCE_FILES,
    TASK14R_R0S_STATUS_FAILED,
    Task14RRendererShiftError,
    aggregate_pixel_differences,
    configure_renderer,
    pixel_difference,
    renderer_invariant_python_state,
    renderer_normalized_model_fingerprint,
    split_observation,
    task14r_r0s_terminal_summary,
    validate_required_r0_final,
    validate_task14r_r0s_protocol,
)
from task14r_reset_transaction import (
    TASK14R_R0_PROBE_ACTIONS,
    build_complete_state_capsule,
    capture_mujoco_integration_state,
    capture_python_transaction_state,
    contact_state,
    restore_mujoco_integration_state,
    restore_python_transaction_state,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PHYSICS_FIELDS = (
    "step",
    "action",
    "integration_state_sha256",
    "renderer_invariant_python_state_sha256",
    "robot_state_sha256",
    "contact_state_sha256",
    "terminated",
    "truncated",
    "success_predicate",
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


def write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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


def write_frame_payload(path: Path, frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    arrays = {
        camera: np.stack([np.asarray(frame["pixels"][camera]) for frame in frames])
        for camera in TASK14R_R0S_CAMERA_KEYS
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "frame_count": len(frames),
        "camera_keys": list(TASK14R_R0S_CAMERA_KEYS),
        "arrays": {
            camera: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for camera, value in arrays.items()
        },
    }


def source_files_sha256(repository_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    hashes = {}
    for relative in TASK14R_R0S_SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Task14R R0S source file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def validate_pre_simulator_gate(
    protocol: Mapping[str, Any], repository_root: Path = REPOSITORY_ROOT
) -> tuple[dict[str, str], dict[str, Any], str]:
    hashes = source_files_sha256(repository_root)
    validate_task14r_r0s_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=hashes,
    )
    r0_path = repository_root / TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH
    if not r0_path.is_file():
        raise FileNotFoundError(f"Required Task14R R0 final is missing: {r0_path}")
    r0_sha256 = file_sha256(r0_path)
    r0_final = json.loads(r0_path.read_text())
    validate_required_r0_final(r0_final, actual_sha256=r0_sha256)
    return hashes, r0_final, r0_sha256


def _nvidia_provenance() -> dict[str, Any]:
    try:
        line = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            .strip()
            .splitlines()[0]
        )
        gpu_name, driver_version = (part.strip() for part in line.split(",", maxsplit=1))
        return {"gpu_name": gpu_name, "nvidia_driver_version": driver_version}
    except Exception as error:
        return {
            "gpu_name": None,
            "nvidia_driver_version": None,
            "nvidia_query_error": f"{type(error).__name__}: {error}",
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


def restore_renderer_transaction(
    env: Any,
    capsule: Mapping[str, Any],
    *,
    renderer: Mapping[str, Any],
    canonical_normalized_model_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    core = env._env.env
    renderer_check = configure_renderer(env, offsamples=int(renderer["offsamples"]))
    if renderer_check["offsamples_after"] != int(renderer["offsamples"]):
        raise Task14RRendererShiftError("renderer_configuration", "Renderer setting drift")
    try:
        restore_mujoco_integration_state(core.sim, capsule["integration"])
        restore_python_transaction_state(env, capsule["entry_python"])
    except Exception as error:
        raise Task14RRendererShiftError("complete_state_restore", str(error)) from error
    try:
        raw_observation = core._get_observations(force_update=True)
        observation = env._format_raw_obs(raw_observation)
    except Exception as error:
        raise Task14RRendererShiftError("observation_regeneration", str(error)) from error
    ready_python = capture_python_transaction_state(env)
    ready_integration = capture_mujoco_integration_state(core.sim)
    _, robot_state = split_observation(observation)
    _, canonical_robot_state = split_observation(capsule["canonical_observation"])
    normalized_model = renderer_normalized_model_fingerprint(core.sim)
    checks = {
        "renderer": dict(renderer_check),
        "normalized_model_identity": normalized_model["complete_sha256"] == canonical_normalized_model_sha256,
        "integration_identity": np.array_equal(
            ready_integration["state"], np.asarray(capsule["integration"]["state"])
        ),
        "renderer_invariant_python_identity": structured_hash(renderer_invariant_python_state(ready_python))
        == structured_hash(renderer_invariant_python_state(capsule["ready_python"])),
        "robot_state_identity": structured_hash(robot_state) == structured_hash(canonical_robot_state),
        "success_predicate_identity": bool(env._env.check_success())
        == bool(capsule["initial_success_predicate"]),
    }
    checks["passed"] = all(value for key, value in checks.items() if key != "renderer")
    if not checks["passed"]:
        failed = sorted(
            key for key, value in checks.items() if key not in {"renderer", "passed"} and not value
        )
        raise Task14RRendererShiftError(
            "renderer_transaction_restore", f"Renderer transaction checks failed: {failed}"
        )
    return observation, checks


def capture_frame(
    env: Any,
    observation: Mapping[str, Any],
    *,
    renderer_name: str,
    repeat_index: int,
    step: int,
    action: Sequence[float] | None,
    terminated: bool,
    truncated: bool,
    success: bool,
) -> dict[str, Any]:
    pixels, robot_state = split_observation(observation)
    core = env._env.env
    integration = capture_mujoco_integration_state(core.sim)
    record = {
        "renderer": str(renderer_name),
        "repeat_index": int(repeat_index),
        "step": int(step),
        "action": None if action is None else [float(value) for value in action],
        "integration_state_sha256": integration["state_sha256"],
        "renderer_invariant_python_state_sha256": structured_hash(
            renderer_invariant_python_state(capture_python_transaction_state(env))
        ),
        "robot_state_sha256": structured_hash(robot_state),
        "contact_state_sha256": structured_hash(contact_state(core)),
        "pixel_sha256": {
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in pixels.items()
        },
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_predicate": bool(success),
    }
    return {"record": record, "pixels": pixels}


def compare_physics(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    left_record = left["record"]
    right_record = right["record"]
    return [field for field in _PHYSICS_FIELDS if left_record.get(field) != right_record.get(field)]


def compare_pixel_frames(
    left: Mapping[str, Any], right: Mapping[str, Any], *, comparison: str
) -> list[dict[str, Any]]:
    return [
        {
            "comparison": comparison,
            "camera": camera,
            "left_renderer": left["record"]["renderer"],
            "right_renderer": right["record"]["renderer"],
            "left_repeat_index": left["record"]["repeat_index"],
            "right_repeat_index": right["record"]["repeat_index"],
            "step": left["record"]["step"],
            **pixel_difference(left["pixels"][camera], right["pixels"][camera]),
        }
        for camera in TASK14R_R0S_CAMERA_KEYS
    ]


def run_trajectory(
    env: Any,
    capsule: Mapping[str, Any],
    *,
    renderer: Mapping[str, Any],
    repeat_index: int,
    canonical_normalized_model_sha256: str,
    task_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation, restore_check = restore_renderer_transaction(
        env,
        capsule,
        renderer=renderer,
        canonical_normalized_model_sha256=canonical_normalized_model_sha256,
    )
    frames = [
        capture_frame(
            env,
            observation,
            renderer_name=str(renderer["name"]),
            repeat_index=repeat_index,
            step=-1,
            action=None,
            terminated=False,
            truncated=False,
            success=bool(env._env.check_success()),
        )
    ]
    rows = []
    for step, action in enumerate(TASK14R_R0_PROBE_ACTIONS):
        observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        frame = capture_frame(
            env,
            observation,
            renderer_name=str(renderer["name"]),
            repeat_index=repeat_index,
            step=step,
            action=action,
            terminated=terminated,
            truncated=truncated,
            success=bool(info.get("is_success", False)),
        )
        frames.append(frame)
        rows.append(frame["record"])
        if terminated or truncated:
            break
    if len(rows) != TASK14R_R0S_PROBE_STEPS:
        raise Task14RRendererShiftError(
            "probe_trajectory", f"Probe terminated early at {len(rows)}/{TASK14R_R0S_PROBE_STEPS}"
        )
    if any(bool(row["success_predicate"]) for row in rows):
        raise Task14RRendererShiftError("probe_trajectory", "Probe reached task success")
    normalized_model = renderer_normalized_model_fingerprint(env._env.env.sim)
    if normalized_model["complete_sha256"] != canonical_normalized_model_sha256:
        raise Task14RRendererShiftError("model_identity", "Normalized model changed during probe")
    stem = f"{renderer['name']}.repeat{repeat_index:02d}"
    trace_manifest = write_trace(task_root / "probe_traces" / f"{stem}.steps.jsonl.gz", rows)
    payload_manifest = write_frame_payload(task_root / "frame_payloads" / f"{stem}.frames.npz", frames)
    return frames, restore_check, trace_manifest, payload_manifest


def _partial_counts(
    trajectories: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
) -> tuple[int, int]:
    trajectory_count = sum(len(repeats) for repeats in trajectories.values())
    step_count = sum(max(0, len(frames) - 1) for repeats in trajectories.values() for frames in repeats)
    return trajectory_count, step_count


def run_task(env: Any, case: Mapping[str, Any], task_root: Path) -> dict[str, Any]:
    restore_checks: list[dict[str, Any]] = []
    trace_manifests: list[dict[str, Any]] = []
    payload_manifests: list[dict[str, Any]] = []
    trajectories: dict[str, list[list[dict[str, Any]]]] = {
        str(renderer["name"]): [] for renderer in TASK14R_R0S_RENDERERS
    }
    failure_stage = "canonical_capsule"
    try:
        env.init_state_id = int(case["init_state_id"])
        env.reset(seed=int(case["env_seed"]))
        standard_offsamples = int(env._env.env.sim.model.vis.quality.offsamples)
        if standard_offsamples != 4:
            raise Task14RRendererShiftError(
                "standard_renderer_contract",
                f"Expected standard offsamples=4, observed {standard_offsamples}",
            )
        capsule = build_complete_state_capsule(env, case=case)
        if bool(capsule["initial_success_predicate"]):
            raise Task14RRendererShiftError("canonical_capsule", "Canonical state is successful")
        canonical_normalized_model = renderer_normalized_model_fingerprint(env._env.env.sim)

        for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER):
            for renderer in TASK14R_R0S_RENDERERS:
                failure_stage = f"trajectory:{renderer['name']}:repeat{repeat_index}"
                frames, restore_check, trace_manifest, payload_manifest = run_trajectory(
                    env,
                    capsule,
                    renderer=renderer,
                    repeat_index=repeat_index,
                    canonical_normalized_model_sha256=canonical_normalized_model["complete_sha256"],
                    task_root=task_root,
                )
                trajectories[str(renderer["name"])].append(frames)
                restore_checks.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        **restore_check,
                    }
                )
                trace_manifests.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        **trace_manifest,
                    }
                )
                payload_manifests.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        **payload_manifest,
                    }
                )

        failure_stage = "within_renderer_physics_identity"
        within_physics_count = 0
        within_physics_identity_count = 0
        standard_repeat_pixels: list[dict[str, Any]] = []
        no_msaa_repeat_pixels: list[dict[str, Any]] = []
        for renderer_name, repeats in trajectories.items():
            reference = repeats[0]
            for repeat_index in range(1, len(repeats)):
                for frame_index, (left, right) in enumerate(
                    zip(reference, repeats[repeat_index], strict=True)
                ):
                    within_physics_count += 1
                    mismatches = compare_physics(left, right)
                    if mismatches:
                        raise Task14RRendererShiftError(
                            "within_renderer_physics_identity",
                            f"{renderer_name} repeat {repeat_index} frame {frame_index}: {mismatches}",
                        )
                    within_physics_identity_count += 1
                    rows = compare_pixel_frames(left, right, comparison=f"{renderer_name}_repeat_identity")
                    if renderer_name == "no_msaa":
                        no_msaa_repeat_pixels.extend(rows)
                    else:
                        standard_repeat_pixels.extend(rows)
        if not all(bool(row["exact"]) for row in no_msaa_repeat_pixels):
            first = next(row for row in no_msaa_repeat_pixels if not bool(row["exact"]))
            raise Task14RRendererShiftError(
                "no_msaa_repeat_pixel_identity", f"No-MSAA repeat mismatch: {json.dumps(first)}"
            )

        failure_stage = "cross_renderer_identity"
        paired_frame_count = 0
        physics_identity_count = 0
        non_image_identity_count = 0
        cross_renderer_pixels: list[dict[str, Any]] = []
        for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER):
            standard = trajectories["standard_msaa"][repeat_index]
            no_msaa = trajectories["no_msaa"][repeat_index]
            for frame_index, (left, right) in enumerate(zip(standard, no_msaa, strict=True)):
                paired_frame_count += 1
                mismatches = compare_physics(left, right)
                if mismatches:
                    raise Task14RRendererShiftError(
                        "cross_renderer_physics_identity",
                        f"repeat {repeat_index} frame {frame_index}: {mismatches}",
                    )
                physics_identity_count += 1
                if left["record"]["robot_state_sha256"] != right["record"]["robot_state_sha256"]:
                    raise Task14RRendererShiftError(
                        "cross_renderer_non_image_identity",
                        f"repeat {repeat_index} frame {frame_index}: robot_state",
                    )
                non_image_identity_count += 1
                cross_renderer_pixels.extend(
                    compare_pixel_frames(left, right, comparison="standard_msaa_vs_no_msaa")
                )

        cross_by_camera = {
            camera: aggregate_pixel_differences(
                [row for row in cross_renderer_pixels if row["camera"] == camera]
            )
            for camera in TASK14R_R0S_CAMERA_KEYS
        }
        standard_repeat_summary = aggregate_pixel_differences(standard_repeat_pixels)
        no_msaa_repeat_summary = aggregate_pixel_differences(no_msaa_repeat_pixels)
        comparison_manifest = write_trace(
            task_root / "pixel_comparisons" / "renderer_pixel_comparisons.jsonl.gz",
            [*standard_repeat_pixels, *no_msaa_repeat_pixels, *cross_renderer_pixels],
        )
        comparison_manifest["comparison_count"] = comparison_manifest.pop("step_count")
        return {
            "schema_version": "task14r.r0s.task_audit.v1",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": True,
            "standard_offsamples": standard_offsamples,
            "canonical_normalized_model_complete_sha256": canonical_normalized_model["complete_sha256"],
            "restore_transaction_count": len(restore_checks),
            "restore_checks": restore_checks,
            "probe_trajectory_count": sum(len(rows) for rows in trajectories.values()),
            "probe_step_count": sum(
                len(frames) - 1 for repeats in trajectories.values() for frames in repeats
            ),
            "trace_manifests": trace_manifests,
            "frame_payload_manifests": payload_manifests,
            "pixel_comparison_manifest": comparison_manifest,
            "renderer_pair_count": TASK14R_R0S_REPEATS_PER_RENDERER,
            "paired_frame_count": paired_frame_count,
            "physics_identity_frame_count": physics_identity_count,
            "non_image_identity_frame_count": non_image_identity_count,
            "camera_comparison_count": len(cross_renderer_pixels),
            "within_renderer_physics_frame_count": within_physics_count,
            "within_renderer_physics_identity_frame_count": within_physics_identity_count,
            "standard_repeat_camera_comparison_count": len(standard_repeat_pixels),
            "standard_repeat_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in standard_repeat_pixels
            ),
            "standard_repeat_pixels_exact": bool(standard_repeat_summary["exact"]),
            "standard_repeat_pixel_summary": standard_repeat_summary,
            "no_msaa_repeat_camera_comparison_count": len(no_msaa_repeat_pixels),
            "no_msaa_repeat_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in no_msaa_repeat_pixels
            ),
            "no_msaa_repeat_pixels_exact": bool(no_msaa_repeat_summary["exact"]),
            "no_msaa_repeat_pixel_summary": no_msaa_repeat_summary,
            "cross_renderer_pixels_exact": all(bool(row["exact"]) for row in cross_renderer_pixels),
            "cross_renderer_pixel_summary_by_camera": cross_by_camera,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
    except Exception as error:
        trajectory_count, step_count = _partial_counts(trajectories)
        return {
            "schema_version": "task14r.r0s.task_failure_audit.v1",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": False,
            "failure_stage": str(getattr(error, "failure_stage", failure_stage)),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "restore_transaction_count": len(restore_checks),
            "partial_restore_checks": restore_checks,
            "probe_trajectory_count": trajectory_count,
            "probe_step_count": step_count,
            "partial_trace_manifests": trace_manifests,
            "partial_frame_payload_manifests": payload_manifests,
            "renderer_pair_count": 0,
            "paired_frame_count": 0,
            "physics_identity_frame_count": 0,
            "non_image_identity_frame_count": 0,
            "camera_comparison_count": 0,
            "within_renderer_physics_identity_frame_count": 0,
            "no_msaa_repeat_camera_comparison_count": 0,
            "no_msaa_repeat_exact_camera_comparison_count": 0,
            "cross_renderer_pixels_exact": False,
            "standard_repeat_pixels_exact": False,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }


def execute_task_with_audit(
    *,
    case: Mapping[str, Any],
    task_root: Path,
    environment_factory: Callable[[], Any],
    task_runner: Callable[[Any, Mapping[str, Any], Path], dict[str, Any]] = run_task,
) -> dict[str, Any]:
    env = None
    audit = None
    try:
        env = environment_factory()
        audit = task_runner(env, case, task_root)
    except Exception as error:
        audit = {
            "schema_version": "task14r.r0s.task_failure_audit.v1",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": False,
            "failure_stage": str(getattr(error, "failure_stage", "environment_creation")),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "restore_transaction_count": 0,
            "probe_trajectory_count": 0,
            "probe_step_count": 0,
            "renderer_pair_count": 0,
            "paired_frame_count": 0,
            "physics_identity_frame_count": 0,
            "non_image_identity_frame_count": 0,
            "camera_comparison_count": 0,
            "within_renderer_physics_identity_frame_count": 0,
            "no_msaa_repeat_camera_comparison_count": 0,
            "no_msaa_repeat_exact_camera_comparison_count": 0,
            "cross_renderer_pixels_exact": False,
            "standard_repeat_pixels_exact": False,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if audit is None or bool(audit.get("passed")):
                    audit = {
                        **({} if audit is None else audit),
                        "passed": False,
                        "failure_stage": "environment_close",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    }
    if audit is None:
        raise AssertionError("Task14R R0S task audit was not constructed")
    filename = "TASK_AUDIT.json" if bool(audit.get("passed")) else "TASK_AUDIT_FAILURE.json"
    write_json(task_root / filename, audit)
    return audit


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task14R R0S")
    protocol = json.loads(args.protocol.read_text())
    source_hashes, r0_final, r0_final_sha256 = validate_pre_simulator_gate(protocol)
    if git_value("rev-parse", "HEAD^") != TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT:
        raise RuntimeError("Task14R R0S implementation parent commit drift")
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        raise RuntimeError("Task14R R0S requires MUJOCO_GL=egl")
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    if libero_commit != TASK14R_R0S_LIBERO_COMMIT:
        raise RuntimeError("Pinned LIBERO commit drift")

    protocol_sha256 = file_sha256(args.protocol)
    args.output_root.mkdir(parents=True, exist_ok=False)
    configure_standard_libero(args.libero_root, args.output_root / "libero_standard_config")

    import libero.libero as libero_module
    import mujoco
    import robosuite
    from libero.libero import benchmark

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    repository_commit = git_value("rev-parse", "HEAD")
    renderer_provenance = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **_nvidia_provenance(),
        "mujoco_version": mujoco.__version__,
        "robosuite_version": robosuite.__version__,
        "egl_vendor": None,
        "egl_version": None,
        "egl_query_status": "PENDING_CURRENT_CONTEXT",
    }
    run_metadata = {
        "schema_version": "task14r.r0s.run_metadata.v1",
        "status": "TASK14R_R0S_RUNNING",
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "implementation_parent_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "source_files_sha256": source_hashes,
        "libero_commit": libero_commit,
        "required_r0_final_path": TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH,
        "required_r0_final_sha256": r0_final_sha256,
        "required_r0_status": r0_final["status"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "renderer_contract": protocol["renderer_contract"],
        "renderer_provenance": renderer_provenance,
        "qualification_boundary": list(TASK14R_R0S_QUALIFICATION_BOUNDARY),
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
    }
    write_json(args.output_root / "RUN_METADATA.json", run_metadata)

    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in sorted({row["suite"] for row in protocol["tasks"]})}
    task_audits = []
    current_case = None
    try:
        for current_case in protocol["tasks"]:
            suite_name = str(current_case["suite"])
            task_id = int(current_case["task_id"])
            task_root = args.output_root / "tasks" / str(current_case["case_id"])
            audit = execute_task_with_audit(
                case=current_case,
                task_root=task_root,
                environment_factory=lambda suite_name=suite_name, task_id=task_id: make_environment(
                    suites[suite_name], suite_name, task_id
                ),
            )
            task_audits.append(audit)
            renderer_provenance.update(query_current_egl_provenance())
            partial = {
                **task14r_r0s_terminal_summary(task_audits),
                "status": "TASK14R_R0S_RUNNING_PARTIAL",
                "repository_commit": repository_commit,
                "protocol_sha256": protocol_sha256,
                "source_files_sha256": source_hashes,
                "required_r0_final_sha256": r0_final_sha256,
                "renderer_provenance": renderer_provenance,
            }
            write_json(args.output_root / "TASK14R_R0S_PARTIAL.json", partial)
            if not bool(audit["passed"]):
                raise RuntimeError(f"Task14R R0S task gate failed: {current_case['case_id']}")

        summary = {
            **task14r_r0s_terminal_summary(task_audits),
            "repository_commit": repository_commit,
            "repository_dirty": False,
            "implementation_parent_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
            "protocol_sha256": protocol_sha256,
            "source_files_sha256": source_hashes,
            "libero_commit": libero_commit,
            "required_r0_final_sha256": r0_final_sha256,
            "renderer_provenance": renderer_provenance,
            "formal_outcomes_revealed": False,
        }
        if summary["status"] == TASK14R_R0S_STATUS_FAILED:
            raise RuntimeError("Task14R R0S aggregate gate failed")
        write_json(args.output_root / "TASK14R_R0S_FINAL.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as error:
        summary = {
            **task14r_r0s_terminal_summary(task_audits),
            "status": TASK14R_R0S_STATUS_FAILED,
            "repository_commit": repository_commit,
            "repository_dirty": False,
            "implementation_parent_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
            "protocol_sha256": protocol_sha256,
            "source_files_sha256": source_hashes,
            "libero_commit": libero_commit,
            "required_r0_final_sha256": r0_final_sha256,
            "renderer_provenance": renderer_provenance,
            "failed_case": current_case,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "formal_outcomes_revealed": False,
        }
        write_json(args.output_root / "TASK14R_R0S_FAILURE.json", summary)
        print(json.dumps(summary, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
