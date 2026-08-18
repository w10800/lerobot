#!/usr/bin/env python
"""Frozen policy-free renderer-shift contract for Task14R R0S."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .replay_v2_common import array_sha256
    from .task14r_reset_transaction import (
        TASK14R_R0_PROBE_ACTIONS,
        TASK14R_R0_SUITES,
        TASK14R_R0_TASKS_PER_SUITE,
        mujoco_model_fingerprint,
        task14r_r0_seed,
    )
except ImportError:  # Direct script execution adds this directory to sys.path.
    from replay_v2_common import array_sha256
    from task14r_reset_transaction import (
        TASK14R_R0_PROBE_ACTIONS,
        TASK14R_R0_SUITES,
        TASK14R_R0_TASKS_PER_SUITE,
        mujoco_model_fingerprint,
        task14r_r0_seed,
    )

TASK14R_R0S_NAME = "TASK14R_R0S_POLICY_FREE_RENDERER_SHIFT_AUDIT"
TASK14R_R0S_SCHEMA_VERSION = "task14r.r0s.renderer_shift.protocol.v1"
TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT = "60fbf7370d8668316b34a9d0d1a334247f792cca"
TASK14R_R0S_LIBERO_COMMIT = "8460457bfca6e0ef2e856bc104e2c60b023ef2a7"
TASK14R_R0S_REQUIRED_R0_FINAL_SHA256 = "ae443481ec991f6c46cb74a8bf4adf90817cd528d5f1ff4e44d04f6727df1a3e"
TASK14R_R0S_REQUIRED_R0_STATUS = "TASK14R_R0_RESTORE_TRANSACTION_PASSED"
TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH = (
    "outputs/crp_vla/task14r_reset_transaction_recovery/r0_attempt001/TASK14R_R0_FINAL.json"
)
TASK14R_R0S_STATUS_FROZEN = "TASK14R_R0S_PROTOCOL_FROZEN"
TASK14R_R0S_STATUS_EXACT = "TASK14R_R0S_RENDERERS_EXACTLY_EQUIVALENT"
TASK14R_R0S_STATUS_SHIFT = "TASK14R_R0S_RENDERER_SHIFT_CHARACTERIZED"
TASK14R_R0S_STATUS_FAILED = "TASK14R_R0S_RENDERER_AUDIT_FAILED"
TASK14R_R0S_EVIDENCE_LABELS = (
    "ENGINEERING_ONLY",
    "RENDERER_DIAGNOSTIC",
    "NON_PRIMARY",
)
TASK14R_R0S_RENDERERS = (
    {"name": "standard_msaa", "offsamples": 4},
    {"name": "no_msaa", "offsamples": 0},
)
TASK14R_R0S_REPEATS_PER_RENDERER = 3
TASK14R_R0S_PROBE_STEPS = len(TASK14R_R0_PROBE_ACTIONS)
TASK14R_R0S_FRAMES_PER_TRAJECTORY = TASK14R_R0S_PROBE_STEPS + 1
TASK14R_R0S_CAMERA_KEYS = ("image1", "image2")
TASK14R_R0S_EXPECTED_TASK_COUNT = 40
TASK14R_R0S_EXPECTED_RESTORE_TRANSACTION_COUNT = 240
TASK14R_R0S_EXPECTED_PROBE_TRAJECTORY_COUNT = 240
TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT = 3600
TASK14R_R0S_EXPECTED_RENDERER_PAIR_COUNT = 120
TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT = 1920
TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT = 3840
TASK14R_R0S_EXPECTED_WITHIN_RENDERER_PHYSICS_FRAME_COUNT = 2560
TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT = 2560
TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT = 2560
TASK14R_R0S_SOURCE_FILES = (
    "scripts/research/crp_vla/run_task14r_r0s.py",
    "scripts/research/crp_vla/task14r_renderer_shift.py",
    "scripts/research/crp_vla/task14r_reset_transaction.py",
    "scripts/research/crp_vla/launch_task14r_r0s.sh",
)
TASK14R_R0S_QUALIFICATION_BOUNDARY = (
    "R0S is policy-free and compares standard EGL MSAA with offsamples=0.",
    "Only exact cross-render pixels establish exact renderer equivalence.",
    "A nonzero pixel shift is characterized without an outcome-tuned tolerance.",
    "No R0S terminal status automatically authorizes R1, policy execution, or formal outcomes.",
)


class Task14RRendererShiftError(RuntimeError):
    """A fail-closed R0S error with a stable audit stage."""

    def __init__(self, failure_stage: str, message: str) -> None:
        super().__init__(message)
        self.failure_stage = str(failure_stage)


def _validate_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Task14R R0S invalid SHA-256: {field}")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Task14R R0S invalid SHA-256: {field}")


def build_task14r_r0s_protocol(
    *, implementation_parent_commit: str, source_files_sha256: Mapping[str, str]
) -> dict[str, Any]:
    tasks = [
        {
            "suite": suite,
            "task_id": task_id,
            "init_state_id": 0,
            "env_seed": task14r_r0_seed(suite, task_id),
            "case_id": f"r0s-{suite}-task{task_id:02d}-state00",
        }
        for suite in TASK14R_R0_SUITES
        for task_id in range(TASK14R_R0_TASKS_PER_SUITE)
    ]
    return {
        "schema_version": TASK14R_R0S_SCHEMA_VERSION,
        "task_name": TASK14R_R0S_NAME,
        "phase": "R0S_POLICY_FREE_RENDERER_SHIFT_AUDIT",
        "status": TASK14R_R0S_STATUS_FROZEN,
        "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
        "implementation_parent_commit": str(implementation_parent_commit),
        "libero_commit": TASK14R_R0S_LIBERO_COMMIT,
        "required_r0_final": {
            "relative_path": TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH,
            "sha256": TASK14R_R0S_REQUIRED_R0_FINAL_SHA256,
            "status": TASK14R_R0S_REQUIRED_R0_STATUS,
            "repository_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
            "task_count": 40,
            "passed_task_count": 40,
            "failed_task_count": 0,
            "restore_transaction_count": 120,
            "probe_trajectory_count": 120,
            "probe_step_count": 1800,
        },
        "renderer_contract": {
            "backend": "egl",
            "arms": [dict(row) for row in TASK14R_R0S_RENDERERS],
            "context_rebuilt_before_every_trajectory": True,
            "simulator_or_model_rebuild_after_capsule": False,
            "pixel_gate": "EXACT_OR_CHARACTERIZE_WITHOUT_TOLERANCE",
        },
        "qualification_boundary": list(TASK14R_R0S_QUALIFICATION_BOUNDARY),
        "source_files_sha256": dict(sorted(source_files_sha256.items())),
        "task_count": len(tasks),
        "repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
        "probe_steps_per_trajectory": TASK14R_R0S_PROBE_STEPS,
        "frames_per_trajectory": TASK14R_R0S_FRAMES_PER_TRAJECTORY,
        "camera_keys": list(TASK14R_R0S_CAMERA_KEYS),
        "probe_actions": [list(row) for row in TASK14R_R0_PROBE_ACTIONS],
        "tasks": tasks,
        "expected_task_count": TASK14R_R0S_EXPECTED_TASK_COUNT,
        "expected_restore_transaction_count": TASK14R_R0S_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "expected_probe_trajectory_count": TASK14R_R0S_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "expected_probe_step_count": TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT,
        "expected_renderer_pair_count": TASK14R_R0S_EXPECTED_RENDERER_PAIR_COUNT,
        "expected_paired_frame_count": TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "expected_camera_comparison_count": TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
        "expected_within_renderer_physics_frame_count": (
            TASK14R_R0S_EXPECTED_WITHIN_RENDERER_PHYSICS_FRAME_COUNT
        ),
        "expected_standard_repeat_camera_comparison_count": (
            TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT
        ),
        "expected_no_msaa_repeat_camera_comparison_count": (
            TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT
        ),
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
        "permitted_terminal_statuses": [
            TASK14R_R0S_STATUS_EXACT,
            TASK14R_R0S_STATUS_SHIFT,
            TASK14R_R0S_STATUS_FAILED,
        ],
    }


def validate_task14r_r0s_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_implementation_parent_commit: str,
    expected_source_files_sha256: Mapping[str, str],
) -> None:
    if set(expected_source_files_sha256) != set(TASK14R_R0S_SOURCE_FILES):
        raise ValueError("Task14R R0S source hash inventory drift")
    for relative, digest in expected_source_files_sha256.items():
        _validate_sha256(digest, field=relative)
    expected = build_task14r_r0s_protocol(
        implementation_parent_commit=expected_implementation_parent_commit,
        source_files_sha256=expected_source_files_sha256,
    )
    for field, expected_value in expected.items():
        if protocol.get(field) != expected_value:
            raise ValueError(f"Task14R R0S protocol field drift: {field}")
    if len(protocol) != len(expected):
        raise ValueError("Task14R R0S protocol field inventory drift")


def validate_required_r0_final(value: Mapping[str, Any], *, actual_sha256: str) -> None:
    _validate_sha256(actual_sha256, field="required_r0_final")
    if actual_sha256 != TASK14R_R0S_REQUIRED_R0_FINAL_SHA256:
        raise ValueError("Task14R R0S required R0 final SHA-256 drift")
    expected_scalars = {
        "status": TASK14R_R0S_REQUIRED_R0_STATUS,
        "repository_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        "task_count": 40,
        "passed_task_count": 40,
        "failed_task_count": 0,
        "restore_transaction_count": 120,
        "probe_trajectory_count": 120,
        "probe_step_count": 1800,
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
    }
    for field, expected in expected_scalars.items():
        if value.get(field) != expected:
            raise ValueError(f"Task14R R0S required R0 final field drift: {field}")
    terminal = value.get("terminal_gate_checks")
    if not isinstance(terminal, Mapping) or not terminal or not all(terminal.values()):
        raise ValueError("Task14R R0S requires every R0 terminal gate to be true")


def configure_renderer(env: Any, *, offsamples: int) -> dict[str, Any]:
    if int(offsamples) not in {0, 4}:
        raise ValueError("Task14R R0S renderer offsamples must be 0 or 4")
    sim = env._env.env.sim
    context = sim._render_context_offscreen
    if context is None:
        raise RuntimeError("Offscreen render context is unavailable")
    before = int(sim.model.vis.quality.offsamples)
    sim.model.vis.quality.offsamples = int(offsamples)
    context.con.free()
    context._set_mujoco_context_and_buffers()
    sim.forward()
    after = int(sim.model.vis.quality.offsamples)
    if after != int(offsamples):
        raise RuntimeError("Task14R R0S failed to set renderer offsamples")
    return {
        "backend": "egl",
        "offsamples_before": before,
        "offsamples_after": after,
        "framebuffer_context_rebuilt": True,
        "simulator_or_model_rebuilt": False,
    }


def renderer_normalized_model_fingerprint(sim: Any) -> dict[str, Any]:
    """Hash a model after normalizing only the audited renderer field."""
    original = int(sim.model.vis.quality.offsamples)
    try:
        sim.model.vis.quality.offsamples = 0
        return mujoco_model_fingerprint(sim)
    finally:
        sim.model.vis.quality.offsamples = original


def _is_rendered_entry(name: Any, value: Any) -> bool:
    lowered = str(name).lower()
    return (
        isinstance(value, np.ndarray)
        and value.ndim >= 2
        and value.size > 0
        and ("image" in lowered or "rgb" in lowered or "camera" in lowered)
    )


def renderer_invariant_python_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only rendered array values while retaining timing and dynamics state."""
    value = copy.deepcopy(dict(snapshot))
    cache = value.get("obs_cache")
    if isinstance(cache, Mapping):
        value["obs_cache"] = {
            str(key): child for key, child in cache.items() if not _is_rendered_entry(key, child)
        }
    observables = value.get("observables")
    if isinstance(observables, Mapping):
        for name, observable in observables.items():
            if not isinstance(observable, dict):
                continue
            if _is_rendered_entry(name, observable.get("_current_observed_value")):
                observable.pop("_current_observed_value", None)
    return value


def split_observation(observation: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping) or set(pixels) != set(TASK14R_R0S_CAMERA_KEYS):
        raise ValueError("Task14R R0S observation camera inventory drift")
    images = {str(key): np.ascontiguousarray(np.asarray(pixels[key])).copy() for key in sorted(pixels)}
    for key, image in images.items():
        if image.ndim != 3 or image.size == 0 or not np.all(np.isfinite(image)):
            raise ValueError(f"Task14R R0S invalid camera observation: {key}")
    if "robot_state" not in observation:
        raise ValueError("Task14R R0S robot_state is missing")
    return images, observation["robot_state"]


def pixel_difference(left: Any, right: Any) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ValueError("Task14R R0S camera shape or dtype drift")
    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    if left_array.ndim != 3:
        raise ValueError("Task14R R0S camera arrays must be three-dimensional")
    channel_axis = 2 if left_array.shape[2] in {1, 3, 4} else 0
    changed_pixel_mask = np.any(difference != 0.0, axis=channel_axis)
    return {
        "shape": list(left_array.shape),
        "dtype": str(left_array.dtype),
        "left_sha256": array_sha256(np.ascontiguousarray(left_array)),
        "right_sha256": array_sha256(np.ascontiguousarray(right_array)),
        "exact": bool(np.array_equal(left_array, right_array)),
        "value_count": int(difference.size),
        "changed_value_count": int(np.count_nonzero(difference)),
        "pixel_count": int(changed_pixel_mask.size),
        "changed_pixel_count": int(np.count_nonzero(changed_pixel_mask)),
        "maximum_absolute_difference": float(np.max(difference, initial=0.0)),
        "sum_absolute_difference": float(np.sum(difference, dtype=np.float64)),
        "sum_squared_difference": float(np.sum(np.square(difference), dtype=np.float64)),
    }


def aggregate_pixel_differences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value_count = sum(int(row["value_count"]) for row in rows)
    pixel_count = sum(int(row["pixel_count"]) for row in rows)
    sum_absolute = sum(float(row["sum_absolute_difference"]) for row in rows)
    sum_squared = sum(float(row["sum_squared_difference"]) for row in rows)
    return {
        "comparison_count": len(rows),
        "exact_comparison_count": sum(bool(row["exact"]) for row in rows),
        "shifted_comparison_count": sum(not bool(row["exact"]) for row in rows),
        "value_count": value_count,
        "changed_value_count": sum(int(row["changed_value_count"]) for row in rows),
        "pixel_count": pixel_count,
        "changed_pixel_count": sum(int(row["changed_pixel_count"]) for row in rows),
        "maximum_absolute_difference": max(
            (float(row["maximum_absolute_difference"]) for row in rows), default=0.0
        ),
        "mean_absolute_difference": 0.0 if value_count == 0 else sum_absolute / value_count,
        "root_mean_squared_difference": (
            0.0 if value_count == 0 else float(np.sqrt(sum_squared / value_count))
        ),
        "exact": all(bool(row["exact"]) for row in rows),
    }


def task14r_r0s_terminal_summary(task_audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(task_audits)
    passed = sum(bool(row.get("passed")) for row in task_audits)
    failed = task_count - passed

    def total(field: str) -> int:
        return sum(int(row.get(field, 0)) for row in task_audits)

    restore_count = total("restore_transaction_count")
    trajectory_count = total("probe_trajectory_count")
    step_count = total("probe_step_count")
    pair_count = total("renderer_pair_count")
    paired_frames = total("paired_frame_count")
    physics_frames = total("physics_identity_frame_count")
    non_image_frames = total("non_image_identity_frame_count")
    camera_comparisons = total("camera_comparison_count")
    within_physics = total("within_renderer_physics_frame_count")
    within_physics_identity = total("within_renderer_physics_identity_frame_count")
    standard_comparisons = total("standard_repeat_camera_comparison_count")
    no_msaa_comparisons = total("no_msaa_repeat_camera_comparison_count")
    no_msaa_exact = total("no_msaa_repeat_exact_camera_comparison_count")
    policy_queries = total("policy_query_count")
    formal_cases = total("formal_case_count")
    formal_outcomes = total("formal_outcome_rollout_count")
    training_updates = any(bool(row.get("training_or_parameter_updates")) for row in task_audits)
    automatic_next = any(bool(row.get("automatic_next_phase")) for row in task_audits)
    gate = {
        "task_count": task_count == TASK14R_R0S_EXPECTED_TASK_COUNT,
        "passed_task_count": passed == TASK14R_R0S_EXPECTED_TASK_COUNT,
        "failed_task_count": failed == 0,
        "restore_transaction_count": restore_count == TASK14R_R0S_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "probe_trajectory_count": trajectory_count == TASK14R_R0S_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "probe_step_count": step_count == TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT,
        "renderer_pair_count": pair_count == TASK14R_R0S_EXPECTED_RENDERER_PAIR_COUNT,
        "paired_frame_count": paired_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "physics_identity_frame_count": physics_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "non_image_identity_frame_count": non_image_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "camera_comparison_count": camera_comparisons == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
        "within_renderer_physics_frame_count": within_physics
        == TASK14R_R0S_EXPECTED_WITHIN_RENDERER_PHYSICS_FRAME_COUNT,
        "within_renderer_physics_identity_frame_count": within_physics_identity
        == TASK14R_R0S_EXPECTED_WITHIN_RENDERER_PHYSICS_FRAME_COUNT,
        "standard_repeat_camera_comparison_count": standard_comparisons
        == TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT,
        "no_msaa_repeat_camera_comparison_count": no_msaa_comparisons
        == TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT,
        "no_msaa_repeat_exact_camera_comparison_count": no_msaa_exact
        == TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT,
        "policy_query_count": policy_queries == 0,
        "formal_case_count": formal_cases == 0,
        "formal_outcome_rollout_count": formal_outcomes == 0,
        "training_or_parameter_updates": not training_updates,
        "automatic_next_phase": not automatic_next,
    }
    technical_pass = all(gate.values())
    exact_pixels = technical_pass and all(
        bool(row.get("cross_renderer_pixels_exact")) and bool(row.get("standard_repeat_pixels_exact"))
        for row in task_audits
    )
    if not technical_pass:
        status = TASK14R_R0S_STATUS_FAILED
    elif exact_pixels:
        status = TASK14R_R0S_STATUS_EXACT
    else:
        status = TASK14R_R0S_STATUS_SHIFT
    return {
        "status": status,
        "task_count": task_count,
        "passed_task_count": passed,
        "failed_task_count": failed,
        "restore_transaction_count": restore_count,
        "probe_trajectory_count": trajectory_count,
        "probe_step_count": step_count,
        "renderer_pair_count": pair_count,
        "paired_frame_count": paired_frames,
        "physics_identity_frame_count": physics_frames,
        "non_image_identity_frame_count": non_image_frames,
        "camera_comparison_count": camera_comparisons,
        "within_renderer_physics_frame_count": within_physics,
        "within_renderer_physics_identity_frame_count": within_physics_identity,
        "standard_repeat_camera_comparison_count": standard_comparisons,
        "no_msaa_repeat_camera_comparison_count": no_msaa_comparisons,
        "no_msaa_repeat_exact_camera_comparison_count": no_msaa_exact,
        "pixel_shift_task_count": sum(
            not bool(row.get("cross_renderer_pixels_exact"))
            or not bool(row.get("standard_repeat_pixels_exact"))
            for row in task_audits
        ),
        "policy_query_count": policy_queries,
        "formal_case_count": formal_cases,
        "formal_outcome_rollout_count": formal_outcomes,
        "training_or_parameter_updates": training_updates,
        "automatic_next_phase": automatic_next,
        "terminal_gate_checks": gate,
        "qualification_boundary": list(TASK14R_R0S_QUALIFICATION_BOUNDARY),
    }
