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
TASK14R_R0S_SCHEMA_VERSION = "task14r.r0s.renderer_shift.protocol.v2"
TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT = "609118b7016a411c52a0ce5a6bdd8dc36ab74d79"
TASK14R_R0S_LIBERO_COMMIT = "8460457bfca6e0ef2e856bc104e2c60b023ef2a7"
TASK14R_R0S_REQUIRED_R0_FINAL_SHA256 = "ae443481ec991f6c46cb74a8bf4adf90817cd528d5f1ff4e44d04f6727df1a3e"
TASK14R_R0S_REQUIRED_R0_STATUS = "TASK14R_R0_RESTORE_TRANSACTION_PASSED"
TASK14R_R0S_REQUIRED_R0_REPOSITORY_COMMIT = "60fbf7370d8668316b34a9d0d1a334247f792cca"
TASK14R_R0S_REQUIRED_R0_PROTOCOL_SHA256 = "2e6a2507b45444d998f456ea6f8a7e1438a8e94ab918f428cb02becb1f4aad16"
TASK14R_R0S_REQUIRED_R0_SOURCE_FILES_SHA256 = {
    "scripts/research/crp_vla/launch_task14r_r0.sh": (
        "3794050b8869477c82cf5f8ee51d2a8441407176dda8767eff533eba883a08b7"
    ),
    "scripts/research/crp_vla/run_task14r_r0.py": (
        "ba20bd915dc498734ee3c273aed1145ce4456b2808d8d32e31c9d96774e46788"
    ),
    "scripts/research/crp_vla/task14r_reset_transaction.py": (
        "40985f05d415a418f18d9eac6243d65a31f9a28f6a870cfa1ecde0585c79cecd"
    ),
}
TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH = (
    "outputs/crp_vla/task14r_reset_transaction_recovery/r0_attempt001/TASK14R_R0_FINAL.json"
)
TASK14R_R0S_PROTOCOL_RELATIVE_PATH = (
    "artifacts/crp_vla/task14r_reset_transaction_recovery/TASK14R_R0S_PROTOCOL.json"
)
TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH = (
    "artifacts/crp_vla/task14r_reset_transaction_recovery/TASK14R_R0S_PROTOCOL.sha256"
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
    {"name": "standard_msaa", "backend": "egl", "offsamples": 4},
    {"name": "no_msaa", "backend": "egl", "offsamples": 0},
)
TASK14R_R0S_REPEATS_PER_RENDERER = 3
TASK14R_R0S_PROBE_STEPS = len(TASK14R_R0_PROBE_ACTIONS)
TASK14R_R0S_FRAMES_PER_TRAJECTORY = TASK14R_R0S_PROBE_STEPS + 1
TASK14R_R0S_CAMERA_KEYS = ("image1", "image2")
TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)
TASK14R_R0S_EXPECTED_TASK_COUNT = 40
TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT = 2
TASK14R_R0S_EXPECTED_RESTORE_TRANSACTION_COUNT = 240
TASK14R_R0S_EXPECTED_PROBE_TRAJECTORY_COUNT = 240
TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT = 3600
TASK14R_R0S_EXPECTED_RENDERER_PAIR_COUNT = 120
TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT = 1920
TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT = 3840
TASK14R_R0S_EXPECTED_WITHIN_RENDERER_PHYSICS_FRAME_COUNT = 2560
TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT = 2560
TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT = 2560
TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_COUNT = 8960
TASK14R_R0S_EXPECTED_TRACE_MANIFEST_COUNT = 240
TASK14R_R0S_EXPECTED_FRAME_RECORD_MANIFEST_COUNT = 240
TASK14R_R0S_EXPECTED_FRAME_PAYLOAD_MANIFEST_COUNT = 240
TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_MANIFEST_COUNT = 120
TASK14R_R0S_EXPECTED_TASK_AUDIT_MANIFEST_COUNT = 40
TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT = 60
TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT = 60
TASK14R_R0S_SOURCE_FILES = (
    "scripts/research/crp_vla/run_task14r_r0s.py",
    "scripts/research/crp_vla/task14r_renderer_shift.py",
    "scripts/research/crp_vla/task14r_reset_transaction.py",
    "scripts/research/crp_vla/launch_task14r_r0s.sh",
)
TASK14R_R0S_QUALIFICATION_BOUNDARY = (
    "R0S is policy-free and compares standard EGL MSAA (offsamples=4) against EGL no-MSAA (offsamples=0).",
    "Only exact cross-render pixels establish exact renderer equivalence.",
    "A nonzero pixel shift is characterized without an outcome-tuned tolerance.",
    "RENDERER_SHIFT_CHARACTERIZED does not authorize R1.",
    "R0S does not establish policy-input or policy-output equivalence.",
    "A separately frozen policy-impact bridge is required before policy execution.",
    "No R0S terminal status automatically authorizes R1, policy execution, or formal outcomes.",
)
TASK14R_R0S_REQUIRED_R0_TERMINAL_GATE_FIELDS = (
    "task_count",
    "passed_task_count",
    "failed_task_count",
    "restore_transaction_count",
    "probe_trajectory_count",
    "probe_step_count",
    "policy_query_count",
    "formal_case_count",
    "formal_outcome_rollout_count",
    "training_or_parameter_updates",
    "automatic_next_phase",
)
TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS = (
    "reference_image_sha256",
    "actual_image_sha256",
    "exact",
    "changed_pixel_count",
    "changed_channel_value_count",
    "maximum_absolute_difference",
    "mean_absolute_difference",
    "per_channel_changed_counts",
    "difference_bounding_box",
    "first_differing_pixel_coordinate",
    "first_differing_channel",
    "reference_pixel_values",
    "actual_pixel_values",
    "reference_channel_value",
    "actual_channel_value",
)


class Task14RRendererShiftError(RuntimeError):
    """A fail-closed R0S error with a stable audit stage."""

    def __init__(
        self,
        failure_stage: str,
        message: str,
        *,
        renderer_arm: str | None = None,
        repeat_index: int | None = None,
        frame_index: int | None = None,
        camera: str | None = None,
        first_mismatched_field: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_stage = str(failure_stage)
        self.renderer_arm = renderer_arm
        self.repeat_index = repeat_index
        self.frame_index = frame_index
        self.camera = camera
        self.first_mismatched_field = first_mismatched_field
        self.details = None if details is None else dict(details)


def renderer_order_for(task_index: int, repeat_index: int) -> tuple[dict[str, Any], ...]:
    """Return the preregistered balanced arm order for one task/repeat pair."""
    if task_index not in range(TASK14R_R0S_EXPECTED_TASK_COUNT):
        raise ValueError("Task14R R0S task_index is outside the frozen 40-task scope")
    if repeat_index not in range(TASK14R_R0S_REPEATS_PER_RENDERER):
        raise ValueError("Task14R R0S repeat_index is outside the frozen three-repeat scope")
    standard_first = (task_index + repeat_index) % 2 == 0
    ordered = TASK14R_R0S_RENDERERS if standard_first else tuple(reversed(TASK14R_R0S_RENDERERS))
    return tuple(dict(renderer) for renderer in ordered)


def _validate_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Task14R R0S invalid SHA-256: {field}")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Task14R R0S invalid SHA-256: {field}")


def build_task14r_r0s_protocol(
    *, implementation_parent_commit: str, source_files_sha256: Mapping[str, str]
) -> dict[str, Any]:
    tasks = []
    for suite_index, suite in enumerate(TASK14R_R0_SUITES):
        for task_id in range(TASK14R_R0_TASKS_PER_SUITE):
            task_index = suite_index * TASK14R_R0_TASKS_PER_SUITE + task_id
            tasks.append(
                {
                    "task_index": task_index,
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": 0,
                    "env_seed": task14r_r0_seed(suite, task_id),
                    "case_id": f"r0s-{suite}-task{task_id:02d}-state00",
                }
            )
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
            "repository_commit": TASK14R_R0S_REQUIRED_R0_REPOSITORY_COMMIT,
            "protocol_sha256": TASK14R_R0S_REQUIRED_R0_PROTOCOL_SHA256,
            "source_files_sha256": dict(sorted(TASK14R_R0S_REQUIRED_R0_SOURCE_FILES_SHA256.items())),
            "libero_commit": TASK14R_R0S_LIBERO_COMMIT,
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
            "terminal_gate_fields": list(TASK14R_R0S_REQUIRED_R0_TERMINAL_GATE_FIELDS),
            "terminal_gate_values": "ALL_TRUE",
        },
        "renderer_contract": {
            "backend": "egl",
            "arms": [dict(row) for row in TASK14R_R0S_RENDERERS],
            "context_rebuilt_before_every_trajectory": True,
            "simulator_or_model_rebuild_after_capsule": False,
            "pixel_gate": "EXACT_OR_CHARACTERIZE_WITHOUT_TOLERANCE",
        },
        "execution_order": {
            "algorithm": "standard_first_if_(task_index+repeat_index)_is_even",
            "even_task": [
                ["standard_msaa", "no_msaa"],
                ["no_msaa", "standard_msaa"],
                ["standard_msaa", "no_msaa"],
            ],
            "odd_task": [
                ["no_msaa", "standard_msaa"],
                ["standard_msaa", "no_msaa"],
                ["no_msaa", "standard_msaa"],
            ],
            "expected_standard_first_pair_count": TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT,
            "expected_no_msaa_first_pair_count": TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT,
            "adaptive_reordering": False,
        },
        "model_difference_contract": {
            "canonical_offsamples": 4,
            "whitelisted_model_fields": ["sim.model.vis.quality.offsamples"],
            "whitelisted_runtime_provenance": ["framebuffer_context", "render_context"],
            "full_mjb_identity_after_whitelist_restore": True,
            "all_numeric_model_arrays_exact": True,
            "generic_fingerprint_normalization_is_sufficient": False,
        },
        "non_image_state_contract": {
            "excluded_render_array_keys": list(TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS),
            "excluded_render_array_values_hash_manifested_per_frame": True,
            "unknown_camera_named_arrays_are_excluded": False,
            "integration_controller_python_contact_action_terminal_success_exact": True,
            "per_frame_model_physics_fingerprint_exact": True,
        },
        "comparison_contract": {
            "equality": "np.array_equal",
            "dtype": "uint8",
            "tolerance": None,
            "post_hoc_thresholds": False,
            "within_renderer_repeat_pairs": [[0, 1], [0, 2]],
            "cross_renderer_repeat_pairs": [[0, 0], [1, 1], [2, 2]],
            "classes": [
                "standard_msaa_within_repeat",
                "no_msaa_within_repeat",
                "standard_msaa_vs_no_msaa",
            ],
            "required_characterization_fields": list(TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS),
            "verdict_depends_on_tolerance": False,
        },
        "artifact_integrity_contract": {
            "trace_rows_per_trajectory": TASK14R_R0S_PROBE_STEPS,
            "frame_records_per_trajectory": TASK14R_R0S_FRAMES_PER_TRAJECTORY,
            "lossless_pixel_frames_per_payload": TASK14R_R0S_FRAMES_PER_TRAJECTORY,
            "frame_records_cross_linked_to_trace_and_payload": True,
            "comparison_rows_recomputed_from_payloads": True,
            "task_and_terminal_revalidation": True,
            "partial_failure_artifacts_required": True,
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
        "expected_renderer_arm_count": TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT,
        "expected_repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
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
        "expected_pixel_comparison_count": TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_COUNT,
        "expected_trace_manifest_count": TASK14R_R0S_EXPECTED_TRACE_MANIFEST_COUNT,
        "expected_frame_record_manifest_count": TASK14R_R0S_EXPECTED_FRAME_RECORD_MANIFEST_COUNT,
        "expected_frame_payload_manifest_count": TASK14R_R0S_EXPECTED_FRAME_PAYLOAD_MANIFEST_COUNT,
        "expected_pixel_comparison_manifest_count": (TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_MANIFEST_COUNT),
        "expected_task_audit_manifest_count": TASK14R_R0S_EXPECTED_TASK_AUDIT_MANIFEST_COUNT,
        "expected_standard_first_pair_count": TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT,
        "expected_no_msaa_first_pair_count": TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT,
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
        "r1_authorized": False,
        "policy_impact_bridge_required": True,
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
        "repository_commit": TASK14R_R0S_REQUIRED_R0_REPOSITORY_COMMIT,
        "protocol_sha256": TASK14R_R0S_REQUIRED_R0_PROTOCOL_SHA256,
        "source_files_sha256": dict(sorted(TASK14R_R0S_REQUIRED_R0_SOURCE_FILES_SHA256.items())),
        "libero_commit": TASK14R_R0S_LIBERO_COMMIT,
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
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != set(TASK14R_R0S_REQUIRED_R0_TERMINAL_GATE_FIELDS)
        or not all(terminal.values())
    ):
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
        "render_context_provenance": {
            "class": f"{context.__class__.__module__}.{context.__class__.__qualname__}",
            "context_object_identity": int(id(context)),
        },
    }


def renderer_model_whitelist_audit(
    sim: Any,
    *,
    canonical_model_fingerprint: Mapping[str, Any],
    expected_offsamples: int,
) -> dict[str, Any]:
    """Prove that restoring the sole model whitelist field recovers the full canonical model."""
    actual_offsamples = int(sim.model.vis.quality.offsamples)
    if actual_offsamples != int(expected_offsamples):
        raise ValueError("Task14R R0S renderer offsamples drift before model whitelist audit")
    actual = mujoco_model_fingerprint(sim)
    try:
        sim.model.vis.quality.offsamples = 4
        whitelist_restored = mujoco_model_fingerprint(sim)
    finally:
        sim.model.vis.quality.offsamples = actual_offsamples
    canonical_arrays = canonical_model_fingerprint.get("arrays")
    actual_arrays = actual.get("arrays")
    restored_arrays = whitelist_restored.get("arrays")
    checks = {
        "offsamples_is_expected": actual_offsamples == int(expected_offsamples),
        "numeric_model_array_inventory_identity": (
            isinstance(canonical_arrays, Mapping)
            and isinstance(actual_arrays, Mapping)
            and set(actual_arrays) == set(canonical_arrays)
        ),
        "numeric_model_arrays_exact": actual_arrays == canonical_arrays,
        "full_mjb_identity_after_whitelist_restore": (
            whitelist_restored.get("mjb_sha256") == canonical_model_fingerprint.get("mjb_sha256")
        ),
        "full_model_identity_after_whitelist_restore": (
            whitelist_restored.get("complete_sha256") == canonical_model_fingerprint.get("complete_sha256")
        ),
        "restored_numeric_model_arrays_exact": restored_arrays == canonical_arrays,
    }
    checks["only_whitelisted_model_difference"] = all(checks.values())
    return {
        "whitelisted_model_fields": {
            "sim.model.vis.quality.offsamples": {
                "canonical": 4,
                "actual": actual_offsamples,
            }
        },
        "actual_model_complete_sha256": actual.get("complete_sha256"),
        "actual_model_mjb_sha256": actual.get("mjb_sha256"),
        "physics_model_fingerprint_sha256": whitelist_restored.get("complete_sha256"),
        "whitelist_restored_model_complete_sha256": whitelist_restored.get("complete_sha256"),
        "whitelist_restored_model_mjb_sha256": whitelist_restored.get("mjb_sha256"),
        **checks,
    }


def _is_rendered_entry(name: Any, value: Any) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim >= 2
        and value.size > 0
        and str(name) in TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS
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


def rendered_python_state_manifest(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every rendered cache value excluded from the non-image identity gate."""
    entries: dict[str, dict[str, Any]] = {}
    cache = snapshot.get("obs_cache")
    if isinstance(cache, Mapping):
        for name, child in cache.items():
            if _is_rendered_entry(name, child):
                array = np.ascontiguousarray(child)
                entries[f"obs_cache.{name}"] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": array_sha256(array),
                }
    observables = snapshot.get("observables")
    if isinstance(observables, Mapping):
        for name, observable in observables.items():
            if not isinstance(observable, Mapping):
                continue
            child = observable.get("_current_observed_value")
            if _is_rendered_entry(name, child):
                array = np.ascontiguousarray(child)
                entries[f"observables.{name}._current_observed_value"] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": array_sha256(array),
                }
    return {
        "allowed_rendered_keys": list(TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS),
        "entry_count": len(entries),
        "entries": dict(sorted(entries.items())),
    }


def split_observation(observation: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping) or set(pixels) != set(TASK14R_R0S_CAMERA_KEYS):
        raise ValueError("Task14R R0S observation camera inventory drift")
    images = {str(key): np.ascontiguousarray(np.asarray(pixels[key])).copy() for key in sorted(pixels)}
    for key, image in images.items():
        if image.ndim != 3 or image.size == 0 or image.dtype != np.uint8 or image.shape[-1] != 3:
            raise ValueError(f"Task14R R0S invalid camera observation: {key}")
    if "robot_state" not in observation:
        raise ValueError("Task14R R0S robot_state is missing")
    return images, observation["robot_state"]


def pixel_difference(left: Any, right: Any) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ValueError("Task14R R0S camera shape or dtype drift")
    if left_array.dtype != np.uint8:
        raise ValueError("Task14R R0S camera arrays must be raw uint8")
    if left_array.ndim != 3 or left_array.shape[-1] != 3:
        raise ValueError("Task14R R0S camera arrays must be raw HWC uint8 RGB")
    channel_axis = 2
    left_hwc = left_array
    right_hwc = right_array
    signed_difference = left_hwc.astype(np.int16) - right_hwc.astype(np.int16)
    difference = np.abs(signed_difference).astype(np.float64)
    changed_channel_mask = difference != 0.0
    changed_pixel_mask = np.any(changed_channel_mask, axis=2)
    changed_coordinates = np.argwhere(changed_channel_mask)
    changed_pixels = np.argwhere(changed_pixel_mask)
    if changed_coordinates.size:
        first_row, first_column, first_channel = (int(value) for value in changed_coordinates[0])
        first_pixel_coordinate: dict[str, int] | None = {
            "row": first_row,
            "column": first_column,
        }
        reference_pixel_values: list[int] | None = [
            int(value) for value in left_hwc[first_row, first_column].tolist()
        ]
        actual_pixel_values: list[int] | None = [
            int(value) for value in right_hwc[first_row, first_column].tolist()
        ]
        reference_channel_value: int | None = int(left_hwc[first_row, first_column, first_channel])
        actual_channel_value: int | None = int(right_hwc[first_row, first_column, first_channel])
    else:
        first_pixel_coordinate = None
        first_channel = None
        reference_pixel_values = None
        actual_pixel_values = None
        reference_channel_value = None
        actual_channel_value = None
    if changed_pixels.size:
        minimum = changed_pixels.min(axis=0)
        maximum = changed_pixels.max(axis=0)
        difference_bounding_box: dict[str, int] | None = {
            "minimum_row": int(minimum[0]),
            "maximum_row": int(maximum[0]),
            "minimum_column": int(minimum[1]),
            "maximum_column": int(maximum[1]),
        }
    else:
        difference_bounding_box = None
    value_count = int(difference.size)
    changed_channel_value_count = int(np.count_nonzero(changed_channel_mask))
    return {
        "shape": list(left_array.shape),
        "dtype": str(left_array.dtype),
        "channel_axis": channel_axis,
        "reference_image_sha256": array_sha256(np.ascontiguousarray(left_array)),
        "actual_image_sha256": array_sha256(np.ascontiguousarray(right_array)),
        "exact": bool(np.array_equal(left_array, right_array)),
        "value_count": value_count,
        "changed_channel_value_count": changed_channel_value_count,
        "pixel_count": int(changed_pixel_mask.size),
        "changed_pixel_count": int(np.count_nonzero(changed_pixel_mask)),
        "maximum_absolute_difference": float(np.max(difference, initial=0.0)),
        "mean_absolute_difference": (
            0.0 if value_count == 0 else float(np.sum(difference, dtype=np.float64) / value_count)
        ),
        "per_channel_changed_counts": [
            int(np.count_nonzero(changed_channel_mask[:, :, channel]))
            for channel in range(changed_channel_mask.shape[2])
        ],
        "difference_bounding_box": difference_bounding_box,
        "first_differing_pixel_coordinate": first_pixel_coordinate,
        "first_differing_channel": first_channel,
        "reference_pixel_values": reference_pixel_values,
        "actual_pixel_values": actual_pixel_values,
        "reference_channel_value": reference_channel_value,
        "actual_channel_value": actual_channel_value,
        "sum_absolute_difference": float(np.sum(difference, dtype=np.float64)),
        "sum_squared_difference": float(np.sum(np.square(difference), dtype=np.float64)),
    }


def aggregate_pixel_differences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(not all(field in row for field in TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS) for row in rows):
        raise ValueError("Task14R R0S pixel characterization field inventory drift")
    value_count = sum(int(row["value_count"]) for row in rows)
    pixel_count = sum(int(row["pixel_count"]) for row in rows)
    sum_absolute = sum(float(row["sum_absolute_difference"]) for row in rows)
    sum_squared = sum(float(row["sum_squared_difference"]) for row in rows)
    channel_counts = [list(row["per_channel_changed_counts"]) for row in rows]
    channel_count = 0 if not channel_counts else len(channel_counts[0])
    if any(len(counts) != channel_count for counts in channel_counts):
        raise ValueError("Task14R R0S per-channel metric inventory drift")
    first_shift = next((dict(row) for row in rows if not bool(row["exact"])), None)
    return {
        "comparison_count": len(rows),
        "exact_comparison_count": sum(bool(row["exact"]) for row in rows),
        "shifted_comparison_count": sum(not bool(row["exact"]) for row in rows),
        "value_count": value_count,
        "changed_channel_value_count": sum(int(row["changed_channel_value_count"]) for row in rows),
        "pixel_count": pixel_count,
        "changed_pixel_count": sum(int(row["changed_pixel_count"]) for row in rows),
        "maximum_absolute_difference": max(
            (float(row["maximum_absolute_difference"]) for row in rows), default=0.0
        ),
        "mean_absolute_difference": 0.0 if value_count == 0 else sum_absolute / value_count,
        "per_channel_changed_counts": [
            sum(int(counts[channel]) for counts in channel_counts) for channel in range(channel_count)
        ],
        "root_mean_squared_difference": (
            0.0 if value_count == 0 else float(np.sqrt(sum_squared / value_count))
        ),
        "exact": all(bool(row["exact"]) for row in rows),
        "characterization_complete": True,
        "first_shift": first_shift,
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
    cross_renderer_camera_comparisons = total("cross_renderer_camera_comparison_count")
    within_physics = total("within_renderer_physics_frame_count")
    within_physics_identity = total("within_renderer_physics_identity_frame_count")
    standard_comparisons = total("standard_repeat_camera_comparison_count")
    no_msaa_comparisons = total("no_msaa_repeat_camera_comparison_count")
    no_msaa_exact = total("no_msaa_repeat_exact_camera_comparison_count")
    standard_exact = total("standard_repeat_exact_camera_comparison_count")
    cross_exact = total("cross_renderer_exact_camera_comparison_count")
    standard_first = total("standard_first_pair_count")
    no_msaa_first = total("no_msaa_first_pair_count")
    trace_manifests = total("trace_manifest_count")
    frame_record_manifests = total("frame_record_manifest_count")
    frame_payload_manifests = total("frame_payload_manifest_count")
    comparison_manifests = total("pixel_comparison_manifest_count")
    task_audit_manifests = total("task_audit_manifest_count")
    pixel_comparisons = standard_comparisons + no_msaa_comparisons + camera_comparisons
    policy_queries = total("policy_query_count")
    formal_cases = total("formal_case_count")
    formal_outcomes = total("formal_outcome_rollout_count")
    training_updates = any(bool(row.get("training_or_parameter_updates")) for row in task_audits)
    automatic_next = any(bool(row.get("automatic_next_phase")) for row in task_audits)

    def execution_order_exact(row: Mapping[str, Any]) -> bool:
        case = row.get("case")
        if not isinstance(case, Mapping):
            return False
        try:
            task_index = int(case["task_index"])
            expected = [
                {
                    "task_index": task_index,
                    "repeat_index": repeat_index,
                    "renderer_order": [
                        str(renderer["name"]) for renderer in renderer_order_for(task_index, repeat_index)
                    ],
                }
                for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER)
            ]
        except (KeyError, TypeError, ValueError):
            return False
        return row.get("execution_order") == expected

    per_task_counts_exact = all(
        int(row.get("renderer_arm_count", 0)) == TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT
        and int(row.get("repeats_per_renderer", 0)) == TASK14R_R0S_REPEATS_PER_RENDERER
        and int(row.get("restore_transaction_count", 0)) == 6
        and int(row.get("probe_trajectory_count", 0)) == 6
        and int(row.get("probe_step_count", 0)) == 90
        and int(row.get("renderer_pair_count", 0)) == 3
        and int(row.get("paired_frame_count", 0)) == 48
        and int(row.get("physics_identity_frame_count", 0)) == 48
        and int(row.get("non_image_identity_frame_count", 0)) == 48
        and int(row.get("camera_comparison_count", 0)) == 96
        and int(row.get("cross_renderer_camera_comparison_count", 0)) == 96
        and int(row.get("within_renderer_physics_frame_count", 0)) == 64
        and int(row.get("within_renderer_physics_identity_frame_count", 0)) == 64
        and int(row.get("standard_repeat_camera_comparison_count", 0)) == 64
        and int(row.get("no_msaa_repeat_camera_comparison_count", 0)) == 64
        and int(row.get("standard_first_pair_count", 0)) + int(row.get("no_msaa_first_pair_count", 0)) == 3
        and int(row.get("trace_manifest_count", 0)) == 6
        and int(row.get("frame_record_manifest_count", 0)) == 6
        and int(row.get("frame_payload_manifest_count", 0)) == 6
        and int(row.get("pixel_comparison_manifest_count", 0)) == 3
        and int(row.get("task_audit_manifest_count", 0)) == 1
        for row in task_audits
    )
    gate = {
        "task_count": task_count == TASK14R_R0S_EXPECTED_TASK_COUNT,
        "passed_task_count": passed == TASK14R_R0S_EXPECTED_TASK_COUNT,
        "failed_task_count": failed == 0,
        "renderer_arm_count": all(
            int(row.get("renderer_arm_count", 0)) == TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT
            for row in task_audits
        ),
        "repeats_per_renderer": all(
            int(row.get("repeats_per_renderer", 0)) == TASK14R_R0S_REPEATS_PER_RENDERER for row in task_audits
        ),
        "per_task_counts_exact": per_task_counts_exact,
        "execution_order_frozen": all(execution_order_exact(row) for row in task_audits),
        "restore_transaction_count": restore_count == TASK14R_R0S_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "probe_trajectory_count": trajectory_count == TASK14R_R0S_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "probe_step_count": step_count == TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT,
        "renderer_pair_count": pair_count == TASK14R_R0S_EXPECTED_RENDERER_PAIR_COUNT,
        "paired_frame_count": paired_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "physics_identity_frame_count": physics_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "non_image_identity_frame_count": non_image_frames == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
        "camera_comparison_count": camera_comparisons == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
        "cross_renderer_camera_comparison_count": cross_renderer_camera_comparisons
        == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
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
        "standard_first_pair_count": standard_first == TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT,
        "no_msaa_first_pair_count": no_msaa_first == TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT,
        "trace_manifest_count": trace_manifests == TASK14R_R0S_EXPECTED_TRACE_MANIFEST_COUNT,
        "frame_record_manifest_count": frame_record_manifests
        == TASK14R_R0S_EXPECTED_FRAME_RECORD_MANIFEST_COUNT,
        "frame_payload_manifest_count": frame_payload_manifests
        == TASK14R_R0S_EXPECTED_FRAME_PAYLOAD_MANIFEST_COUNT,
        "pixel_comparison_manifest_count": comparison_manifests
        == TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_MANIFEST_COUNT,
        "task_audit_manifest_count": task_audit_manifests == TASK14R_R0S_EXPECTED_TASK_AUDIT_MANIFEST_COUNT,
        "pixel_comparison_count": pixel_comparisons == TASK14R_R0S_EXPECTED_PIXEL_COMPARISON_COUNT,
        "artifact_integrity": all(bool(row.get("artifact_integrity_passed")) for row in task_audits),
        "task_audit_integrity": all(bool(row.get("task_audit_integrity_passed")) for row in task_audits),
        "model_renderer_whitelist": all(bool(row.get("model_whitelist_only")) for row in task_audits),
        "non_image_exact": all(bool(row.get("non_image_exact")) for row in task_audits),
        "no_msaa_repeat_pixels_exact": all(
            bool(row.get("no_msaa_repeat_pixels_exact")) for row in task_audits
        ),
        "pixel_characterization_complete": all(
            bool(row.get("pixel_characterization_complete")) for row in task_audits
        ),
        "policy_query_count": policy_queries == 0,
        "formal_case_count": formal_cases == 0,
        "formal_outcome_rollout_count": formal_outcomes == 0,
        "training_or_parameter_updates": not training_updates,
        "automatic_next_phase": not automatic_next,
    }
    technical_pass = all(gate.values())
    exact_pixels = (
        technical_pass
        and standard_exact == TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT
        and no_msaa_exact == TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT
        and cross_exact == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT
    )
    characterized_shift = technical_pass and (
        standard_exact < TASK14R_R0S_EXPECTED_STANDARD_REPEAT_CAMERA_COMPARISON_COUNT
        or cross_exact < TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT
    )
    if not technical_pass:
        status = TASK14R_R0S_STATUS_FAILED
    elif exact_pixels:
        status = TASK14R_R0S_STATUS_EXACT
    elif characterized_shift:
        status = TASK14R_R0S_STATUS_SHIFT
    else:
        status = TASK14R_R0S_STATUS_FAILED
    return {
        "status": status,
        "task_count": task_count,
        "passed_task_count": passed,
        "failed_task_count": failed,
        "restore_transaction_count": restore_count,
        "probe_trajectory_count": trajectory_count,
        "probe_step_count": step_count,
        "renderer_pair_count": pair_count,
        "renderer_arm_count": TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT,
        "repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
        "paired_frame_count": paired_frames,
        "physics_identity_frame_count": physics_frames,
        "non_image_identity_frame_count": non_image_frames,
        "camera_comparison_count": camera_comparisons,
        "cross_renderer_camera_comparison_count": cross_renderer_camera_comparisons,
        "within_renderer_physics_frame_count": within_physics,
        "within_renderer_physics_identity_frame_count": within_physics_identity,
        "standard_repeat_camera_comparison_count": standard_comparisons,
        "standard_repeat_exact_camera_comparison_count": standard_exact,
        "no_msaa_repeat_camera_comparison_count": no_msaa_comparisons,
        "no_msaa_repeat_exact_camera_comparison_count": no_msaa_exact,
        "cross_renderer_exact_camera_comparison_count": cross_exact,
        "standard_first_pair_count": standard_first,
        "no_msaa_first_pair_count": no_msaa_first,
        "trace_manifest_count": trace_manifests,
        "frame_record_manifest_count": frame_record_manifests,
        "frame_payload_manifest_count": frame_payload_manifests,
        "pixel_comparison_manifest_count": comparison_manifests,
        "task_audit_manifest_count": task_audit_manifests,
        "pixel_comparison_count": pixel_comparisons,
        "standard_msaa_within_variation_task_count": sum(
            not bool(row.get("standard_repeat_pixels_exact")) for row in task_audits
        ),
        "cross_renderer_variation_task_count": sum(
            not bool(row.get("cross_renderer_pixels_exact")) for row in task_audits
        ),
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
        "r1_authorized": False,
        "policy_impact_bridge_required": True,
        "terminal_gate_checks": gate,
        "qualification_boundary": list(TASK14R_R0S_QUALIFICATION_BOUNDARY),
    }
