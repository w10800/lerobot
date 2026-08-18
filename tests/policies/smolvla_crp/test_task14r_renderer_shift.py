from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research.crp_vla.task14r_renderer_shift import (
    TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
    TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT,
    TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
    TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT,
    TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0S_REQUIRED_R0_FINAL_SHA256,
    TASK14R_R0S_SOURCE_FILES,
    TASK14R_R0S_STATUS_EXACT,
    TASK14R_R0S_STATUS_FAILED,
    TASK14R_R0S_STATUS_SHIFT,
    aggregate_pixel_differences,
    build_task14r_r0s_protocol,
    configure_renderer,
    pixel_difference,
    renderer_invariant_python_state,
    task14r_r0s_terminal_summary,
    validate_required_r0_final,
    validate_task14r_r0s_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts" / "research" / "crp_vla"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_task14r_r0s import execute_task_with_audit, write_frame_payload  # noqa: E402


def _source_hashes(fill: str = "a") -> dict[str, str]:
    return dict.fromkeys(TASK14R_R0S_SOURCE_FILES, fill * 64)


def _protocol() -> dict[str, object]:
    return build_task14r_r0s_protocol(
        implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=_source_hashes(),
    )


def _validate(protocol: dict[str, object]) -> None:
    validate_task14r_r0s_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=_source_hashes(),
    )


def test_r0s_protocol_freezes_policy_free_two_renderer_full_scope() -> None:
    protocol = _protocol()
    _validate(protocol)

    assert protocol["task_count"] == 40
    assert len(protocol["tasks"]) == 40
    assert {row["init_state_id"] for row in protocol["tasks"]} == {0}
    assert protocol["renderer_contract"]["arms"] == [
        {"name": "standard_msaa", "offsamples": 4},
        {"name": "no_msaa", "offsamples": 0},
    ]
    assert protocol["repeats_per_renderer"] == 3
    assert protocol["probe_steps_per_trajectory"] == 15
    assert protocol["frames_per_trajectory"] == 16
    assert protocol["camera_keys"] == ["image1", "image2"]
    assert protocol["expected_restore_transaction_count"] == 240
    assert protocol["expected_probe_trajectory_count"] == 240
    assert protocol["expected_probe_step_count"] == 3600
    assert protocol["expected_renderer_pair_count"] == 120
    assert protocol["expected_paired_frame_count"] == 1920
    assert protocol["expected_camera_comparison_count"] == 3840
    assert protocol["policy_query_count"] == 0
    assert protocol["formal_case_count"] == 0
    assert protocol["formal_outcome_rollout_count"] == 0
    assert not protocol["training_or_parameter_updates"]
    assert not protocol["automatic_next_phase"]


def test_r0s_protocol_rejects_parent_source_renderer_counts_and_terminal_drift() -> None:
    mutations = (
        ("implementation_parent_commit", "wrong"),
        ("renderer_contract", {}),
        ("expected_probe_step_count", 3599),
        ("permitted_terminal_statuses", [TASK14R_R0S_STATUS_EXACT]),
    )
    for field, value in mutations:
        protocol = _protocol()
        protocol[field] = value
        with pytest.raises(ValueError, match="field drift"):
            _validate(protocol)

    protocol = _protocol()
    protocol["source_files_sha256"][TASK14R_R0S_SOURCE_FILES[0]] = "b" * 64
    with pytest.raises(ValueError, match="source_files_sha256"):
        _validate(protocol)


def _r0_final() -> dict[str, object]:
    gate = {
        "task_count": True,
        "passed_task_count": True,
        "failed_task_count": True,
        "restore_transaction_count": True,
        "probe_trajectory_count": True,
        "probe_step_count": True,
        "policy_query_count": True,
        "formal_case_count": True,
        "formal_outcome_rollout_count": True,
        "training_or_parameter_updates": True,
        "automatic_next_phase": True,
    }
    return {
        "status": "TASK14R_R0_RESTORE_TRANSACTION_PASSED",
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
        "terminal_gate_checks": gate,
    }


def test_r0s_requires_the_exact_passed_r0_artifact_and_all_terminal_gates() -> None:
    validate_required_r0_final(_r0_final(), actual_sha256=TASK14R_R0S_REQUIRED_R0_FINAL_SHA256)
    drift = _r0_final()
    drift["probe_step_count"] = 1799
    with pytest.raises(ValueError, match="probe_step_count"):
        validate_required_r0_final(drift, actual_sha256=TASK14R_R0S_REQUIRED_R0_FINAL_SHA256)
    gate_drift = _r0_final()
    gate_drift["terminal_gate_checks"]["task_count"] = False
    with pytest.raises(ValueError, match="every R0 terminal gate"):
        validate_required_r0_final(gate_drift, actual_sha256=TASK14R_R0S_REQUIRED_R0_FINAL_SHA256)
    with pytest.raises(ValueError, match="SHA-256 drift"):
        validate_required_r0_final(_r0_final(), actual_sha256="0" * 64)


def test_renderer_configuration_rebuilds_only_framebuffer_for_four_and_zero_samples() -> None:
    class Context:
        def __init__(self) -> None:
            self.free_count = 0
            self.rebuild_count = 0
            self.con = SimpleNamespace(free=self.free)

        def free(self) -> None:
            self.free_count += 1

        def _set_mujoco_context_and_buffers(self) -> None:
            self.rebuild_count += 1

    context = Context()
    sim = SimpleNamespace(
        model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4))),
        _render_context_offscreen=context,
        forward=lambda: None,
    )
    env = SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(sim=sim)))
    first = configure_renderer(env, offsamples=4)
    second = configure_renderer(env, offsamples=0)
    assert first["offsamples_after"] == 4
    assert second["offsamples_before"] == 4
    assert second["offsamples_after"] == 0
    assert context.free_count == 2
    assert context.rebuild_count == 2
    assert not second["simulator_or_model_rebuilt"]
    with pytest.raises(ValueError, match="0 or 4"):
        configure_renderer(env, offsamples=2)


def test_renderer_invariant_python_state_drops_only_named_camera_values() -> None:
    camera = np.zeros((4, 4, 3), dtype=np.uint8)
    eef_matrix = np.eye(3)
    snapshot = {
        "obs_cache": {
            "agentview_image": camera,
            "robot0_eef_mat": eef_matrix,
            "robot0_joint_pos": np.arange(7),
        },
        "observables": {
            "agentview_image": {
                "_current_observed_value": camera,
                "_time_since_last_sample": 0.0,
            },
            "robot0_eef_mat": {
                "_current_observed_value": eef_matrix,
                "_time_since_last_sample": 0.0,
            },
        },
    }
    invariant = renderer_invariant_python_state(snapshot)
    assert "agentview_image" not in invariant["obs_cache"]
    assert np.array_equal(invariant["obs_cache"]["robot0_eef_mat"], eef_matrix)
    assert "_current_observed_value" not in invariant["observables"]["agentview_image"]
    assert np.array_equal(invariant["observables"]["robot0_eef_mat"]["_current_observed_value"], eef_matrix)
    assert "_current_observed_value" in snapshot["observables"]["agentview_image"]


def test_pixel_difference_is_exact_or_descriptive_without_a_tolerance() -> None:
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    exact = pixel_difference(left, left.copy())
    assert exact["exact"]
    assert exact["changed_value_count"] == 0
    right = left.copy()
    right[1, 0, 2] = 3
    shifted = pixel_difference(left, right)
    assert not shifted["exact"]
    assert shifted["changed_value_count"] == 1
    assert shifted["changed_pixel_count"] == 1
    assert shifted["maximum_absolute_difference"] == 3.0
    summary = aggregate_pixel_differences([exact, shifted])
    assert summary["comparison_count"] == 2
    assert summary["exact_comparison_count"] == 1
    assert summary["shifted_comparison_count"] == 1
    assert not summary["exact"]


def test_frame_payload_is_lossless_and_hash_manifested(tmp_path: Path) -> None:
    frames = [
        {
            "pixels": {
                "image1": np.full((2, 2, 3), index, dtype=np.uint8),
                "image2": np.full((2, 2, 3), index + 1, dtype=np.uint8),
            }
        }
        for index in range(3)
    ]
    path = tmp_path / "frames.npz"
    manifest = write_frame_payload(path, frames)
    assert manifest["frame_count"] == 3
    assert len(manifest["sha256"]) == 64
    with np.load(path) as payload:
        assert payload["image1"].shape == (3, 2, 2, 3)
        assert payload["image2"][2, 0, 0, 0] == 3


def _passed_tasks() -> list[dict[str, object]]:
    return [
        {
            "passed": True,
            "restore_transaction_count": 6,
            "probe_trajectory_count": 6,
            "probe_step_count": 90,
            "renderer_pair_count": 3,
            "paired_frame_count": 48,
            "physics_identity_frame_count": 48,
            "non_image_identity_frame_count": 48,
            "camera_comparison_count": 96,
            "within_renderer_physics_frame_count": 64,
            "within_renderer_physics_identity_frame_count": 64,
            "standard_repeat_camera_comparison_count": 64,
            "no_msaa_repeat_camera_comparison_count": 64,
            "no_msaa_repeat_exact_camera_comparison_count": 64,
            "cross_renderer_pixels_exact": True,
            "standard_repeat_pixels_exact": True,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
        for _ in range(40)
    ]


def test_terminal_status_separates_exact_shift_and_engineering_failure() -> None:
    exact = task14r_r0s_terminal_summary(_passed_tasks())
    assert exact["status"] == TASK14R_R0S_STATUS_EXACT
    assert exact["probe_step_count"] == TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT
    assert exact["paired_frame_count"] == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT
    assert exact["camera_comparison_count"] == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT
    assert all(exact["terminal_gate_checks"].values())

    shifted_tasks = _passed_tasks()
    shifted_tasks[0]["cross_renderer_pixels_exact"] = False
    shifted = task14r_r0s_terminal_summary(shifted_tasks)
    assert shifted["status"] == TASK14R_R0S_STATUS_SHIFT
    assert all(shifted["terminal_gate_checks"].values())
    assert shifted["pixel_shift_task_count"] == 1

    failed_tasks = _passed_tasks()
    failed_tasks[0]["no_msaa_repeat_exact_camera_comparison_count"] = 63
    failed = task14r_r0s_terminal_summary(failed_tasks)
    assert failed["status"] == TASK14R_R0S_STATUS_FAILED
    assert not failed["terminal_gate_checks"]["no_msaa_repeat_exact_camera_comparison_count"]
    assert failed["no_msaa_repeat_camera_comparison_count"] == (
        TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT
    )


def test_task_exception_writes_failure_audit_without_policy_or_formal_activity(
    tmp_path: Path,
) -> None:
    case = {"case_id": "r0s-first", "suite": "libero_spatial", "task_id": 0}
    env = SimpleNamespace(close=lambda: None)

    def fail(_env: object, _case: dict[str, object], _root: Path) -> dict[str, object]:
        raise RuntimeError("injected")

    audit = execute_task_with_audit(
        case=case,
        task_root=tmp_path / "r0s-first",
        environment_factory=lambda: env,
        task_runner=fail,
    )
    assert not audit["passed"]
    assert audit["failure_stage"] == "environment_creation"
    assert audit["policy_query_count"] == 0
    assert audit["formal_outcome_rollout_count"] == 0
    assert not audit["training_or_parameter_updates"]
    assert (tmp_path / "r0s-first" / "TASK_AUDIT_FAILURE.json").is_file()


def test_r0s_runner_has_one_canonical_reset_and_no_model_policy_or_later_phase_entrypoint() -> None:
    runner = SCRIPT_DIRECTORY / "run_task14r_r0s.py"
    source = runner.read_text()
    tree = ast.parse(source)
    reset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
    ]
    assert len(reset_calls) == 1
    for forbidden in (
        "SmolVLA",
        "predict_action_chunk",
        "load_checkpoint",
        "optimizer",
        "Phase B",
        "Phase C",
        "Task14C",
    ):
        assert forbidden not in source


def test_protocol_copy_does_not_mutate_when_validation_runs() -> None:
    protocol = _protocol()
    before = copy.deepcopy(protocol)
    _validate(protocol)
    assert protocol == before
