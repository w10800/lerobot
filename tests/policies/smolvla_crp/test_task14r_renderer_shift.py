from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.research.crp_vla.task14r_renderer_shift as renderer_shift
from scripts.research.crp_vla.task14r_renderer_shift import (
    TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT,
    TASK14R_R0S_EXPECTED_FRAME_PAYLOAD_MANIFEST_COUNT,
    TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT,
    TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT,
    TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT,
    TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT,
    TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT,
    TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT,
    TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0S_LIBERO_COMMIT,
    TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS,
    TASK14R_R0S_PROTOCOL_RELATIVE_PATH,
    TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH,
    TASK14R_R0S_REQUIRED_R0_FINAL_SHA256,
    TASK14R_R0S_REQUIRED_R0_PROTOCOL_SHA256,
    TASK14R_R0S_REQUIRED_R0_REPOSITORY_COMMIT,
    TASK14R_R0S_REQUIRED_R0_SOURCE_FILES_SHA256,
    TASK14R_R0S_SCHEMA_VERSION,
    TASK14R_R0S_SOURCE_FILES,
    TASK14R_R0S_STATUS_EXACT,
    TASK14R_R0S_STATUS_FAILED,
    TASK14R_R0S_STATUS_SHIFT,
    Task14RRendererShiftError,
    aggregate_pixel_differences,
    build_task14r_r0s_protocol,
    configure_renderer,
    pixel_difference,
    rendered_python_state_manifest,
    renderer_invariant_python_state,
    renderer_model_whitelist_audit,
    renderer_order_for,
    task14r_r0s_terminal_summary,
    validate_required_r0_final,
    validate_task14r_r0s_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts" / "research" / "crp_vla"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_task14r_r0s import (  # noqa: E402
    compare_pixel_frames,
    execute_task_with_audit,
    source_files_sha256,
    validate_pre_simulator_gate,
    verify_task_artifacts,
    write_frame_payload,
    write_trace,
)


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

    assert protocol["schema_version"] == TASK14R_R0S_SCHEMA_VERSION
    assert protocol["task_count"] == 40
    assert len(protocol["tasks"]) == 40
    assert {row["init_state_id"] for row in protocol["tasks"]} == {0}
    assert protocol["renderer_contract"]["arms"] == [
        {"name": "standard_msaa", "backend": "egl", "offsamples": 4},
        {"name": "no_msaa", "backend": "egl", "offsamples": 0},
    ]
    assert protocol["expected_renderer_arm_count"] == TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT
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
    assert protocol["expected_pixel_comparison_count"] == 8960
    assert protocol["expected_trace_manifest_count"] == 240
    assert protocol["expected_frame_record_manifest_count"] == 240
    assert protocol["expected_frame_payload_manifest_count"] == 240
    assert protocol["expected_pixel_comparison_manifest_count"] == 120
    assert protocol["expected_task_audit_manifest_count"] == 40
    assert protocol["execution_order"]["expected_standard_first_pair_count"] == 60
    assert protocol["execution_order"]["expected_no_msaa_first_pair_count"] == 60
    assert protocol["comparison_contract"]["equality"] == "np.array_equal"
    assert protocol["comparison_contract"]["dtype"] == "uint8"
    assert protocol["comparison_contract"]["tolerance"] is None
    assert not protocol["comparison_contract"]["post_hoc_thresholds"]
    assert protocol["model_difference_contract"]["whitelisted_model_fields"] == [
        "sim.model.vis.quality.offsamples"
    ]
    assert protocol["policy_query_count"] == 0
    assert protocol["formal_case_count"] == 0
    assert protocol["formal_outcome_rollout_count"] == 0
    assert not protocol["training_or_parameter_updates"]
    assert not protocol["automatic_next_phase"]
    assert not protocol["r1_authorized"]
    assert protocol["policy_impact_bridge_required"]


def test_renderer_order_is_deterministic_and_exactly_balanced() -> None:
    standard_first = 0
    no_msaa_first = 0
    for task_index in range(40):
        observed = []
        for repeat_index in range(3):
            names = [row["name"] for row in renderer_order_for(task_index, repeat_index)]
            observed.append(names)
            standard_first += names[0] == "standard_msaa"
            no_msaa_first += names[0] == "no_msaa"
        expected = (
            [
                ["standard_msaa", "no_msaa"],
                ["no_msaa", "standard_msaa"],
                ["standard_msaa", "no_msaa"],
            ]
            if task_index % 2 == 0
            else [
                ["no_msaa", "standard_msaa"],
                ["standard_msaa", "no_msaa"],
                ["no_msaa", "standard_msaa"],
            ]
        )
        assert observed == expected
    assert standard_first == TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT
    assert no_msaa_first == TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT


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
        "repository_commit": TASK14R_R0S_REQUIRED_R0_REPOSITORY_COMMIT,
        "protocol_sha256": TASK14R_R0S_REQUIRED_R0_PROTOCOL_SHA256,
        "source_files_sha256": dict(TASK14R_R0S_REQUIRED_R0_SOURCE_FILES_SHA256),
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
    for field, value in (
        ("status", "TASK14R_R0_RESTORE_TRANSACTION_FAILED"),
        ("protocol_sha256", "0" * 64),
        ("source_files_sha256", {}),
        ("libero_commit", "0" * 40),
    ):
        drift = _r0_final()
        drift[field] = value
        with pytest.raises(ValueError, match=str(field)):
            validate_required_r0_final(drift, actual_sha256=TASK14R_R0S_REQUIRED_R0_FINAL_SHA256)


def test_missing_r0_is_rejected_without_creating_output(tmp_path: Path) -> None:
    for relative in TASK14R_R0S_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen:{relative}\n")
    hashes = source_files_sha256(tmp_path)
    protocol = build_task14r_r0s_protocol(
        implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=hashes,
    )
    with pytest.raises(FileNotFoundError, match="Required Task14R R0 final is missing"):
        validate_pre_simulator_gate(protocol, tmp_path)
    assert not (tmp_path / "outputs").exists()


def test_protocol_digest_drift_is_rejected_before_r0_or_output(tmp_path: Path) -> None:
    for relative in TASK14R_R0S_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen:{relative}\n")
    hashes = source_files_sha256(tmp_path)
    protocol = build_task14r_r0s_protocol(
        implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=hashes,
    )
    protocol_path = tmp_path / TASK14R_R0S_PROTOCOL_RELATIVE_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    digest_path = tmp_path / TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH
    digest_path.write_text(f"{'0' * 64}  {protocol_path.name}\n")
    with pytest.raises(ValueError, match="protocol SHA-256 drift"):
        validate_pre_simulator_gate(protocol, tmp_path, protocol_path=protocol_path)
    assert not (tmp_path / "outputs").exists()


def test_source_hash_drift_is_rejected_before_r0_or_output(tmp_path: Path) -> None:
    for relative in TASK14R_R0S_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen:{relative}\n")
    hashes = source_files_sha256(tmp_path)
    protocol = build_task14r_r0s_protocol(
        implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=hashes,
    )
    (tmp_path / TASK14R_R0S_SOURCE_FILES[0]).write_text("drift\n")
    with pytest.raises(ValueError, match="source_files_sha256"):
        validate_pre_simulator_gate(protocol, tmp_path)
    assert not (tmp_path / "outputs").exists()


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
            "camera_calibration_matrix": eef_matrix,
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
    assert np.array_equal(invariant["obs_cache"]["camera_calibration_matrix"], eef_matrix)
    assert np.array_equal(invariant["obs_cache"]["robot0_eef_mat"], eef_matrix)
    assert "_current_observed_value" not in invariant["observables"]["agentview_image"]
    assert np.array_equal(invariant["observables"]["robot0_eef_mat"]["_current_observed_value"], eef_matrix)
    assert "_current_observed_value" in snapshot["observables"]["agentview_image"]
    rendered = rendered_python_state_manifest(snapshot)
    assert rendered["entry_count"] == 2
    assert set(rendered["entries"]) == {
        "obs_cache.agentview_image",
        "observables.agentview_image._current_observed_value",
    }
    assert all(len(row["sha256"]) == 64 for row in rendered["entries"].values())


def test_model_whitelist_audit_restores_only_offsamples(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = SimpleNamespace(model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4))))

    def fingerprint(value: object) -> dict[str, object]:
        offsamples = int(value.model.vis.quality.offsamples)
        arrays = {"body_pos": {"dtype": "float64", "shape": [1, 3], "sha256": "a" * 64}}
        return {
            "mjb_sha256": f"mjb-{offsamples}",
            "arrays": arrays,
            "complete_sha256": f"complete-{offsamples}",
        }

    monkeypatch.setattr(renderer_shift, "mujoco_model_fingerprint", fingerprint)
    canonical = fingerprint(sim)
    sim.model.vis.quality.offsamples = 0
    audit = renderer_model_whitelist_audit(
        sim,
        canonical_model_fingerprint=canonical,
        expected_offsamples=0,
    )
    assert audit["only_whitelisted_model_difference"]
    assert audit["full_mjb_identity_after_whitelist_restore"]
    assert audit["numeric_model_arrays_exact"]
    assert audit["physics_model_fingerprint_sha256"] == canonical["complete_sha256"]
    assert sim.model.vis.quality.offsamples == 0

    def drifted_fingerprint(value: object) -> dict[str, object]:
        result = fingerprint(value)
        result["arrays"] = {"body_pos": {"sha256": "b" * 64}}
        result["complete_sha256"] = "drifted"
        return result

    monkeypatch.setattr(renderer_shift, "mujoco_model_fingerprint", drifted_fingerprint)
    assert not renderer_model_whitelist_audit(
        sim,
        canonical_model_fingerprint=canonical,
        expected_offsamples=0,
    )["only_whitelisted_model_difference"]


def test_pixel_difference_is_exact_or_descriptive_without_a_tolerance() -> None:
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    exact = pixel_difference(left, left.copy())
    assert exact["exact"]
    assert exact["changed_channel_value_count"] == 0
    assert exact["per_channel_changed_counts"] == [0, 0, 0]
    assert exact["difference_bounding_box"] is None
    assert exact["first_differing_pixel_coordinate"] is None
    right = left.copy()
    right[1, 0, 2] = 3
    right[1, 1, 1] = 2
    shifted = pixel_difference(left, right)
    assert not shifted["exact"]
    assert shifted["changed_channel_value_count"] == 2
    assert shifted["changed_pixel_count"] == 2
    assert shifted["maximum_absolute_difference"] == 3.0
    assert shifted["mean_absolute_difference"] == 5 / 12
    assert shifted["per_channel_changed_counts"] == [0, 1, 1]
    assert shifted["difference_bounding_box"] == {
        "minimum_row": 1,
        "maximum_row": 1,
        "minimum_column": 0,
        "maximum_column": 1,
    }
    assert shifted["first_differing_pixel_coordinate"] == {"row": 1, "column": 0}
    assert shifted["first_differing_channel"] == 2
    assert shifted["reference_pixel_values"] == [0, 0, 0]
    assert shifted["actual_pixel_values"] == [0, 0, 3]
    assert shifted["reference_channel_value"] == 0
    assert shifted["actual_channel_value"] == 3
    assert set(TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS).issubset(shifted)
    summary = aggregate_pixel_differences([exact, shifted])
    assert summary["comparison_count"] == 2
    assert summary["exact_comparison_count"] == 1
    assert summary["shifted_comparison_count"] == 1
    assert not summary["exact"]
    assert summary["characterization_complete"]
    with pytest.raises(ValueError, match="raw uint8"):
        pixel_difference(left.astype(np.float32), right.astype(np.float32))
    with pytest.raises(ValueError, match="raw HWC uint8 RGB"):
        pixel_difference(np.zeros((3, 2, 2), dtype=np.uint8), np.zeros((3, 2, 2), dtype=np.uint8))


def test_pixel_comparison_rows_are_scoped_by_task_repeat_frame_camera() -> None:
    case = {
        "case_id": "r0s-libero_spatial-task00-state00",
        "task_index": 0,
        "suite": "libero_spatial",
        "task_id": 0,
    }
    left_pixels = {camera: np.zeros((2, 2, 3), dtype=np.uint8) for camera in ("image1", "image2")}
    left = {
        "record": {"renderer": "standard_msaa", "repeat_index": 0, "step": 4},
        "pixels": left_pixels,
    }
    right = {
        "record": {"renderer": "no_msaa", "repeat_index": 0, "step": 4},
        "pixels": {camera: value.copy() for camera, value in left_pixels.items()},
    }
    rows = compare_pixel_frames(
        left,
        right,
        comparison_class="standard_msaa_vs_no_msaa",
        case=case,
        frame_index=5,
    )
    assert len(rows) == 2
    assert {row["camera"] for row in rows} == {"image1", "image2"}
    for row in rows:
        assert row["case_id"] == case["case_id"]
        assert row["task_index"] == 0
        assert row["frame_index"] == 5
        assert row["step"] == 4
        assert row["comparison_class"] == "standard_msaa_vs_no_msaa"


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


def test_task_artifact_finalizer_rehashes_all_three_manifest_classes(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    case = {
        "case_id": "r0s-libero_spatial-task00-state00",
        "task_index": 0,
        "suite": "libero_spatial",
        "task_id": 0,
    }
    trace_manifests = []
    frame_record_manifests = []
    payload_manifests = []
    frames_by_trajectory = {}
    trace_rows_by_trajectory = {}
    for renderer_name in ("standard_msaa", "no_msaa"):
        for repeat_index in range(3):
            frames = []
            for frame_index in range(16):
                pixels = {camera: np.zeros((2, 2, 3), dtype=np.uint8) for camera in ("image1", "image2")}
                record = {
                    "renderer": renderer_name,
                    "repeat_index": repeat_index,
                    "step": frame_index - 1,
                    "action": None
                    if frame_index == 0
                    else [float(value) for value in renderer_shift.TASK14R_R0_PROBE_ACTIONS[frame_index - 1]],
                    "integration_state_sha256": "1" * 64,
                    "controller_python_state_sha256": "2" * 64,
                    "renderer_invariant_python_state_sha256": "2" * 64,
                    "robot_state_sha256": "3" * 64,
                    "contact_state_sha256": "4" * 64,
                    "physics_model_fingerprint_sha256": "5" * 64,
                    "model_whitelist_audit": {
                        "only_whitelisted_model_difference": True,
                        "physics_model_fingerprint_sha256": "5" * 64,
                    },
                    "rendered_python_state_manifest": {
                        "allowed_rendered_keys": [
                            "agentview_image",
                            "robot0_eye_in_hand_image",
                        ],
                        "entry_count": 0,
                        "entries": {},
                    },
                    "pixel_sha256": {
                        camera: renderer_shift.array_sha256(value) for camera, value in pixels.items()
                    },
                    "terminated": False,
                    "truncated": False,
                    "info_success": None if frame_index == 0 else False,
                    "success_predicate": False,
                }
                frames.append({"record": record, "pixels": pixels})
            key = (renderer_name, repeat_index)
            frames_by_trajectory[key] = frames
            trace_rows_by_trajectory[key] = [frame["record"] for frame in frames[1:]]
            stem = f"{renderer_name}.repeat{repeat_index:02d}"
            trace_manifest = write_trace(
                task_root / "probe_traces" / f"{stem}.steps.jsonl.gz",
                trace_rows_by_trajectory[key],
            )
            trace_manifest.update({"renderer": renderer_name, "repeat_index": repeat_index})
            trace_manifests.append(trace_manifest)
            frame_record_manifest = write_trace(
                task_root / "frame_records" / f"{stem}.frames.jsonl.gz",
                [frame["record"] for frame in frames],
            )
            frame_record_manifest["frame_count"] = frame_record_manifest.pop("step_count")
            frame_record_manifest.update({"renderer": renderer_name, "repeat_index": repeat_index})
            frame_record_manifests.append(frame_record_manifest)
            payload_manifest = write_frame_payload(
                task_root / "frame_payloads" / f"{stem}.frames.npz", frames
            )
            payload_manifest.update({"renderer": renderer_name, "repeat_index": repeat_index})
            payload_manifests.append(payload_manifest)

    comparison_rows = {
        "standard_msaa_within_repeat": [],
        "no_msaa_within_repeat": [],
        "standard_msaa_vs_no_msaa": [],
    }
    for renderer_name in ("standard_msaa", "no_msaa"):
        comparison_class = f"{renderer_name}_within_repeat"
        for actual_repeat in (1, 2):
            for frame_index, (left, right) in enumerate(
                zip(
                    frames_by_trajectory[(renderer_name, 0)],
                    frames_by_trajectory[(renderer_name, actual_repeat)],
                    strict=True,
                )
            ):
                comparison_rows[comparison_class].extend(
                    compare_pixel_frames(
                        left,
                        right,
                        comparison_class=comparison_class,
                        case=case,
                        frame_index=frame_index,
                    )
                )
    for repeat_index in range(3):
        for frame_index, (left, right) in enumerate(
            zip(
                frames_by_trajectory[("standard_msaa", repeat_index)],
                frames_by_trajectory[("no_msaa", repeat_index)],
                strict=True,
            )
        ):
            comparison_rows["standard_msaa_vs_no_msaa"].extend(
                compare_pixel_frames(
                    left,
                    right,
                    comparison_class="standard_msaa_vs_no_msaa",
                    case=case,
                    frame_index=frame_index,
                )
            )
    comparison_manifests = []
    for comparison_class, rows in comparison_rows.items():
        manifest = write_trace(
            task_root / "pixel_comparisons" / f"{comparison_class}.jsonl.gz",
            rows,
        )
        manifest["comparison_count"] = manifest.pop("step_count")
        manifest["comparison_class"] = comparison_class
        comparison_manifests.append(manifest)
    result = verify_task_artifacts(
        task_root,
        case=case,
        trace_manifests=trace_manifests,
        frame_record_manifests=frame_record_manifests,
        frame_payload_manifests=payload_manifests,
        pixel_comparison_manifests=comparison_manifests,
    )
    assert result == {
        "passed": True,
        "trace_manifest_count": 6,
        "frame_record_manifest_count": 6,
        "frame_payload_manifest_count": 6,
        "pixel_comparison_manifest_count": 3,
    }
    Path(trace_manifests[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_task_artifacts(
            task_root,
            case=case,
            trace_manifests=trace_manifests,
            frame_record_manifests=frame_record_manifests,
            frame_payload_manifests=payload_manifests,
            pixel_comparison_manifests=comparison_manifests,
        )

    repaired = write_trace(
        Path(trace_manifests[0]["path"]),
        trace_rows_by_trajectory[("standard_msaa", 0)],
    )
    trace_manifests[0].update(repaired)
    fake_rows = copy.deepcopy(comparison_rows["standard_msaa_vs_no_msaa"])
    fake_rows[0]["reference_image_sha256"] = "f" * 64
    fake_rows[0]["actual_image_sha256"] = "f" * 64
    fake_manifest = write_trace(
        Path(comparison_manifests[2]["path"]),
        fake_rows,
    )
    comparison_manifests[2].update(
        {
            **fake_manifest,
            "comparison_count": fake_manifest["step_count"],
            "comparison_class": "standard_msaa_vs_no_msaa",
        }
    )
    comparison_manifests[2].pop("step_count", None)
    with pytest.raises(ValueError, match="comparison/frame characterization mismatch"):
        verify_task_artifacts(
            task_root,
            case=case,
            trace_manifests=trace_manifests,
            frame_record_manifests=frame_record_manifests,
            frame_payload_manifests=payload_manifests,
            pixel_comparison_manifests=comparison_manifests,
        )


def _passed_tasks() -> list[dict[str, object]]:
    return [
        {
            "case": {"task_index": task_index},
            "passed": True,
            "renderer_arm_count": 2,
            "repeats_per_renderer": 3,
            "restore_transaction_count": 6,
            "probe_trajectory_count": 6,
            "probe_step_count": 90,
            "renderer_pair_count": 3,
            "paired_frame_count": 48,
            "physics_identity_frame_count": 48,
            "non_image_identity_frame_count": 48,
            "camera_comparison_count": 96,
            "cross_renderer_camera_comparison_count": 96,
            "within_renderer_physics_frame_count": 64,
            "within_renderer_physics_identity_frame_count": 64,
            "standard_repeat_camera_comparison_count": 64,
            "standard_repeat_exact_camera_comparison_count": 64,
            "no_msaa_repeat_camera_comparison_count": 64,
            "no_msaa_repeat_exact_camera_comparison_count": 64,
            "cross_renderer_exact_camera_comparison_count": 96,
            "cross_renderer_pixels_exact": True,
            "standard_repeat_pixels_exact": True,
            "no_msaa_repeat_pixels_exact": True,
            "standard_first_pair_count": 2 if task_index % 2 == 0 else 1,
            "no_msaa_first_pair_count": 1 if task_index % 2 == 0 else 2,
            "trace_manifest_count": 6,
            "frame_record_manifest_count": 6,
            "frame_payload_manifest_count": 6,
            "pixel_comparison_manifest_count": 3,
            "task_audit_manifest_count": 1,
            "task_audit_integrity_passed": True,
            "artifact_integrity_passed": True,
            "model_whitelist_only": True,
            "non_image_exact": True,
            "pixel_characterization_complete": True,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
            "execution_order": [
                {
                    "task_index": task_index,
                    "repeat_index": repeat_index,
                    "renderer_order": [row["name"] for row in renderer_order_for(task_index, repeat_index)],
                }
                for repeat_index in range(3)
            ],
        }
        for task_index in range(40)
    ]


def test_terminal_status_separates_exact_shift_and_engineering_failure() -> None:
    exact = task14r_r0s_terminal_summary(_passed_tasks())
    assert exact["status"] == TASK14R_R0S_STATUS_EXACT
    assert exact["probe_step_count"] == TASK14R_R0S_EXPECTED_PROBE_STEP_COUNT
    assert exact["paired_frame_count"] == TASK14R_R0S_EXPECTED_PAIRED_FRAME_COUNT
    assert exact["camera_comparison_count"] == TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT
    assert exact["cross_renderer_camera_comparison_count"] == (TASK14R_R0S_EXPECTED_CAMERA_COMPARISON_COUNT)
    assert exact["renderer_arm_count"] == TASK14R_R0S_EXPECTED_RENDERER_ARM_COUNT
    assert exact["standard_first_pair_count"] == TASK14R_R0S_EXPECTED_STANDARD_FIRST_PAIR_COUNT
    assert exact["no_msaa_first_pair_count"] == TASK14R_R0S_EXPECTED_NO_MSAA_FIRST_PAIR_COUNT
    assert exact["frame_payload_manifest_count"] == TASK14R_R0S_EXPECTED_FRAME_PAYLOAD_MANIFEST_COUNT
    assert exact["frame_record_manifest_count"] == 240
    assert exact["terminal_gate_checks"]["execution_order_frozen"]
    assert not exact["r1_authorized"]
    assert all(exact["terminal_gate_checks"].values())

    shifted_tasks = _passed_tasks()
    shifted_tasks[0]["cross_renderer_pixels_exact"] = False
    shifted_tasks[0]["cross_renderer_exact_camera_comparison_count"] = 95
    shifted = task14r_r0s_terminal_summary(shifted_tasks)
    assert shifted["status"] == TASK14R_R0S_STATUS_SHIFT
    assert all(shifted["terminal_gate_checks"].values())
    assert shifted["pixel_shift_task_count"] == 1
    assert shifted["cross_renderer_variation_task_count"] == 1
    assert shifted["standard_msaa_within_variation_task_count"] == 0

    standard_shift_tasks = _passed_tasks()
    standard_shift_tasks[0]["standard_repeat_pixels_exact"] = False
    standard_shift_tasks[0]["standard_repeat_exact_camera_comparison_count"] = 63
    standard_shift = task14r_r0s_terminal_summary(standard_shift_tasks)
    assert standard_shift["status"] == TASK14R_R0S_STATUS_SHIFT
    assert standard_shift["standard_msaa_within_variation_task_count"] == 1
    assert standard_shift["cross_renderer_variation_task_count"] == 0

    failed_tasks = _passed_tasks()
    failed_tasks[0]["no_msaa_repeat_exact_camera_comparison_count"] = 63
    failed_tasks[0]["no_msaa_repeat_pixels_exact"] = False
    failed = task14r_r0s_terminal_summary(failed_tasks)
    assert failed["status"] == TASK14R_R0S_STATUS_FAILED
    assert not failed["terminal_gate_checks"]["no_msaa_repeat_exact_camera_comparison_count"]
    assert failed["no_msaa_repeat_camera_comparison_count"] == (
        TASK14R_R0S_EXPECTED_NO_MSAA_REPEAT_CAMERA_COMPARISON_COUNT
    )


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("restore_transaction_count", 5, "restore_transaction_count"),
        ("restore_transaction_count", 7, "restore_transaction_count"),
        ("probe_trajectory_count", 5, "probe_trajectory_count"),
        ("probe_trajectory_count", 7, "probe_trajectory_count"),
        ("probe_step_count", 89, "probe_step_count"),
        ("probe_step_count", 91, "probe_step_count"),
        ("paired_frame_count", 47, "paired_frame_count"),
        ("paired_frame_count", 49, "paired_frame_count"),
        ("camera_comparison_count", 95, "camera_comparison_count"),
        ("camera_comparison_count", 97, "camera_comparison_count"),
        (
            "cross_renderer_camera_comparison_count",
            95,
            "cross_renderer_camera_comparison_count",
        ),
        ("renderer_arm_count", 1, "renderer_arm_count"),
        ("repeats_per_renderer", 2, "repeats_per_renderer"),
        ("standard_first_pair_count", 1, "standard_first_pair_count"),
        ("artifact_integrity_passed", False, "artifact_integrity"),
        ("model_whitelist_only", False, "model_renderer_whitelist"),
        ("non_image_exact", False, "non_image_exact"),
        ("pixel_characterization_complete", False, "pixel_characterization_complete"),
        ("frame_record_manifest_count", 5, "frame_record_manifest_count"),
        ("task_audit_manifest_count", 0, "task_audit_manifest_count"),
        ("task_audit_integrity_passed", False, "task_audit_integrity"),
    ],
)
def test_terminal_status_fails_each_engineering_gate(field: str, value: object, gate: str) -> None:
    tasks = _passed_tasks()
    tasks[0][field] = value
    summary = task14r_r0s_terminal_summary(tasks)
    assert summary["status"] == TASK14R_R0S_STATUS_FAILED
    assert not summary["terminal_gate_checks"][gate]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_query_count", 1),
        ("formal_case_count", 1),
        ("formal_outcome_rollout_count", 1),
        ("training_or_parameter_updates", True),
        ("automatic_next_phase", True),
    ],
)
def test_terminal_status_fails_activity_or_next_phase(field: str, value: object) -> None:
    tasks = _passed_tasks()
    tasks[0][field] = value
    summary = task14r_r0s_terminal_summary(tasks)
    assert summary["status"] == TASK14R_R0S_STATUS_FAILED
    assert not summary["terminal_gate_checks"][field]
    assert not summary["r1_authorized"]


def test_terminal_status_rejects_39_tasks_even_when_each_task_is_individually_valid() -> None:
    summary = task14r_r0s_terminal_summary(_passed_tasks()[:-1])
    assert summary["status"] == TASK14R_R0S_STATUS_FAILED
    assert not summary["terminal_gate_checks"]["task_count"]


def test_terminal_status_rejects_balanced_but_nonfrozen_execution_order() -> None:
    tasks = _passed_tasks()
    tasks[0]["execution_order"][0]["renderer_order"] = ["no_msaa", "standard_msaa"]
    summary = task14r_r0s_terminal_summary(tasks)
    assert summary["status"] == TASK14R_R0S_STATUS_FAILED
    assert not summary["terminal_gate_checks"]["execution_order_frozen"]


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
    assert audit["failure_stage"] == "task_runner"
    assert audit["policy_query_count"] == 0
    assert audit["formal_outcome_rollout_count"] == 0
    assert not audit["training_or_parameter_updates"]
    assert audit["renderer_arm"] is None
    assert audit["repeat_index"] is None
    assert audit["step_or_frame_index"] is None
    assert audit["camera"] is None
    assert audit["first_mismatched_field"] is None
    assert audit["partial_trace_manifests"] == []
    assert audit["partial_frame_payload_manifests"] == []
    assert audit["partial_pixel_comparison_manifests"] == []
    assert (tmp_path / "r0s-first" / "TASK_AUDIT_FAILURE.json").is_file()


def test_environment_creation_failure_is_not_mislabeled_as_task_runner(tmp_path: Path) -> None:
    case = {"case_id": "r0s-env-failure", "suite": "libero_spatial", "task_id": 0}

    def fail_environment() -> object:
        raise RuntimeError("injected environment failure")

    audit = execute_task_with_audit(
        case=case,
        task_root=tmp_path / "r0s-env-failure",
        environment_factory=fail_environment,
    )
    assert audit["failure_stage"] == "environment_creation"


def test_first_failure_context_is_structured_in_task_failure_artifact(tmp_path: Path) -> None:
    case = {"case_id": "r0s-first-context", "suite": "libero_spatial", "task_id": 0}
    env = SimpleNamespace(close=lambda: None)

    def fail(_env: object, _case: dict[str, object], _root: Path) -> dict[str, object]:
        raise Task14RRendererShiftError(
            "cross_renderer_physics_identity",
            "injected mismatch",
            renderer_arm="standard_msaa_vs_no_msaa",
            repeat_index=2,
            frame_index=7,
            camera="image2",
            first_mismatched_field="contact_state_sha256",
        )

    audit = execute_task_with_audit(
        case=case,
        task_root=tmp_path / "r0s-first-context",
        environment_factory=lambda: env,
        task_runner=fail,
    )
    assert audit["failure_stage"] == "cross_renderer_physics_identity"
    assert audit["renderer_arm"] == "standard_msaa_vs_no_msaa"
    assert audit["repeat_index"] == 2
    assert audit["step_or_frame_index"] == 7
    assert audit["camera"] == "image2"
    assert audit["first_mismatched_field"] == "contact_state_sha256"


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
        "run_libero_paired_four_arm_pilot",
    ):
        assert forbidden not in source
    helper_source = (SCRIPT_DIRECTORY / "task14r_renderer_shift.py").read_text()
    assert "np.allclose" not in source
    assert "np.allclose" not in helper_source
    assert "load_checkpoint" not in helper_source
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.ImportFrom,)) and node.module is not None
    }
    assert not any(
        "policy" in module.lower() or "checkpoint" in module.lower() for module in imported_modules
    )


def test_launcher_freezes_hardening_parent_and_requires_reviewed_head() -> None:
    source = (SCRIPT_DIRECTORY / "launch_task14r_r0s.sh").read_text()
    assert "TASK14R_R0S_EXPECTED_COMMIT" in source
    assert TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT in source
    assert "--force" not in source
    assert "r0s_attempt001" in source


def test_protocol_copy_does_not_mutate_when_validation_runs() -> None:
    protocol = _protocol()
    before = copy.deepcopy(protocol)
    _validate(protocol)
    assert protocol == before


def test_committed_protocol_is_canonical_and_matches_runtime_sources() -> None:
    path = (
        REPOSITORY_ROOT
        / "artifacts"
        / "crp_vla"
        / "task14r_reset_transaction_recovery"
        / "TASK14R_R0S_PROTOCOL.json"
    )
    protocol = json.loads(path.read_text())
    actual_hashes = source_files_sha256(REPOSITORY_ROOT)
    validate_task14r_r0s_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=actual_hashes,
    )
    canonical = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert path.read_text() == canonical
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(canonical.encode()).hexdigest()
    digest_path = REPOSITORY_ROOT / TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH
    assert digest_path.read_text() == f"{digest}  {path.name}\n"
