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
import yaml
from replay_v2_common import array_sha256, structured_hash
from task14r_renderer_shift import (
    TASK14R_R0S_CAMERA_KEYS,
    TASK14R_R0S_EVIDENCE_LABELS,
    TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0S_LIBERO_COMMIT,
    TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS,
    TASK14R_R0S_PROBE_STEPS,
    TASK14R_R0S_PROTOCOL_RELATIVE_PATH,
    TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH,
    TASK14R_R0S_QUALIFICATION_BOUNDARY,
    TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS,
    TASK14R_R0S_RENDERERS,
    TASK14R_R0S_REPEATS_PER_RENDERER,
    TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH,
    TASK14R_R0S_SOURCE_FILES,
    TASK14R_R0S_STATUS_FAILED,
    Task14RRendererShiftError,
    aggregate_pixel_differences,
    configure_renderer,
    pixel_difference,
    rendered_python_state_manifest,
    renderer_invariant_python_state,
    renderer_model_whitelist_audit,
    renderer_order_for,
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
    "controller_python_state_sha256",
    "robot_state_sha256",
    "contact_state_sha256",
    "physics_model_fingerprint_sha256",
    "terminated",
    "truncated",
    "info_success",
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


def git_repository_value(repository: Path, *args: str) -> str:
    resolved = repository.resolve()
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={resolved}", "-C", str(resolved), *args],
        text=True,
    ).strip()


def configure_standard_libero(root: Path, config_dir: Path) -> None:
    """Configure the pinned standard LIBERO checkout without importing policy code."""
    package_parent = root / "libero"
    benchmark_root = package_parent / "libero"
    required = (
        benchmark_root / "assets" / "scenes" / "libero_tabletop_base_style.xml",
        benchmark_root / "bddl_files" / "libero_spatial",
        benchmark_root / "init_files" / "libero_spatial",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Pinned LIBERO checkout is incomplete: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((package_parent / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())


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


def _manifest_path(manifest: Mapping[str, Any], task_root: Path) -> Path:
    path = Path(str(manifest.get("path", ""))).resolve()
    if not path.is_relative_to(task_root.resolve()):
        raise ValueError("Task14R R0S manifest path escaped the task root")
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != manifest.get("sha256"):
        raise ValueError(f"Task14R R0S manifest SHA-256 mismatch: {path}")
    return path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_frame_evidence(row: Mapping[str, Any], *, path: Path, index: int) -> None:
    model_audit = row.get("model_whitelist_audit")
    rendered_manifest = row.get("rendered_python_state_manifest")
    if not isinstance(model_audit, Mapping) or not bool(model_audit.get("only_whitelisted_model_difference")):
        raise ValueError(f"Task14R R0S frame model-whitelist evidence mismatch: {path}:{index}")
    hash_fields = (
        "integration_state_sha256",
        "controller_python_state_sha256",
        "robot_state_sha256",
        "contact_state_sha256",
        "physics_model_fingerprint_sha256",
    )
    if (
        not all(_is_sha256(row.get(field)) for field in hash_fields)
        or model_audit.get("physics_model_fingerprint_sha256") != row.get("physics_model_fingerprint_sha256")
        or not isinstance(row.get("terminated"), bool)
        or not isinstance(row.get("truncated"), bool)
        or not isinstance(row.get("success_predicate"), bool)
        or (index == 0 and row.get("info_success") is not None)
        or (index > 0 and not isinstance(row.get("info_success"), bool))
    ):
        raise ValueError(f"Task14R R0S frame non-image evidence mismatch: {path}:{index}")
    if (
        not isinstance(rendered_manifest, Mapping)
        or rendered_manifest.get("allowed_rendered_keys") != list(TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS)
        or not isinstance(rendered_manifest.get("entries"), Mapping)
        or int(rendered_manifest.get("entry_count", -1)) != len(rendered_manifest["entries"])
    ):
        raise ValueError(f"Task14R R0S rendered-state manifest mismatch: {path}:{index}")
    allowed_fragments = {f".{key}" for key in TASK14R_R0S_RENDERED_PYTHON_STATE_KEYS}
    for name, value in rendered_manifest["entries"].items():
        if (
            not any(fragment in str(name) for fragment in allowed_fragments)
            or not isinstance(value, Mapping)
            or not isinstance(value.get("shape"), list)
            or not isinstance(value.get("dtype"), str)
            or not _is_sha256(value.get("sha256"))
        ):
            raise ValueError(f"Task14R R0S rendered-state entry mismatch: {path}:{index}:{name}")


def verify_trace_manifest(
    manifest: Mapping[str, Any], task_root: Path, *, expected_step_count: int
) -> list[dict[str, Any]]:
    path = _manifest_path(manifest, task_root)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if int(manifest.get("step_count", -1)) != expected_step_count or len(rows) != expected_step_count:
        raise ValueError(f"Task14R R0S trace row-count mismatch: {path}")
    renderer = str(manifest.get("renderer", ""))
    repeat_index = int(manifest.get("repeat_index", -1))
    for step, (row, action) in enumerate(zip(rows, TASK14R_R0_PROBE_ACTIONS, strict=True)):
        if not set(_PHYSICS_FIELDS).issubset(row) or "pixel_sha256" not in row:
            raise ValueError(f"Task14R R0S trace field inventory mismatch: {path}:{step}")
        if (
            row.get("renderer") != renderer
            or int(row.get("repeat_index", -1)) != repeat_index
            or int(row.get("step", -1)) != step
            or row.get("action") != [float(value) for value in action]
        ):
            raise ValueError(f"Task14R R0S trace schedule mismatch: {path}:{step}")
        pixel_hashes = row.get("pixel_sha256")
        if (
            not isinstance(pixel_hashes, Mapping)
            or set(pixel_hashes) != set(TASK14R_R0S_CAMERA_KEYS)
            or not all(_is_sha256(value) for value in pixel_hashes.values())
        ):
            raise ValueError(f"Task14R R0S trace pixel-hash inventory mismatch: {path}:{step}")
        _validate_frame_evidence(row, path=path, index=step + 1)
    return rows


def verify_frame_record_manifest(
    manifest: Mapping[str, Any], task_root: Path, *, expected_frame_count: int
) -> list[dict[str, Any]]:
    path = _manifest_path(manifest, task_root)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if int(manifest.get("frame_count", -1)) != expected_frame_count or len(rows) != expected_frame_count:
        raise ValueError(f"Task14R R0S frame-record row-count mismatch: {path}")
    renderer = str(manifest.get("renderer", ""))
    repeat_index = int(manifest.get("repeat_index", -1))
    for frame_index, row in enumerate(rows):
        if not set(_PHYSICS_FIELDS).issubset(row) or "pixel_sha256" not in row:
            raise ValueError(f"Task14R R0S frame-record field inventory mismatch: {path}:{frame_index}")
        expected_step = frame_index - 1
        expected_action = (
            None
            if frame_index == 0
            else [float(value) for value in TASK14R_R0_PROBE_ACTIONS[frame_index - 1]]
        )
        if (
            row.get("renderer") != renderer
            or int(row.get("repeat_index", -1)) != repeat_index
            or int(row.get("step", -999)) != expected_step
            or row.get("action") != expected_action
        ):
            raise ValueError(f"Task14R R0S frame-record schedule mismatch: {path}:{frame_index}")
        pixel_hashes = row.get("pixel_sha256")
        if (
            not isinstance(pixel_hashes, Mapping)
            or set(pixel_hashes) != set(TASK14R_R0S_CAMERA_KEYS)
            or not all(_is_sha256(value) for value in pixel_hashes.values())
        ):
            raise ValueError(f"Task14R R0S frame-record pixel-hash mismatch: {path}:{frame_index}")
        _validate_frame_evidence(row, path=path, index=frame_index)
    return rows


def verify_frame_payload_manifest(
    manifest: Mapping[str, Any], task_root: Path, *, expected_frame_count: int
) -> dict[str, np.ndarray]:
    path = _manifest_path(manifest, task_root)
    if int(manifest.get("frame_count", -1)) != expected_frame_count:
        raise ValueError(f"Task14R R0S frame manifest count mismatch: {path}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(TASK14R_R0S_CAMERA_KEYS):
            raise ValueError(f"Task14R R0S frame camera inventory mismatch: {path}")
        for camera in TASK14R_R0S_CAMERA_KEYS:
            value = payload[camera]
            if (
                value.dtype != np.uint8
                or value.ndim != 4
                or value.shape[0] != expected_frame_count
                or value.shape[-1] != 3
                or min(value.shape[1:3]) <= 0
            ):
                raise ValueError(f"Task14R R0S frame payload mismatch: {path}:{camera}")
            expected = manifest.get("arrays", {}).get(camera)
            if expected != {"shape": list(value.shape), "dtype": str(value.dtype)}:
                raise ValueError(f"Task14R R0S frame manifest metadata mismatch: {path}:{camera}")
            arrays[camera] = np.ascontiguousarray(value).copy()
    return arrays


def verify_pixel_comparison_manifest(
    manifest: Mapping[str, Any],
    task_root: Path,
    *,
    expected_comparison_count: int,
    case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    path = _manifest_path(manifest, task_root)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if (
        int(manifest.get("comparison_count", -1)) != expected_comparison_count
        or len(rows) != expected_comparison_count
    ):
        raise ValueError(f"Task14R R0S pixel-comparison row-count mismatch: {path}")
    comparison_class = str(manifest.get("comparison_class", ""))
    if any(str(row.get("comparison_class", "")) != comparison_class for row in rows):
        raise ValueError(f"Task14R R0S pixel-comparison class mismatch: {path}")
    expected_case = {
        "case_id": str(case["case_id"]),
        "task_index": int(case["task_index"]),
        "suite": str(case["suite"]),
        "task_id": int(case["task_id"]),
    }
    for row in rows:
        if any(row.get(field) != expected for field, expected in expected_case.items()):
            raise ValueError(f"Task14R R0S pixel-comparison task scope mismatch: {path}")
        if not set(TASK14R_R0S_PIXEL_CHARACTERIZATION_FIELDS).issubset(row):
            raise ValueError(f"Task14R R0S pixel characterization field inventory mismatch: {path}")
        if (
            row.get("dtype") != "uint8"
            or row.get("channel_axis") != 2
            or not isinstance(row.get("shape"), list)
            or len(row["shape"]) != 3
            or row["shape"][-1] != 3
            or row.get("camera") not in TASK14R_R0S_CAMERA_KEYS
            or not _is_sha256(row.get("reference_image_sha256"))
            or not _is_sha256(row.get("actual_image_sha256"))
        ):
            raise ValueError(f"Task14R R0S pixel characterization schema mismatch: {path}")
        changed_channels = int(row.get("changed_channel_value_count", -1))
        changed_pixels = int(row.get("changed_pixel_count", -1))
        channel_counts = row.get("per_channel_changed_counts")
        exact = bool(row.get("exact"))
        if (
            not isinstance(channel_counts, list)
            or len(channel_counts) != 3
            or any(int(value) < 0 for value in channel_counts)
            or sum(int(value) for value in channel_counts) != changed_channels
            or changed_channels < 0
            or changed_pixels < 0
            or changed_pixels > int(row.get("pixel_count", -1))
        ):
            raise ValueError(f"Task14R R0S pixel characterization count mismatch: {path}")
        if exact:
            if (
                changed_channels != 0
                or changed_pixels != 0
                or float(row.get("maximum_absolute_difference", -1.0)) != 0.0
                or float(row.get("mean_absolute_difference", -1.0)) != 0.0
                or row.get("difference_bounding_box") is not None
                or row.get("first_differing_pixel_coordinate") is not None
                or row.get("first_differing_channel") is not None
                or row.get("reference_pixel_values") is not None
                or row.get("actual_pixel_values") is not None
                or row.get("reference_channel_value") is not None
                or row.get("actual_channel_value") is not None
                or row.get("reference_image_sha256") != row.get("actual_image_sha256")
            ):
                raise ValueError(f"Task14R R0S exact-pixel characterization mismatch: {path}")
        elif (
            changed_channels <= 0
            or changed_pixels <= 0
            or float(row.get("maximum_absolute_difference", 0.0)) <= 0.0
            or float(row.get("mean_absolute_difference", 0.0)) <= 0.0
            or row.get("difference_bounding_box") is None
            or row.get("first_differing_pixel_coordinate") is None
            or row.get("first_differing_channel") is None
            or row.get("reference_pixel_values") is None
            or row.get("actual_pixel_values") is None
            or row.get("reference_channel_value") is None
            or row.get("actual_channel_value") is None
            or row.get("reference_image_sha256") == row.get("actual_image_sha256")
        ):
            raise ValueError(f"Task14R R0S shifted-pixel characterization mismatch: {path}")

    expected_keys: set[tuple[int, int, int, str]]
    if comparison_class == "standard_msaa_within_repeat":
        expected_renderers = ("standard_msaa", "standard_msaa")
        expected_keys = {
            (0, actual_repeat, frame_index, camera)
            for actual_repeat in (1, 2)
            for frame_index in range(TASK14R_R0S_PROBE_STEPS + 1)
            for camera in TASK14R_R0S_CAMERA_KEYS
        }
    elif comparison_class == "no_msaa_within_repeat":
        expected_renderers = ("no_msaa", "no_msaa")
        expected_keys = {
            (0, actual_repeat, frame_index, camera)
            for actual_repeat in (1, 2)
            for frame_index in range(TASK14R_R0S_PROBE_STEPS + 1)
            for camera in TASK14R_R0S_CAMERA_KEYS
        }
    elif comparison_class == "standard_msaa_vs_no_msaa":
        expected_renderers = ("standard_msaa", "no_msaa")
        expected_keys = {
            (repeat_index, repeat_index, frame_index, camera)
            for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER)
            for frame_index in range(TASK14R_R0S_PROBE_STEPS + 1)
            for camera in TASK14R_R0S_CAMERA_KEYS
        }
    else:
        raise ValueError(f"Task14R R0S unknown pixel-comparison class: {comparison_class}")
    actual_keys = {
        (
            int(row.get("reference_repeat_index", -1)),
            int(row.get("actual_repeat_index", -1)),
            int(row.get("frame_index", -1)),
            str(row.get("camera", "")),
        )
        for row in rows
    }
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise ValueError(f"Task14R R0S pixel-comparison matrix mismatch: {path}")
    if any(
        (row.get("reference_renderer"), row.get("actual_renderer")) != expected_renderers
        or int(row.get("step", -999)) != int(row.get("frame_index", -1)) - 1
        for row in rows
    ):
        raise ValueError(f"Task14R R0S pixel-comparison renderer/frame schedule mismatch: {path}")
    aggregate_pixel_differences(rows)
    return rows


def verify_task_artifacts(
    task_root: Path,
    *,
    case: Mapping[str, Any],
    trace_manifests: Sequence[Mapping[str, Any]],
    frame_record_manifests: Sequence[Mapping[str, Any]],
    frame_payload_manifests: Sequence[Mapping[str, Any]],
    pixel_comparison_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(trace_manifests) != 6 or len(frame_record_manifests) != 6 or len(frame_payload_manifests) != 6:
        raise ValueError("Task14R R0S task trace/frame-record/frame-payload manifest count drift")
    if len(pixel_comparison_manifests) != 3:
        raise ValueError("Task14R R0S task comparison manifest count drift")
    expected_trajectory_keys = {
        (str(renderer["name"]), repeat_index)
        for renderer in TASK14R_R0S_RENDERERS
        for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER)
    }
    for name, manifests in (
        ("trace", trace_manifests),
        ("frame-record", frame_record_manifests),
        ("frame", frame_payload_manifests),
    ):
        keys = {(str(row.get("renderer", "")), int(row.get("repeat_index", -1))) for row in manifests}
        if keys != expected_trajectory_keys or len(keys) != len(manifests):
            raise ValueError(f"Task14R R0S {name} renderer/repeat manifest inventory drift")
        paths = {str(row.get("path", "")) for row in manifests}
        if len(paths) != len(manifests):
            raise ValueError(f"Task14R R0S duplicate {name} manifest path")
    trace_rows_by_trajectory: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for manifest in trace_manifests:
        key = (str(manifest["renderer"]), int(manifest["repeat_index"]))
        trace_rows_by_trajectory[key] = verify_trace_manifest(
            manifest, task_root, expected_step_count=TASK14R_R0S_PROBE_STEPS
        )
    frame_records_by_trajectory: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for manifest in frame_record_manifests:
        key = (str(manifest["renderer"]), int(manifest["repeat_index"]))
        frame_records_by_trajectory[key] = verify_frame_record_manifest(
            manifest,
            task_root,
            expected_frame_count=TASK14R_R0S_PROBE_STEPS + 1,
        )
    frames_by_trajectory: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for manifest in frame_payload_manifests:
        key = (str(manifest["renderer"]), int(manifest["repeat_index"]))
        frames_by_trajectory[key] = verify_frame_payload_manifest(
            manifest,
            task_root,
            expected_frame_count=TASK14R_R0S_PROBE_STEPS + 1,
        )
    expected_by_class = {
        "standard_msaa_within_repeat": 64,
        "no_msaa_within_repeat": 64,
        "standard_msaa_vs_no_msaa": 96,
    }
    observed_classes = {str(manifest.get("comparison_class")) for manifest in pixel_comparison_manifests}
    if observed_classes != set(expected_by_class):
        raise ValueError("Task14R R0S comparison-class manifest inventory drift")
    comparisons_by_class: dict[str, list[dict[str, Any]]] = {}
    for manifest in pixel_comparison_manifests:
        comparison_class = str(manifest["comparison_class"])
        comparisons_by_class[comparison_class] = verify_pixel_comparison_manifest(
            manifest,
            task_root,
            expected_comparison_count=expected_by_class[comparison_class],
            case=case,
        )
    for trajectory_key, rows in trace_rows_by_trajectory.items():
        payload = frames_by_trajectory[trajectory_key]
        frame_records = frame_records_by_trajectory[trajectory_key]
        if rows != frame_records[1:]:
            raise ValueError(f"Task14R R0S trace/frame-record mismatch: {trajectory_key}")
        for frame_index, row in enumerate(frame_records):
            for camera in TASK14R_R0S_CAMERA_KEYS:
                if row["pixel_sha256"][camera] != array_sha256(payload[camera][frame_index]):
                    raise ValueError(
                        "Task14R R0S frame-record/payload pixel hash mismatch: "
                        f"{trajectory_key}:{frame_index}:{camera}"
                    )
    for comparison_class, rows in comparisons_by_class.items():
        for row in rows:
            reference_key = (
                str(row["reference_renderer"]),
                int(row["reference_repeat_index"]),
            )
            actual_key = (str(row["actual_renderer"]), int(row["actual_repeat_index"]))
            camera = str(row["camera"])
            frame_index = int(row["frame_index"])
            recomputed = pixel_difference(
                frames_by_trajectory[reference_key][camera][frame_index],
                frames_by_trajectory[actual_key][camera][frame_index],
            )
            if any(row.get(field) != value for field, value in recomputed.items()):
                raise ValueError(
                    "Task14R R0S comparison/frame characterization mismatch: "
                    f"{comparison_class}:{reference_key}:{actual_key}:{frame_index}:{camera}"
                )
    return {
        "passed": True,
        "trace_manifest_count": len(trace_manifests),
        "frame_record_manifest_count": len(frame_record_manifests),
        "frame_payload_manifest_count": len(frame_payload_manifests),
        "pixel_comparison_manifest_count": len(pixel_comparison_manifests),
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
    protocol: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
    *,
    protocol_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any], str]:
    hashes = source_files_sha256(repository_root)
    validate_task14r_r0s_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=hashes,
    )
    if protocol_path is not None:
        expected_protocol_path = (repository_root / TASK14R_R0S_PROTOCOL_RELATIVE_PATH).resolve()
        if protocol_path.resolve() != expected_protocol_path:
            raise ValueError("Task14R R0S protocol path drift")
        digest_path = repository_root / TASK14R_R0S_PROTOCOL_SHA256_RELATIVE_PATH
        if not digest_path.is_file():
            raise FileNotFoundError(f"Frozen Task14R R0S protocol digest is missing: {digest_path}")
        fields = digest_path.read_text().strip().split()
        if fields != [file_sha256(protocol_path), protocol_path.name]:
            raise ValueError("Task14R R0S protocol SHA-256 drift")
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
    repeat_index: int,
    canonical_model_fingerprint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    core = env._env.env
    renderer_check = configure_renderer(env, offsamples=int(renderer["offsamples"]))
    if renderer_check["offsamples_after"] != int(renderer["offsamples"]):
        raise Task14RRendererShiftError(
            "renderer_configuration",
            "Renderer setting drift",
            renderer_arm=str(renderer["name"]),
            repeat_index=repeat_index,
            first_mismatched_field="sim.model.vis.quality.offsamples",
        )
    try:
        restore_mujoco_integration_state(core.sim, capsule["integration"])
        restore_python_transaction_state(env, capsule["entry_python"])
    except Exception as error:
        raise Task14RRendererShiftError(
            "complete_state_restore",
            str(error),
            renderer_arm=str(renderer["name"]),
            repeat_index=repeat_index,
        ) from error
    try:
        raw_observation = core._get_observations(force_update=True)
        observation = env._format_raw_obs(raw_observation)
    except Exception as error:
        raise Task14RRendererShiftError(
            "observation_regeneration",
            str(error),
            renderer_arm=str(renderer["name"]),
            repeat_index=repeat_index,
        ) from error
    ready_python = capture_python_transaction_state(env)
    ready_integration = capture_mujoco_integration_state(core.sim)
    _, robot_state = split_observation(observation)
    _, canonical_robot_state = split_observation(capsule["canonical_observation"])
    model_whitelist_audit = renderer_model_whitelist_audit(
        core.sim,
        canonical_model_fingerprint=canonical_model_fingerprint,
        expected_offsamples=int(renderer["offsamples"]),
    )
    checks = {
        "renderer": dict(renderer_check),
        "model_whitelist_audit": model_whitelist_audit,
        "model_whitelist_only": bool(model_whitelist_audit["only_whitelisted_model_difference"]),
        "integration_identity": np.array_equal(
            ready_integration["state"], np.asarray(capsule["integration"]["state"])
        ),
        "renderer_invariant_python_identity": structured_hash(renderer_invariant_python_state(ready_python))
        == structured_hash(renderer_invariant_python_state(capsule["ready_python"])),
        "robot_state_identity": structured_hash(robot_state) == structured_hash(canonical_robot_state),
        "success_predicate_identity": bool(env._env.check_success())
        == bool(capsule["initial_success_predicate"]),
    }
    checks["passed"] = all(
        bool(value) for key, value in checks.items() if key not in {"renderer", "model_whitelist_audit"}
    )
    if not checks["passed"]:
        failed = sorted(
            key
            for key, value in checks.items()
            if key not in {"renderer", "model_whitelist_audit", "passed"} and not value
        )
        raise Task14RRendererShiftError(
            "renderer_transaction_restore",
            f"Renderer transaction checks failed: {failed}",
            renderer_arm=str(renderer["name"]),
            repeat_index=repeat_index,
            first_mismatched_field=failed[0] if failed else None,
        )
    return observation, checks


def capture_frame(
    env: Any,
    observation: Mapping[str, Any],
    *,
    renderer_name: str,
    repeat_index: int,
    frame_index: int,
    step: int,
    action: Sequence[float] | None,
    terminated: bool,
    truncated: bool,
    success: bool,
    info_success: bool | None,
    canonical_model_fingerprint: Mapping[str, Any],
    expected_offsamples: int,
) -> dict[str, Any]:
    pixels, robot_state = split_observation(observation)
    core = env._env.env
    integration = capture_mujoco_integration_state(core.sim)
    model_whitelist_audit = renderer_model_whitelist_audit(
        core.sim,
        canonical_model_fingerprint=canonical_model_fingerprint,
        expected_offsamples=expected_offsamples,
    )
    if not bool(model_whitelist_audit["only_whitelisted_model_difference"]):
        raise Task14RRendererShiftError(
            "model_renderer_whitelist",
            "Full model differs outside the frozen renderer whitelist",
            renderer_arm=renderer_name,
            repeat_index=repeat_index,
            frame_index=frame_index,
            first_mismatched_field="model_physics_fingerprint",
            details={"model_whitelist_audit": model_whitelist_audit},
        )
    python_state = capture_python_transaction_state(env)
    controller_python_sha256 = structured_hash(renderer_invariant_python_state(python_state))
    record = {
        "renderer": str(renderer_name),
        "repeat_index": int(repeat_index),
        "step": int(step),
        "action": None if action is None else [float(value) for value in action],
        "integration_state_sha256": integration["state_sha256"],
        "controller_python_state_sha256": controller_python_sha256,
        "renderer_invariant_python_state_sha256": controller_python_sha256,
        "robot_state_sha256": structured_hash(robot_state),
        "contact_state_sha256": structured_hash(contact_state(core)),
        "physics_model_fingerprint_sha256": model_whitelist_audit["physics_model_fingerprint_sha256"],
        "model_whitelist_audit": model_whitelist_audit,
        "rendered_python_state_manifest": rendered_python_state_manifest(python_state),
        "pixel_sha256": {key: array_sha256(value) for key, value in pixels.items()},
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info_success": None if info_success is None else bool(info_success),
        "success_predicate": bool(success),
    }
    return {"record": record, "pixels": pixels, "model_whitelist_audit": model_whitelist_audit}


def compare_physics(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    left_record = left["record"]
    right_record = right["record"]
    return [field for field in _PHYSICS_FIELDS if left_record.get(field) != right_record.get(field)]


def compare_pixel_frames(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison_class: str,
    case: Mapping[str, Any],
    frame_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for camera in TASK14R_R0S_CAMERA_KEYS:
        try:
            difference = pixel_difference(left["pixels"][camera], right["pixels"][camera])
        except Exception as error:
            raise Task14RRendererShiftError(
                "pixel_characterization",
                str(error),
                renderer_arm=comparison_class,
                repeat_index=int(right["record"]["repeat_index"]),
                frame_index=int(frame_index),
                camera=camera,
                first_mismatched_field="pixels",
            ) from error
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "task_index": int(case["task_index"]),
                "suite": str(case["suite"]),
                "task_id": int(case["task_id"]),
                "comparison_class": comparison_class,
                "camera": camera,
                "reference_renderer": left["record"]["renderer"],
                "actual_renderer": right["record"]["renderer"],
                "reference_repeat_index": left["record"]["repeat_index"],
                "actual_repeat_index": right["record"]["repeat_index"],
                "frame_index": int(frame_index),
                "step": left["record"]["step"],
                **difference,
            }
        )
    return rows


def run_trajectory(
    env: Any,
    capsule: Mapping[str, Any],
    *,
    renderer: Mapping[str, Any],
    repeat_index: int,
    canonical_model_fingerprint: Mapping[str, Any],
    task_root: Path,
    progress: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    renderer_name = str(renderer["name"])
    stem = f"{renderer_name}.repeat{repeat_index:02d}"
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    progress.clear()
    progress.update(
        {
            "renderer": renderer_name,
            "repeat_index": repeat_index,
            "failure_stage": "renderer_transaction_restore",
            "current_frame_index": None,
            "restore_check": None,
            "completed_step_count": 0,
            "executed_step_count": 0,
            "trace_manifest": None,
            "frame_record_manifest": None,
            "frame_payload_manifest": None,
            "partial_trace_manifest": None,
            "partial_frame_record_manifest": None,
            "partial_frame_payload_manifest": None,
            "partial_artifact_persistence_error": None,
        }
    )
    try:
        observation, restore_check = restore_renderer_transaction(
            env,
            capsule,
            renderer=renderer,
            repeat_index=repeat_index,
            canonical_model_fingerprint=canonical_model_fingerprint,
        )
        progress["restore_check"] = restore_check
        progress["failure_stage"] = "initial_frame_capture"
        progress["current_frame_index"] = 0
        frames.append(
            capture_frame(
                env,
                observation,
                renderer_name=renderer_name,
                repeat_index=repeat_index,
                frame_index=0,
                step=-1,
                action=None,
                terminated=False,
                truncated=False,
                success=bool(env._env.check_success()),
                info_success=None,
                canonical_model_fingerprint=canonical_model_fingerprint,
                expected_offsamples=int(renderer["offsamples"]),
            )
        )
        for step, action in enumerate(TASK14R_R0_PROBE_ACTIONS):
            progress["failure_stage"] = "probe_step"
            progress["current_frame_index"] = step + 1
            observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            progress["executed_step_count"] = step + 1
            success_predicate = bool(env._env.check_success())
            info_success = bool(info.get("is_success", success_predicate))
            if info_success != success_predicate:
                raise Task14RRendererShiftError(
                    "success_predicate",
                    "Environment info success differs from the task success predicate",
                    renderer_arm=renderer_name,
                    repeat_index=repeat_index,
                    frame_index=step + 1,
                    first_mismatched_field="success_predicate",
                )
            progress["failure_stage"] = "probe_frame_capture"
            frame = capture_frame(
                env,
                observation,
                renderer_name=renderer_name,
                repeat_index=repeat_index,
                frame_index=step + 1,
                step=step,
                action=action,
                terminated=terminated,
                truncated=truncated,
                success=success_predicate,
                info_success=info_success,
                canonical_model_fingerprint=canonical_model_fingerprint,
                expected_offsamples=int(renderer["offsamples"]),
            )
            frames.append(frame)
            rows.append(frame["record"])
            progress["completed_step_count"] = len(rows)
            if terminated or truncated:
                break
        if len(rows) != TASK14R_R0S_PROBE_STEPS:
            raise Task14RRendererShiftError(
                "probe_trajectory",
                f"Probe terminated early at {len(rows)}/{TASK14R_R0S_PROBE_STEPS}",
                renderer_arm=renderer_name,
                repeat_index=repeat_index,
                frame_index=len(rows),
                first_mismatched_field="terminated_or_truncated",
            )
        if any(bool(row["success_predicate"]) for row in rows):
            first_success_step = next(int(row["step"]) for row in rows if bool(row["success_predicate"]))
            raise Task14RRendererShiftError(
                "probe_trajectory",
                "Probe reached task success",
                renderer_arm=renderer_name,
                repeat_index=repeat_index,
                frame_index=first_success_step + 1,
                first_mismatched_field="success_predicate",
            )
        progress["failure_stage"] = "final_model_whitelist"
        final_model_audit = renderer_model_whitelist_audit(
            env._env.env.sim,
            canonical_model_fingerprint=canonical_model_fingerprint,
            expected_offsamples=int(renderer["offsamples"]),
        )
        if not bool(final_model_audit["only_whitelisted_model_difference"]):
            raise Task14RRendererShiftError(
                "model_renderer_whitelist",
                "Full model changed outside the frozen renderer whitelist during probe",
                renderer_arm=renderer_name,
                repeat_index=repeat_index,
                first_mismatched_field="model_physics_fingerprint",
                details={"model_whitelist_audit": final_model_audit},
            )
        progress["failure_stage"] = "trace_write"
        trace_manifest = write_trace(task_root / "probe_traces" / f"{stem}.steps.jsonl.gz", rows)
        progress["trace_manifest"] = trace_manifest
        progress["failure_stage"] = "frame_record_write"
        frame_record_manifest = write_trace(
            task_root / "frame_records" / f"{stem}.frames.jsonl.gz",
            [frame["record"] for frame in frames],
        )
        frame_record_manifest["frame_count"] = frame_record_manifest.pop("step_count")
        progress["frame_record_manifest"] = frame_record_manifest
        progress["failure_stage"] = "frame_payload_write"
        payload_manifest = write_frame_payload(task_root / "frame_payloads" / f"{stem}.frames.npz", frames)
        progress["frame_payload_manifest"] = payload_manifest
        progress["failure_stage"] = "complete"
        return frames, restore_check, trace_manifest, frame_record_manifest, payload_manifest
    except Exception as error:
        persistence_errors = []
        if rows and progress.get("trace_manifest") is None:
            try:
                manifest = write_trace(task_root / "probe_traces" / f"{stem}.partial.steps.jsonl.gz", rows)
                manifest.update(
                    {
                        "renderer": renderer_name,
                        "repeat_index": repeat_index,
                        "complete": False,
                    }
                )
                progress["partial_trace_manifest"] = manifest
            except Exception as persistence_error:
                persistence_errors.append(f"trace:{type(persistence_error).__name__}:{persistence_error}")
        if frames and progress.get("frame_record_manifest") is None:
            try:
                manifest = write_trace(
                    task_root / "frame_records" / f"{stem}.partial.frames.jsonl.gz",
                    [frame["record"] for frame in frames],
                )
                manifest["frame_count"] = manifest.pop("step_count")
                manifest.update(
                    {
                        "renderer": renderer_name,
                        "repeat_index": repeat_index,
                        "complete": False,
                    }
                )
                progress["partial_frame_record_manifest"] = manifest
            except Exception as persistence_error:
                persistence_errors.append(
                    f"frame-records:{type(persistence_error).__name__}:{persistence_error}"
                )
        if frames and progress.get("frame_payload_manifest") is None:
            try:
                manifest = write_frame_payload(
                    task_root / "frame_payloads" / f"{stem}.partial.frames.npz", frames
                )
                manifest.update(
                    {
                        "renderer": renderer_name,
                        "repeat_index": repeat_index,
                        "complete": False,
                    }
                )
                progress["partial_frame_payload_manifest"] = manifest
            except Exception as persistence_error:
                persistence_errors.append(f"frames:{type(persistence_error).__name__}:{persistence_error}")
        progress["partial_artifact_persistence_error"] = persistence_errors or None
        if isinstance(error, Task14RRendererShiftError):
            progress["failure_details"] = error.details
            raise
        raise Task14RRendererShiftError(
            str(progress.get("failure_stage") or "probe_trajectory"),
            str(error),
            renderer_arm=renderer_name,
            repeat_index=repeat_index,
            frame_index=progress.get("current_frame_index"),
            first_mismatched_field=str(progress.get("failure_stage") or "probe_trajectory"),
        ) from error


def _partial_counts(
    trajectories: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
) -> tuple[int, int]:
    trajectory_count = sum(len(repeats) for repeats in trajectories.values())
    step_count = sum(max(0, len(frames) - 1) for repeats in trajectories.values() for frames in repeats)
    return trajectory_count, step_count


def run_task(env: Any, case: Mapping[str, Any], task_root: Path) -> dict[str, Any]:
    restore_checks: list[dict[str, Any]] = []
    trace_manifests: list[dict[str, Any]] = []
    frame_record_manifests: list[dict[str, Any]] = []
    payload_manifests: list[dict[str, Any]] = []
    pixel_comparison_manifests: list[dict[str, Any]] = []
    trajectories: dict[str, list[list[dict[str, Any]]]] = {
        str(renderer["name"]): [] for renderer in TASK14R_R0S_RENDERERS
    }
    execution_order: list[dict[str, Any]] = []
    standard_first_pair_count = 0
    no_msaa_first_pair_count = 0
    completed_renderer_pair_count = 0
    within_physics_count = 0
    within_physics_identity_count = 0
    paired_frame_count = 0
    physics_identity_count = 0
    non_image_identity_count = 0
    standard_repeat_pixels: list[dict[str, Any]] = []
    no_msaa_repeat_pixels: list[dict[str, Any]] = []
    cross_renderer_pixels: list[dict[str, Any]] = []
    active_renderer_arm: str | None = None
    active_repeat_index: int | None = None
    active_frame_index: int | None = None
    active_camera: str | None = None
    active_execution_position: int | None = None
    active_trajectory_progress: dict[str, Any] = {}
    first_mismatched_field: str | None = None
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
        canonical_model = capsule["model"]

        for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER):
            renderer_order = renderer_order_for(int(case["task_index"]), repeat_index)
            execution_order.append(
                {
                    "task_index": int(case["task_index"]),
                    "repeat_index": repeat_index,
                    "renderer_order": [str(renderer["name"]) for renderer in renderer_order],
                }
            )
            if renderer_order[0]["name"] == "standard_msaa":
                standard_first_pair_count += 1
            else:
                no_msaa_first_pair_count += 1
            for execution_position, renderer in enumerate(renderer_order):
                active_renderer_arm = str(renderer["name"])
                active_repeat_index = repeat_index
                active_execution_position = execution_position
                active_trajectory_progress = {}
                failure_stage = f"trajectory:{renderer['name']}:repeat{repeat_index}"
                (
                    frames,
                    restore_check,
                    trace_manifest,
                    frame_record_manifest,
                    payload_manifest,
                ) = run_trajectory(
                    env,
                    capsule,
                    renderer=renderer,
                    repeat_index=repeat_index,
                    canonical_model_fingerprint=canonical_model,
                    task_root=task_root,
                    progress=active_trajectory_progress,
                )
                trajectories[str(renderer["name"])].append(frames)
                restore_checks.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        "execution_position": execution_position,
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
                frame_record_manifests.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        **frame_record_manifest,
                    }
                )
                payload_manifests.append(
                    {
                        "renderer": renderer["name"],
                        "repeat_index": repeat_index,
                        **payload_manifest,
                    }
                )
                active_trajectory_progress = {}
            completed_renderer_pair_count += 1

        failure_stage = "within_renderer_physics_identity"
        for renderer_name, repeats in trajectories.items():
            reference = repeats[0]
            for repeat_index in range(1, len(repeats)):
                for frame_index, (left, right) in enumerate(
                    zip(reference, repeats[repeat_index], strict=True)
                ):
                    active_renderer_arm = renderer_name
                    active_repeat_index = repeat_index
                    active_frame_index = frame_index
                    within_physics_count += 1
                    mismatches = compare_physics(left, right)
                    if mismatches:
                        first_mismatched_field = mismatches[0]
                        raise Task14RRendererShiftError(
                            "within_renderer_physics_identity",
                            f"{renderer_name} repeat {repeat_index} frame {frame_index}: {mismatches}",
                            renderer_arm=renderer_name,
                            repeat_index=repeat_index,
                            frame_index=frame_index,
                            first_mismatched_field=mismatches[0],
                        )
                    within_physics_identity_count += 1
                    comparison_class = (
                        "no_msaa_within_repeat"
                        if renderer_name == "no_msaa"
                        else "standard_msaa_within_repeat"
                    )
                    rows = compare_pixel_frames(
                        left,
                        right,
                        comparison_class=comparison_class,
                        case=case,
                        frame_index=frame_index,
                    )
                    if renderer_name == "no_msaa":
                        no_msaa_repeat_pixels.extend(rows)
                    else:
                        standard_repeat_pixels.extend(rows)
        failure_stage = "within_comparison_write"
        active_repeat_index = None
        active_frame_index = None
        active_camera = None
        first_mismatched_field = None
        for comparison_class, rows in (
            ("standard_msaa_within_repeat", standard_repeat_pixels),
            ("no_msaa_within_repeat", no_msaa_repeat_pixels),
        ):
            active_renderer_arm = comparison_class
            manifest = write_trace(
                task_root / "pixel_comparisons" / f"{comparison_class}.jsonl.gz",
                rows,
            )
            manifest["comparison_count"] = manifest.pop("step_count")
            manifest["comparison_class"] = comparison_class
            pixel_comparison_manifests.append(manifest)
        if not all(bool(row["exact"]) for row in no_msaa_repeat_pixels):
            first = next(row for row in no_msaa_repeat_pixels if not bool(row["exact"]))
            active_renderer_arm = "no_msaa"
            active_repeat_index = int(first["actual_repeat_index"])
            active_frame_index = int(first["frame_index"])
            active_camera = str(first["camera"])
            first_mismatched_field = "pixels"
            raise Task14RRendererShiftError(
                "no_msaa_repeat_pixel_identity",
                f"No-MSAA repeat mismatch: {json.dumps(first)}",
                renderer_arm="no_msaa",
                repeat_index=int(first["actual_repeat_index"]),
                frame_index=int(first["frame_index"]),
                camera=str(first["camera"]),
                first_mismatched_field="pixels",
            )

        failure_stage = "cross_renderer_identity"
        for repeat_index in range(TASK14R_R0S_REPEATS_PER_RENDERER):
            standard = trajectories["standard_msaa"][repeat_index]
            no_msaa = trajectories["no_msaa"][repeat_index]
            for frame_index, (left, right) in enumerate(zip(standard, no_msaa, strict=True)):
                active_renderer_arm = "standard_msaa_vs_no_msaa"
                active_repeat_index = repeat_index
                active_frame_index = frame_index
                paired_frame_count += 1
                mismatches = compare_physics(left, right)
                if mismatches:
                    first_mismatched_field = mismatches[0]
                    raise Task14RRendererShiftError(
                        "cross_renderer_physics_identity",
                        f"repeat {repeat_index} frame {frame_index}: {mismatches}",
                        renderer_arm="standard_msaa_vs_no_msaa",
                        repeat_index=repeat_index,
                        frame_index=frame_index,
                        first_mismatched_field=mismatches[0],
                    )
                physics_identity_count += 1
                if left["record"]["robot_state_sha256"] != right["record"]["robot_state_sha256"]:
                    first_mismatched_field = "robot_state_sha256"
                    raise Task14RRendererShiftError(
                        "cross_renderer_non_image_identity",
                        f"repeat {repeat_index} frame {frame_index}: robot_state",
                        renderer_arm="standard_msaa_vs_no_msaa",
                        repeat_index=repeat_index,
                        frame_index=frame_index,
                        first_mismatched_field="robot_state_sha256",
                    )
                non_image_identity_count += 1
                cross_renderer_pixels.extend(
                    compare_pixel_frames(
                        left,
                        right,
                        comparison_class="standard_msaa_vs_no_msaa",
                        case=case,
                        frame_index=frame_index,
                    )
                )

        failure_stage = "cross_comparison_write"
        active_renderer_arm = "standard_msaa_vs_no_msaa"
        active_repeat_index = None
        active_frame_index = None
        active_camera = None
        first_mismatched_field = None
        cross_manifest = write_trace(
            task_root / "pixel_comparisons" / "standard_msaa_vs_no_msaa.jsonl.gz",
            cross_renderer_pixels,
        )
        cross_manifest["comparison_count"] = cross_manifest.pop("step_count")
        cross_manifest["comparison_class"] = "standard_msaa_vs_no_msaa"
        pixel_comparison_manifests.append(cross_manifest)

        failure_stage = "pixel_aggregation"
        cross_by_camera = {
            camera: aggregate_pixel_differences(
                [row for row in cross_renderer_pixels if row["camera"] == camera]
            )
            for camera in TASK14R_R0S_CAMERA_KEYS
        }
        standard_repeat_summary = aggregate_pixel_differences(standard_repeat_pixels)
        no_msaa_repeat_summary = aggregate_pixel_differences(no_msaa_repeat_pixels)
        cross_summary = aggregate_pixel_differences(cross_renderer_pixels)
        failure_stage = "artifact_integrity"
        active_renderer_arm = None
        artifact_integrity = verify_task_artifacts(
            task_root,
            case=case,
            trace_manifests=trace_manifests,
            frame_record_manifests=frame_record_manifests,
            frame_payload_manifests=payload_manifests,
            pixel_comparison_manifests=pixel_comparison_manifests,
        )
        return {
            "schema_version": "task14r.r0s.task_audit.v2",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": True,
            "renderer_arm_count": len(TASK14R_R0S_RENDERERS),
            "repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
            "standard_offsamples": standard_offsamples,
            "canonical_model_complete_sha256": canonical_model["complete_sha256"],
            "model_whitelist_only": all(bool(check["model_whitelist_only"]) for check in restore_checks),
            "restore_transaction_count": len(restore_checks),
            "restore_checks": restore_checks,
            "execution_order": execution_order,
            "standard_first_pair_count": standard_first_pair_count,
            "no_msaa_first_pair_count": no_msaa_first_pair_count,
            "probe_trajectory_count": sum(len(rows) for rows in trajectories.values()),
            "probe_step_count": sum(
                len(frames) - 1 for repeats in trajectories.values() for frames in repeats
            ),
            "trace_manifests": trace_manifests,
            "frame_record_manifests": frame_record_manifests,
            "frame_payload_manifests": payload_manifests,
            "pixel_comparison_manifests": pixel_comparison_manifests,
            "artifact_integrity": artifact_integrity,
            "artifact_integrity_passed": True,
            "trace_manifest_count": len(trace_manifests),
            "frame_record_manifest_count": len(frame_record_manifests),
            "frame_payload_manifest_count": len(payload_manifests),
            "pixel_comparison_manifest_count": len(pixel_comparison_manifests),
            "task_audit_manifest_count": 0,
            "task_audit_integrity_passed": False,
            "renderer_pair_count": TASK14R_R0S_REPEATS_PER_RENDERER,
            "paired_frame_count": paired_frame_count,
            "physics_identity_frame_count": physics_identity_count,
            "non_image_identity_frame_count": non_image_identity_count,
            "non_image_exact": non_image_identity_count == paired_frame_count,
            "camera_comparison_count": len(cross_renderer_pixels),
            "cross_renderer_camera_comparison_count": len(cross_renderer_pixels),
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
            "cross_renderer_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in cross_renderer_pixels
            ),
            "cross_renderer_pixel_summary": cross_summary,
            "cross_renderer_pixel_summary_by_camera": cross_by_camera,
            "pixel_characterization_complete": all(
                bool(summary["characterization_complete"])
                for summary in (standard_repeat_summary, no_msaa_repeat_summary, cross_summary)
            ),
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
    except Exception as error:
        trajectory_count, step_count = _partial_counts(trajectories)
        active_completed_steps = int(active_trajectory_progress.get("completed_step_count", 0))
        active_executed_steps = int(active_trajectory_progress.get("executed_step_count", 0))
        step_count += active_completed_steps
        executed_step_count = step_count - active_completed_steps + active_executed_steps
        if active_completed_steps == TASK14R_R0S_PROBE_STEPS:
            trajectory_count += 1
        active_restore_check = active_trajectory_progress.get("restore_check")
        if isinstance(active_restore_check, Mapping):
            restore_checks.append(
                {
                    "renderer": active_renderer_arm,
                    "repeat_index": active_repeat_index,
                    "execution_position": active_execution_position,
                    **active_restore_check,
                }
            )
        active_trace_manifest = active_trajectory_progress.get("trace_manifest") or (
            active_trajectory_progress.get("partial_trace_manifest")
        )
        if isinstance(active_trace_manifest, Mapping):
            enriched = {
                "renderer": active_renderer_arm,
                "repeat_index": active_repeat_index,
                **active_trace_manifest,
            }
            if str(enriched.get("path", "")) not in {
                str(manifest.get("path", "")) for manifest in trace_manifests
            }:
                trace_manifests.append(enriched)
        active_frame_record_manifest = active_trajectory_progress.get("frame_record_manifest") or (
            active_trajectory_progress.get("partial_frame_record_manifest")
        )
        if isinstance(active_frame_record_manifest, Mapping):
            enriched = {
                "renderer": active_renderer_arm,
                "repeat_index": active_repeat_index,
                **active_frame_record_manifest,
            }
            if str(enriched.get("path", "")) not in {
                str(manifest.get("path", "")) for manifest in frame_record_manifests
            }:
                frame_record_manifests.append(enriched)
        active_payload_manifest = active_trajectory_progress.get("frame_payload_manifest") or (
            active_trajectory_progress.get("partial_frame_payload_manifest")
        )
        if isinstance(active_payload_manifest, Mapping):
            enriched = {
                "renderer": active_renderer_arm,
                "repeat_index": active_repeat_index,
                **active_payload_manifest,
            }
            if str(enriched.get("path", "")) not in {
                str(manifest.get("path", "")) for manifest in payload_manifests
            }:
                payload_manifests.append(enriched)
        partial_comparison_persistence_error: Any = None
        persisted_classes = {str(manifest.get("comparison_class")) for manifest in pixel_comparison_manifests}
        for comparison_class, rows in (
            ("standard_msaa_within_repeat", standard_repeat_pixels),
            ("no_msaa_within_repeat", no_msaa_repeat_pixels),
            ("standard_msaa_vs_no_msaa", cross_renderer_pixels),
        ):
            if not rows or comparison_class in persisted_classes:
                continue
            try:
                manifest = write_trace(
                    task_root / "pixel_comparisons" / f"{comparison_class}.partial.jsonl.gz",
                    rows,
                )
                manifest["comparison_count"] = manifest.pop("step_count")
                manifest["comparison_class"] = comparison_class
                manifest["complete"] = False
                pixel_comparison_manifests.append(manifest)
            except Exception as persistence_error:
                partial_comparison_persistence_error = (
                    f"{type(persistence_error).__name__}: {persistence_error}"
                )
                break
        partial_artifact_persistence_error = {
            "trajectory": active_trajectory_progress.get("partial_artifact_persistence_error"),
            "comparison": partial_comparison_persistence_error,
        }
        if not any(partial_artifact_persistence_error.values()):
            partial_artifact_persistence_error = None
        return {
            "schema_version": "task14r.r0s.task_failure_audit.v2",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": False,
            "failure_stage": str(getattr(error, "failure_stage", failure_stage)),
            "renderer_arm": getattr(error, "renderer_arm", None) or active_renderer_arm,
            "repeat_index": getattr(error, "repeat_index", None)
            if getattr(error, "repeat_index", None) is not None
            else active_repeat_index,
            "step_or_frame_index": getattr(error, "frame_index", None)
            if getattr(error, "frame_index", None) is not None
            else active_frame_index,
            "camera": getattr(error, "camera", None) or active_camera,
            "first_mismatched_field": getattr(error, "first_mismatched_field", None)
            or first_mismatched_field,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "restore_transaction_count": len(restore_checks),
            "partial_restore_checks": restore_checks,
            "probe_trajectory_count": trajectory_count,
            "probe_step_count": step_count,
            "recorded_probe_step_count": step_count,
            "executed_probe_step_count": executed_step_count,
            "partial_trace_manifests": trace_manifests,
            "partial_frame_record_manifests": frame_record_manifests,
            "partial_frame_payload_manifests": payload_manifests,
            "partial_pixel_comparison_manifests": pixel_comparison_manifests,
            "partial_artifact_persistence_error": partial_artifact_persistence_error,
            "trace_manifest_count": len(trace_manifests),
            "frame_record_manifest_count": len(frame_record_manifests),
            "frame_payload_manifest_count": len(payload_manifests),
            "pixel_comparison_manifest_count": len(pixel_comparison_manifests),
            "task_audit_manifest_count": 0,
            "task_audit_integrity_passed": False,
            "artifact_integrity_passed": False,
            "partial_model_whitelist_checks": [
                check.get("model_whitelist_audit") for check in restore_checks
            ],
            "model_whitelist_only": bool(restore_checks)
            and all(bool(check.get("model_whitelist_only")) for check in restore_checks)
            and str(getattr(error, "failure_stage", "")) != "model_renderer_whitelist",
            "failure_details": getattr(error, "details", None)
            or active_trajectory_progress.get("failure_details"),
            "non_image_exact": False,
            "pixel_characterization_complete": False,
            "renderer_arm_count": len(TASK14R_R0S_RENDERERS),
            "repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
            "execution_order": execution_order,
            "standard_first_pair_count": standard_first_pair_count,
            "no_msaa_first_pair_count": no_msaa_first_pair_count,
            "renderer_pair_count": completed_renderer_pair_count,
            "paired_frame_count": paired_frame_count,
            "physics_identity_frame_count": physics_identity_count,
            "non_image_identity_frame_count": non_image_identity_count,
            "camera_comparison_count": len(cross_renderer_pixels),
            "cross_renderer_camera_comparison_count": len(cross_renderer_pixels),
            "within_renderer_physics_frame_count": within_physics_count,
            "within_renderer_physics_identity_frame_count": within_physics_identity_count,
            "standard_repeat_camera_comparison_count": len(standard_repeat_pixels),
            "standard_repeat_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in standard_repeat_pixels
            ),
            "no_msaa_repeat_camera_comparison_count": len(no_msaa_repeat_pixels),
            "no_msaa_repeat_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in no_msaa_repeat_pixels
            ),
            "cross_renderer_exact_camera_comparison_count": sum(
                bool(row["exact"]) for row in cross_renderer_pixels
            ),
            "no_msaa_repeat_pixels_exact": len(no_msaa_repeat_pixels) == 64
            and all(bool(row["exact"]) for row in no_msaa_repeat_pixels),
            "cross_renderer_pixels_exact": len(cross_renderer_pixels) == 96
            and all(bool(row["exact"]) for row in cross_renderer_pixels),
            "standard_repeat_pixels_exact": len(standard_repeat_pixels) == 64
            and all(bool(row["exact"]) for row in standard_repeat_pixels),
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
    failure_stage = "environment_creation"
    try:
        env = environment_factory()
        failure_stage = "task_runner"
        audit = task_runner(env, case, task_root)
    except Exception as error:
        audit = {
            "schema_version": "task14r.r0s.task_failure_audit.v2",
            "case": dict(case),
            "evidence_labels": list(TASK14R_R0S_EVIDENCE_LABELS),
            "passed": False,
            "failure_stage": str(getattr(error, "failure_stage", failure_stage)),
            "renderer_arm": getattr(error, "renderer_arm", None),
            "repeat_index": getattr(error, "repeat_index", None),
            "step_or_frame_index": getattr(error, "frame_index", None),
            "camera": getattr(error, "camera", None),
            "first_mismatched_field": getattr(error, "first_mismatched_field", None),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "restore_transaction_count": 0,
            "partial_restore_checks": [],
            "probe_trajectory_count": 0,
            "probe_step_count": 0,
            "partial_trace_manifests": [],
            "partial_frame_record_manifests": [],
            "partial_frame_payload_manifests": [],
            "partial_pixel_comparison_manifests": [],
            "partial_artifact_persistence_error": None,
            "trace_manifest_count": 0,
            "frame_record_manifest_count": 0,
            "frame_payload_manifest_count": 0,
            "pixel_comparison_manifest_count": 0,
            "task_audit_manifest_count": 0,
            "task_audit_integrity_passed": False,
            "artifact_integrity_passed": False,
            "model_whitelist_only": False,
            "partial_model_whitelist_checks": [],
            "non_image_exact": False,
            "pixel_characterization_complete": False,
            "renderer_arm_count": len(TASK14R_R0S_RENDERERS),
            "repeats_per_renderer": TASK14R_R0S_REPEATS_PER_RENDERER,
            "execution_order": [],
            "standard_first_pair_count": 0,
            "no_msaa_first_pair_count": 0,
            "renderer_pair_count": 0,
            "paired_frame_count": 0,
            "physics_identity_frame_count": 0,
            "non_image_identity_frame_count": 0,
            "camera_comparison_count": 0,
            "cross_renderer_camera_comparison_count": 0,
            "within_renderer_physics_frame_count": 0,
            "within_renderer_physics_identity_frame_count": 0,
            "standard_repeat_camera_comparison_count": 0,
            "standard_repeat_exact_camera_comparison_count": 0,
            "no_msaa_repeat_camera_comparison_count": 0,
            "no_msaa_repeat_exact_camera_comparison_count": 0,
            "cross_renderer_exact_camera_comparison_count": 0,
            "no_msaa_repeat_pixels_exact": False,
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


def _collect_manifests(
    task_audits: Sequence[Mapping[str, Any]], *, complete_field: str, partial_field: str
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for audit in task_audits:
        manifests = audit.get(complete_field)
        if not isinstance(manifests, list):
            manifests = audit.get(partial_field, [])
        if not isinstance(manifests, list):
            continue
        for manifest in manifests:
            if not isinstance(manifest, Mapping):
                continue
            path = str(manifest.get("path", ""))
            if path in observed_paths:
                continue
            observed_paths.add(path)
            collected.append(dict(manifest))
    return collected


def _failure_provenance(args: argparse.Namespace) -> dict[str, Any]:
    value: dict[str, Any] = {}
    checks: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("actual_repository_commit", lambda: git_value("rev-parse", "HEAD")),
        ("actual_repository_dirty", lambda: bool(git_value("status", "--porcelain"))),
        ("actual_protocol_sha256", lambda: file_sha256(args.protocol)),
        ("actual_source_files_sha256", lambda: source_files_sha256(REPOSITORY_ROOT)),
        (
            "actual_required_r0_final_sha256",
            lambda: file_sha256(REPOSITORY_ROOT / TASK14R_R0S_REQUIRED_R0_RELATIVE_PATH),
        ),
        ("actual_libero_commit", lambda: git_repository_value(args.libero_root, "rev-parse", "HEAD")),
    )
    for field, operation in checks:
        try:
            value[field] = operation()
        except Exception as error:
            value[field] = None
            value[f"{field}_query_error"] = f"{type(error).__name__}: {error}"
    return value


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task14R R0S")
    protocol = json.loads(args.protocol.read_text())
    source_hashes, r0_final, r0_final_sha256 = validate_pre_simulator_gate(
        protocol, protocol_path=args.protocol
    )
    if git_value("rev-parse", "HEAD^") != TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT:
        raise RuntimeError("Task14R R0S implementation parent commit drift")
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        raise RuntimeError("Task14R R0S requires MUJOCO_GL=egl")
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    if libero_commit != TASK14R_R0S_LIBERO_COMMIT:
        raise RuntimeError("Pinned LIBERO commit drift")

    protocol_sha256 = file_sha256(args.protocol)
    repository_commit = git_value("rev-parse", "HEAD")
    renderer_provenance = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **_nvidia_provenance(),
        "mujoco_version": None,
        "robosuite_version": None,
        "egl_vendor": None,
        "egl_version": None,
        "egl_query_status": "PENDING_CURRENT_CONTEXT",
    }
    task_audits: list[dict[str, Any]] = []
    current_case = None
    args.output_root.mkdir(parents=True, exist_ok=False)
    try:
        configure_standard_libero(args.libero_root, args.output_root / "libero_standard_config")

        import libero.libero as libero_module
        import mujoco
        import robosuite
        from libero.libero import benchmark

        assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
        libero_module._assets_path_cache = str(assets)
        renderer_provenance.update(
            {
                "mujoco_version": mujoco.__version__,
                "robosuite_version": robosuite.__version__,
            }
        )
        run_metadata = {
            "schema_version": "task14r.r0s.run_metadata.v2",
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
            "execution_order": protocol["execution_order"],
            "model_difference_contract": protocol["model_difference_contract"],
            "comparison_contract": protocol["comparison_contract"],
            "artifact_integrity_contract": protocol["artifact_integrity_contract"],
            "renderer_provenance": renderer_provenance,
            "qualification_boundary": list(TASK14R_R0S_QUALIFICATION_BOUNDARY),
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
            "r1_authorized": False,
            "policy_impact_bridge_required": True,
        }
        write_json(args.output_root / "RUN_METADATA.json", run_metadata)

        factories = benchmark.get_benchmark_dict()
        suites = {name: factories[name]() for name in sorted({row["suite"] for row in protocol["tasks"]})}
    except Exception as error:
        setup_failure_provenance = _failure_provenance(args)
        setup_failure = {
            **task14r_r0s_terminal_summary(task_audits),
            "status": TASK14R_R0S_STATUS_FAILED,
            "failure_stage": "runtime_setup",
            "failed_case": None,
            "renderer_arm": None,
            "repeat_index": None,
            "step_or_frame_index": None,
            "camera": None,
            "first_mismatched_field": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "repository_commit": repository_commit,
            "repository_dirty": setup_failure_provenance.get("actual_repository_dirty"),
            "protocol_sha256": protocol_sha256,
            "source_files_sha256": source_hashes,
            "libero_commit": libero_commit,
            "required_r0_final_sha256": r0_final_sha256,
            "renderer_provenance": renderer_provenance,
            "partial_trace_manifests": [],
            "partial_frame_record_manifests": [],
            "partial_frame_payload_manifests": [],
            "partial_pixel_comparison_manifests": [],
            "formal_outcomes_revealed": False,
            "r1_authorized": False,
            "policy_impact_bridge_required": True,
            **setup_failure_provenance,
        }
        write_json(args.output_root / "TASK14R_R0S_FAILURE.json", setup_failure)
        print(json.dumps(setup_failure, sort_keys=True))
        raise

    task_audit_manifests: list[dict[str, Any]] = []
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

        for audit in task_audits:
            task_root = args.output_root / "tasks" / str(audit["case"]["case_id"])
            try:
                task_audit_path = task_root / "TASK_AUDIT.json"
                if not task_audit_path.is_file():
                    raise FileNotFoundError(task_audit_path)
                persisted_audit = json.loads(task_audit_path.read_text())
                if persisted_audit != audit:
                    raise ValueError(f"Task14R R0S task audit content drift: {task_audit_path}")
                audit["artifact_integrity"] = verify_task_artifacts(
                    task_root,
                    case=audit["case"],
                    trace_manifests=audit["trace_manifests"],
                    frame_record_manifests=audit["frame_record_manifests"],
                    frame_payload_manifests=audit["frame_payload_manifests"],
                    pixel_comparison_manifests=audit["pixel_comparison_manifests"],
                )
                audit["artifact_integrity_passed"] = True
                task_audit_manifest = {
                    "case_id": str(audit["case"]["case_id"]),
                    "path": str(task_audit_path),
                    "sha256": file_sha256(task_audit_path),
                }
                task_audit_manifests.append(task_audit_manifest)
                audit["task_audit_manifest_count"] = 1
                audit["task_audit_integrity_passed"] = True
            except Exception as error:
                audit.update(
                    {
                        "passed": False,
                        "artifact_integrity_passed": False,
                        "failure_stage": "artifact_finalization",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "renderer_arm": None,
                        "repeat_index": None,
                        "step_or_frame_index": None,
                        "camera": None,
                        "first_mismatched_field": "artifact_manifest",
                    }
                )
                write_json(task_root / "TASK_AUDIT_FAILURE.json", audit)
                raise Task14RRendererShiftError(
                    "artifact_finalization",
                    str(error),
                    first_mismatched_field="artifact_manifest",
                ) from error
        final_source_hashes, _, final_r0_sha256 = validate_pre_simulator_gate(
            protocol, protocol_path=args.protocol
        )
        if final_source_hashes != source_hashes or final_r0_sha256 != r0_final_sha256:
            raise Task14RRendererShiftError("final_provenance", "Source or R0 provenance drift")
        if file_sha256(args.protocol) != protocol_sha256:
            raise Task14RRendererShiftError("final_provenance", "Protocol SHA-256 drift")
        if git_repository_value(args.libero_root, "rev-parse", "HEAD") != libero_commit:
            raise Task14RRendererShiftError("final_provenance", "LIBERO commit drift")
        if git_value("rev-parse", "HEAD") != repository_commit:
            raise Task14RRendererShiftError("final_provenance", "Repository HEAD changed during R0S")
        if git_value("status", "--porcelain"):
            raise Task14RRendererShiftError("final_provenance", "Repository became dirty")

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
            "task_audit_manifests": task_audit_manifests,
            "formal_outcomes_revealed": False,
            "r1_authorized": False,
            "policy_impact_bridge_required": True,
        }
        if summary["status"] == TASK14R_R0S_STATUS_FAILED:
            raise RuntimeError("Task14R R0S aggregate gate failed")
        write_json(args.output_root / "TASK14R_R0S_FINAL.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as error:
        failed_audit = next(
            (audit for audit in reversed(task_audits) if not bool(audit.get("passed"))),
            {},
        )
        failure_provenance = _failure_provenance(args)
        partial_trace_manifests = _collect_manifests(
            task_audits,
            complete_field="trace_manifests",
            partial_field="partial_trace_manifests",
        )
        partial_frame_record_manifests = _collect_manifests(
            task_audits,
            complete_field="frame_record_manifests",
            partial_field="partial_frame_record_manifests",
        )
        partial_frame_payload_manifests = _collect_manifests(
            task_audits,
            complete_field="frame_payload_manifests",
            partial_field="partial_frame_payload_manifests",
        )
        partial_pixel_comparison_manifests = _collect_manifests(
            task_audits,
            complete_field="pixel_comparison_manifests",
            partial_field="partial_pixel_comparison_manifests",
        )
        summary = {
            **task14r_r0s_terminal_summary(task_audits),
            "status": TASK14R_R0S_STATUS_FAILED,
            "repository_commit": repository_commit,
            "repository_dirty": failure_provenance.get("actual_repository_dirty"),
            "implementation_parent_commit": TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
            "protocol_sha256": protocol_sha256,
            "source_files_sha256": source_hashes,
            "libero_commit": libero_commit,
            "required_r0_final_sha256": r0_final_sha256,
            "renderer_provenance": renderer_provenance,
            "failed_case": failed_audit.get("case") if failed_audit else None,
            "failure_stage": str(
                getattr(error, "failure_stage", None)
                or failed_audit.get("failure_stage")
                or "aggregate_or_task_gate"
            ),
            "renderer_arm": getattr(error, "renderer_arm", None) or failed_audit.get("renderer_arm"),
            "repeat_index": getattr(error, "repeat_index", None)
            if getattr(error, "repeat_index", None) is not None
            else failed_audit.get("repeat_index"),
            "step_or_frame_index": getattr(error, "frame_index", None)
            if getattr(error, "frame_index", None) is not None
            else failed_audit.get("step_or_frame_index"),
            "camera": getattr(error, "camera", None) or failed_audit.get("camera"),
            "first_mismatched_field": getattr(error, "first_mismatched_field", None)
            or failed_audit.get("first_mismatched_field"),
            "partial_restore_checks": failed_audit.get("partial_restore_checks")
            or failed_audit.get("restore_checks", []),
            "partial_model_whitelist_checks": failed_audit.get("partial_model_whitelist_checks")
            or [
                check.get("model_whitelist_audit")
                for check in failed_audit.get("restore_checks", [])
                if isinstance(check, Mapping)
            ],
            "partial_trace_manifests": partial_trace_manifests,
            "partial_frame_record_manifests": partial_frame_record_manifests,
            "partial_frame_payload_manifests": partial_frame_payload_manifests,
            "partial_pixel_comparison_manifests": partial_pixel_comparison_manifests,
            "partial_task_audit_manifests": task_audit_manifests,
            "partial_artifact_persistence_error": failed_audit.get("partial_artifact_persistence_error"),
            "failure_details": getattr(error, "details", None) or failed_audit.get("failure_details"),
            "error_type": failed_audit.get("error_type", type(error).__name__),
            "error": failed_audit.get("error", str(error)),
            "traceback": failed_audit.get("traceback", traceback.format_exc()),
            "outer_error_type": type(error).__name__,
            "outer_error": str(error),
            "formal_outcomes_revealed": False,
            "r1_authorized": False,
            "policy_impact_bridge_required": True,
            **failure_provenance,
        }
        write_json(args.output_root / "TASK14R_R0S_FAILURE.json", summary)
        print(json.dumps(summary, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
