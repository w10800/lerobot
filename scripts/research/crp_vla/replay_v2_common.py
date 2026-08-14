#!/usr/bin/env python
"""Pure helpers for the replayable six-arm LIBERO protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ARMS = ("base10", "base2", "base1", "snap10", "snap2", "snap1")
ARM_SPECS = {
    "base10": ("base", 10, None),
    "base2": ("base", 2, None),
    "base1": ("base", 1, None),
    "snap10": ("snap", 10, None),
    "snap2": ("snap", 2, None),
    "snap1": ("snap", 1, 0.0),
}
PAIRWISE_COMPARISONS = (
    ("base2", "base1"),
    ("base10", "base2"),
    ("base10", "base1"),
    ("snap2", "snap1"),
    ("snap10", "snap2"),
    ("snap10", "snap1"),
    ("snap1", "base10"),
    ("snap10", "base10"),
)
FAILURE_CATEGORIES = (
    "WRONG_TARGET",
    "WRONG_RELATION",
    "GRASP_FAILURE",
    "GRIPPER_TIMING",
    "TRAJECTORY_COLLISION",
    "OVERSHOOT",
    "OBJECT_DROPPED",
    "RECEPTACLE_FAILURE",
    "UNKNOWN",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(json.dumps(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def flatten_arrays(value: Any, prefix: str = "") -> tuple[dict[str, np.ndarray], Any]:
    """Split nested data into NPZ-safe arrays and a JSON skeleton."""
    arrays: dict[str, np.ndarray] = {}

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, torch.Tensor):
            key = f"tensor__{len(arrays):04d}"
            tensor = item.detach().cpu().contiguous()
            torch_dtype = str(tensor.dtype)
            arrays[key] = (
                tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            )
            return {
                "__array__": key,
                "kind": "tensor",
                "logical_path": path,
                "torch_dtype": torch_dtype,
            }
        if isinstance(item, np.ndarray):
            key = f"array__{len(arrays):04d}"
            arrays[key] = np.ascontiguousarray(item)
            return {"__array__": key, "kind": "ndarray", "logical_path": path}
        if isinstance(item, dict):
            return {
                str(key): visit(item[key], f"{path}.{key}" if path else str(key))
                for key in sorted(item, key=str)
            }
        if isinstance(item, (list, tuple)):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, np.generic):
            return item.item()
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported capsule value at {path or prefix}: {type(item)!r}")

    return arrays, visit(value, prefix)


def restore_arrays(skeleton: Any, arrays: dict[str, np.ndarray], device: str | torch.device) -> Any:
    if isinstance(skeleton, dict) and "__array__" in skeleton:
        value = arrays[skeleton["__array__"]]
        if skeleton["kind"] == "tensor":
            tensor = torch.from_numpy(value.copy())
            if skeleton.get("torch_dtype") == "torch.bfloat16":
                tensor = tensor.view(torch.bfloat16)
            return tensor.to(device)
        return value.copy()
    if isinstance(skeleton, dict):
        return {key: restore_arrays(value, arrays, device) for key, value in skeleton.items()}
    if isinstance(skeleton, list):
        return [restore_arrays(value, arrays, device) for value in skeleton]
    return skeleton


def structured_hash(value: Any) -> str:
    arrays, skeleton = flatten_arrays(value)
    record = {
        "skeleton": skeleton,
        "arrays": {
            key: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": array_sha256(array),
            }
            for key, array in arrays.items()
        },
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_capsule(path: Path, metadata: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    arrays, skeleton = flatten_arrays(payload)
    np.savez_compressed(path / "arrays.npz", **arrays)
    array_manifest = {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }
        for key, value in arrays.items()
    }
    capsule = {
        **metadata,
        "payload_skeleton": skeleton,
        "array_manifest": array_manifest,
    }
    (path / "metadata.json").write_text(json.dumps(capsule, indent=2, sort_keys=True) + "\n")
    members = (path / "arrays.npz", path / "metadata.json")
    (path / "sha256_manifest.txt").write_text(
        "".join(f"{file_sha256(member)}  {member.name}\n" for member in members)
    )
    capsule_hash = file_sha256(path / "sha256_manifest.txt")
    return {"path": str(path.resolve()), "sha256": capsule_hash, "metadata": capsule}


def load_capsule(path: Path, device: str | torch.device = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = path / "sha256_manifest.txt"
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        expected[name] = digest
    for name, digest in expected.items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Capsule member hash mismatch: {name}")
    metadata = json.loads((path / "metadata.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, item in metadata["array_manifest"].items():
        value = arrays[key]
        if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
            raise ValueError(f"Capsule member schema mismatch: {key}")
        if array_sha256(value) != item["sha256"]:
            raise ValueError(f"Capsule member content mismatch: {key}")
    payload = restore_arrays(metadata["payload_skeleton"], arrays, device)
    return metadata, payload


def classify_failure(events: dict[str, Any], success: bool) -> dict[str, Any]:
    """Return only evidence-backed failure labels; ambiguous traces stay UNKNOWN."""
    if success:
        return {"category": None, "evidence": {"success": True}}
    if events.get("object_dropped", {}).get("observed"):
        return {"category": "OBJECT_DROPPED", "evidence": events["object_dropped"]}
    if events.get("collision", {}).get("observed"):
        return {"category": "TRAJECTORY_COLLISION", "evidence": events["collision"]}
    target_contact = events.get("first_target_contact", {})
    gripper_close = events.get("first_gripper_close", {})
    object_lift = events.get("first_object_lift", {})
    if target_contact.get("observed") and gripper_close.get("observed") and not object_lift.get("observed"):
        return {
            "category": "GRASP_FAILURE",
            "evidence": {
                "first_target_contact": target_contact,
                "first_gripper_close": gripper_close,
                "first_object_lift": object_lift,
            },
        }
    if (
        gripper_close.get("observed")
        and target_contact.get("available")
        and not target_contact.get("observed")
    ):
        return {
            "category": "GRIPPER_TIMING",
            "evidence": {
                "first_gripper_close": gripper_close,
                "first_target_contact": target_contact,
            },
        }
    return {
        "category": "UNKNOWN",
        "evidence": {
            "reason": "No registered category is directly established by the available event trace",
            "available_events": sorted(key for key, value in events.items() if value.get("available")),
        },
    }


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_win = int(np.sum((left == 1) & (right == 0)))
    right_win = int(np.sum((left == 0) & (right == 1)))
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_win, right_win) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_success_right_failure": left_win,
        "left_failure_right_success": right_win,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def bootstrap_interval(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_comparison(
    rows: list[dict[str, Any]], left: str, right: str, repeats: int, seed: int
) -> dict[str, Any]:
    left_values = np.asarray([row[f"{left}_success"] for row in rows], dtype=np.int8)
    right_values = np.asarray([row[f"{right}_success"] for row in rows], dtype=np.int8)
    differences = left_values.astype(np.float64) - right_values.astype(np.float64)
    task_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row, difference in zip(rows, differences, strict=True):
        task_groups[(row["suite"], int(row["task_id"]))].append(float(difference))
    task_values = np.asarray([np.mean(values) for values in task_groups.values()])
    return {
        "left": left,
        "right": right,
        "success_rate_difference": float(differences.mean()),
        "case_paired_bootstrap_ci95": bootstrap_interval(differences, repeats, seed),
        "task_cluster_bootstrap_ci95": bootstrap_interval(task_values, repeats, seed + 10_000),
        "mcnemar": exact_mcnemar(left_values, right_values),
    }


def validate_case_records(records: list[dict[str, Any]]) -> None:
    if {record["arm"] for record in records} != set(ARMS):
        raise ValueError("Replay-v2 case must contain exactly the six registered arms")
    reference = records[0]
    invariant_keys = (
        "capsule_sha256",
        "initial_sim_state_sha256",
        "canonical_input_sha256",
        "noise_schedule_sha256",
        "evaluator_sha256",
    )
    for record in records[1:]:
        mismatches = [key for key in invariant_keys if record.get(key) != reference.get(key)]
        if mismatches:
            raise ValueError(f"Replay-v2 invariant mismatch for {record['arm']}: {mismatches}")
    statuses = {record["status"] for record in records}
    if statuses not in ({"COMPLETED"}, {"ADMITTED_FAILURE"}):
        raise ValueError("All six arms must complete or all six must carry admitted failure records")
