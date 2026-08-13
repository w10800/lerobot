#!/usr/bin/env python
"""Assemble Base-10/Base-1/Snap-10/Snap-1 matched-response attribution."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.smolvla_crp.response_metrics import compute_response_metrics

ARM_KEYS = ("base10", "base1", "snap10", "snap1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-records", nargs="+", type=Path, required=True)
    parser.add_argument("--snap10-records", nargs="+", type=Path, required=True)
    parser.add_argument("--snap1-records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def record_key(record: dict) -> tuple:
    return (
        record["task_group"],
        record["task_id"],
        record["intervention_type"],
        record["noise_seed"],
        record["factual_batch_sha256"],
        record["counterfactual_batch_sha256"],
    )


def load_records(paths: list[Path]) -> dict[tuple, tuple[dict, dict[str, np.ndarray]]]:
    records = {}
    for path in paths:
        record = json.loads(path.read_text())
        if record.get("repository_dirty"):
            raise ValueError(f"Dirty diagnostic record is inadmissible: {path}")
        archive_path = Path(record["action_archive"])
        if not archive_path.is_file():
            archive_path = path.with_suffix(".npz")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        key = record_key(record)
        if key in records:
            raise ValueError(f"Duplicate record key: {key}")
        records[key] = (record, arrays)
    return records


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def dimension_metrics(reference_delta: torch.Tensor, candidate_delta: torch.Tensor) -> dict[str, list[float]]:
    epsilon = 1e-8
    reference = reference_delta.transpose(1, 2).reshape(reference_delta.shape[0], reference_delta.shape[2], -1)
    candidate = candidate_delta.transpose(1, 2).reshape(candidate_delta.shape[0], candidate_delta.shape[2], -1)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
    error_norm = torch.linalg.vector_norm(candidate - reference, dim=-1)
    cosine = torch.sum(candidate * reference, dim=-1) / (
        candidate_norm * reference_norm + epsilon
    )
    return {
        "response_norm_ratio": (
            candidate_norm.mean(dim=0) / (reference_norm.mean(dim=0) + epsilon)
        ).tolist(),
        "response_cosine": cosine.mean(dim=0).tolist(),
        "normalized_response_error": (
            error_norm.mean(dim=0) / (reference_norm.mean(dim=0) + epsilon)
        ).tolist(),
    }


def compare(
    reference_factual: torch.Tensor,
    reference_counterfactual: torch.Tensor,
    candidate_factual: torch.Tensor,
    candidate_counterfactual: torch.Tensor,
) -> dict[str, Any]:
    metrics = compute_response_metrics(
        reference_factual,
        reference_counterfactual,
        candidate_factual,
        candidate_counterfactual,
    )
    return {
        "global": {
            key: scalar(value) for key, value in metrics["global"].items()
        },
        "horizon": {
            key: value.detach().cpu().tolist() for key, value in metrics["horizon"].items()
        },
        "dimension": dimension_metrics(metrics["delta_teacher"], metrics["delta_student"]),
        "factual_action_mse": scalar(
            torch.mean((candidate_factual - reference_factual) ** 2)
        ),
        "counterfactual_action_mse": scalar(
            torch.mean((candidate_counterfactual - reference_counterfactual) ** 2)
        ),
    }


def tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value)


def exact_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"Matched-arm invariant failed for {label}")


def main() -> None:
    args = parse_args()
    base = load_records(args.base_records)
    snap10 = load_records(args.snap10_records)
    snap1 = load_records(args.snap1_records)
    if set(base) != set(snap10) or set(base) != set(snap1):
        raise ValueError("Base, Snap-10, and Snap-1 record keys do not match")

    outputs = []
    for key in sorted(base):
        base_record, base_arrays = base[key]
        snap10_record, snap10_arrays = snap10[key]
        snap1_record, snap1_arrays = snap1[key]
        if not (
            base_record["noise_sha256"]
            == snap10_record["noise_sha256"]
            == snap1_record["noise_sha256"]
        ):
            raise ValueError(f"Noise hash mismatch for {key}")
        for branch in ("teacher_factual", "teacher_counterfactual"):
            exact_equal(base_arrays[branch], snap10_arrays[branch], f"{key}:{branch}:snap10")
            exact_equal(base_arrays[branch], snap1_arrays[branch], f"{key}:{branch}:snap1")

        arms = {
            "base10": (
                tensor(base_arrays["teacher_factual"]),
                tensor(base_arrays["teacher_counterfactual"]),
            ),
            "base1": (
                tensor(base_arrays["student_factual"]),
                tensor(base_arrays["student_counterfactual"]),
            ),
            "snap10": (
                tensor(snap10_arrays["student_factual"]),
                tensor(snap10_arrays["student_counterfactual"]),
            ),
            "snap1": (
                tensor(snap1_arrays["student_factual"]),
                tensor(snap1_arrays["student_counterfactual"]),
            ),
        }
        comparisons = {
            "base_step_effect": compare(*arms["base10"], *arms["base1"]),
            "training_drift_at_10nfe": compare(*arms["base10"], *arms["snap10"]),
            "snap_step_effect": compare(*arms["snap10"], *arms["snap1"]),
            "snap1_vs_base10": compare(*arms["base10"], *arms["snap1"]),
        }
        outputs.append(
            {
                "task_group": base_record["task_group"],
                "task_id": base_record["task_id"],
                "intervention_type": base_record["intervention_type"],
                "noise_seed": base_record["noise_seed"],
                "noise_sha256": base_record["noise_sha256"],
                "factual_batch_sha256": base_record["factual_batch_sha256"],
                "counterfactual_batch_sha256": base_record["counterfactual_batch_sha256"],
                "arms": {
                    "base10": base_record["teacher"],
                    "base1": base_record["student"],
                    "snap10": snap10_record["student"],
                    "snap1": snap1_record["student"],
                },
                "comparisons": comparisons,
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in outputs:
        grouped[f"{record['task_group']} / {record['intervention_type']}"].append(record)
    summary = {}
    for label, records in sorted(grouped.items()):
        summary[label] = {}
        for comparison in records[0]["comparisons"]:
            summary[label][comparison] = {}
            for metric in (
                "factual_action_mse",
                "counterfactual_action_mse",
                "response_norm_ratio",
                "response_cosine",
                "normalized_response_error",
            ):
                values = [
                    record["comparisons"][comparison]["global"].get(
                        metric,
                        record["comparisons"][comparison].get(metric),
                    )
                    for record in records
                ]
                summary[label][comparison][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n_noise_records": len(values),
                }

    output = {
        "schema_version": 1,
        "status": "ANALYZED",
        "design": {arm: arm for arm in ARM_KEYS},
        "record_count": len(outputs),
        "independent_semantic_pair_count": len(
            {(record["task_group"], record["task_id"], record["intervention_type"]) for record in outputs}
        ),
        "summary": summary,
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
