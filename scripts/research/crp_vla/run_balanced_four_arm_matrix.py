#!/usr/bin/env python
"""Run Base-10/Base-1/Snap-10/Snap-1 on a frozen balanced fixture manifest."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_crp.paired_inputs import evaluate_condition_pair, tensor_sha256
from lerobot.policies.smolvla_crp.response_metrics import compute_response_metrics
from lerobot.utils.constants import OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-revision", required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--teacher-response-threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def checkpoint_sha256(checkpoint: Path) -> str:
    path = checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return file_sha256(path)


def load_policy(checkpoint: Path, revision: str, device: str) -> SmolVLAPolicy:
    config = SmolVLAConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    return SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        revision=revision,
        strict=False,
    )


def load_batch(path: str, expected_sha256: str, device: str) -> dict:
    resolved = Path(path)
    if file_sha256(resolved) != expected_sha256:
        raise ValueError(f"Fixture hash mismatch: {resolved}")
    batch = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict) or OBS_STATE not in batch:
        raise ValueError(f"Invalid processed batch: {resolved}")
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def dimension_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, list[float]]:
    epsilon = 1e-8
    reference = reference.transpose(1, 2).reshape(reference.shape[0], reference.shape[2], -1)
    candidate = candidate.transpose(1, 2).reshape(candidate.shape[0], candidate.shape[2], -1)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
    error_norm = torch.linalg.vector_norm(candidate - reference, dim=-1)
    cosine = torch.sum(candidate * reference, dim=-1) / (
        candidate_norm * reference_norm + epsilon
    )
    return {
        "response_norm_ratio": (
            candidate_norm.mean(dim=0) / (reference_norm.mean(dim=0) + epsilon)
        ).detach().cpu().tolist(),
        "response_cosine": cosine.mean(dim=0).detach().cpu().tolist(),
        "normalized_response_error": (
            error_norm.mean(dim=0) / (reference_norm.mean(dim=0) + epsilon)
        ).detach().cpu().tolist(),
    }


def compare(reference: tuple[torch.Tensor, torch.Tensor], candidate: tuple[torch.Tensor, torch.Tensor]) -> dict:
    metrics = compute_response_metrics(*reference, *candidate)
    return {
        "global": {key: scalar(value) for key, value in metrics["global"].items()},
        "horizon": {
            key: value.detach().cpu().tolist() for key, value in metrics["horizon"].items()
        },
        "dimension": dimension_metrics(metrics["delta_teacher"], metrics["delta_student"]),
        "factual_action_mse": scalar(torch.mean((candidate[0] - reference[0]) ** 2)),
        "counterfactual_action_mse": scalar(torch.mean((candidate[1] - reference[1]) ** 2)),
    }


def metric_value(comparison: dict, metric: str) -> float:
    return comparison["global"].get(metric, comparison.get(metric))


def bootstrap_pair_mean(values: list[float], repeats: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_pairs": 0}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repeats, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_pairs": len(array),
    }


def aggregate(records: list[dict], repeats: int, seed: int) -> dict:
    comparisons = (
        "base_step_effect",
        "training_drift_at_10nfe",
        "snap_step_effect",
        "snap1_vs_base10",
    )
    metrics = (
        "factual_action_mse",
        "counterfactual_action_mse",
        "response_norm_ratio",
        "response_cosine",
        "normalized_response_error",
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["intervention_type"], record["pair_id"])].append(record)

    summary = {}
    for group in ("all", "object_target_swap", "relation_location_swap"):
        pair_groups = {
            pair_id: items
            for (intervention, pair_id), items in grouped.items()
            if group == "all" or intervention == group
        }
        summary[group] = {}
        for comparison in comparisons:
            summary[group][comparison] = {}
            for metric in metrics:
                pair_values = [
                    float(np.mean([metric_value(item["comparisons"][comparison], metric) for item in items]))
                    for items in pair_groups.values()
                ]
                summary[group][comparison][metric] = bootstrap_pair_mean(
                    pair_values,
                    repeats,
                    seed + len(summary[group][comparison]),
                )
    return summary


def main() -> None:
    args = parse_args()
    if len(args.noise_seeds) != len(set(args.noise_seeds)):
        raise ValueError("noise-seeds must be unique")
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    if git_value("status", "--porcelain"):
        raise ValueError("Repository must be clean before the admitted four-arm run")

    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    if fixture_manifest["status"] != "FIXTURES_MATERIALIZED":
        raise ValueError("Fixture manifest is not finalized")
    base = load_policy(args.base_checkpoint, args.base_revision, args.device)
    snap = load_policy(args.snap_checkpoint, args.snap_revision, args.device)
    if (base.config.chunk_size, base.config.max_action_dim) != (
        snap.config.chunk_size,
        snap.config.max_action_dim,
    ):
        raise ValueError("Base and SnapFlow action shapes differ")

    action_arrays = []
    results = []
    for pair_index, fixture in enumerate(fixture_manifest["records"]):
        factual = load_batch(
            fixture["factual_batch"], fixture["factual_batch_sha256"], args.device
        )
        counterfactual = load_batch(
            fixture["counterfactual_batch"],
            fixture["counterfactual_batch_sha256"],
            args.device,
        )
        for noise_seed in args.noise_seeds:
            generator = torch.Generator(device=args.device).manual_seed(noise_seed)
            noise = torch.randn(
                (factual[OBS_STATE].shape[0], base.config.chunk_size, base.config.max_action_dim),
                generator=generator,
                device=args.device,
                dtype=torch.float32,
            )
            arms = {
                "base10": evaluate_condition_pair(base, factual, counterfactual, noise, num_steps=10),
                "base1": evaluate_condition_pair(base, factual, counterfactual, noise, num_steps=1),
                "snap10": evaluate_condition_pair(snap, factual, counterfactual, noise, num_steps=10),
                "snap1": evaluate_condition_pair(
                    snap,
                    factual,
                    counterfactual,
                    noise,
                    num_steps=1,
                    target_time=0.0,
                ),
            }
            teacher_response_norm = scalar(torch.linalg.vector_norm(arms["base10"][0] - arms["base10"][1]))
            results.append(
                {
                    "pair_id": fixture["pair_id"],
                    "source_suite": fixture["source_suite"],
                    "source_task": fixture["source_task"],
                    "intervention_type": fixture["intervention_type"],
                    "noise_seed": noise_seed,
                    "noise_sha256": tensor_sha256(noise),
                    "non_condition_sha256": fixture["non_condition_sha256"],
                    "teacher_response_norm": teacher_response_norm,
                    "accepted": teacher_response_norm >= args.teacher_response_threshold,
                    "comparisons": {
                        "base_step_effect": compare(arms["base10"], arms["base1"]),
                        "training_drift_at_10nfe": compare(arms["base10"], arms["snap10"]),
                        "snap_step_effect": compare(arms["snap10"], arms["snap1"]),
                        "snap1_vs_base10": compare(arms["base10"], arms["snap1"]),
                    },
                }
            )
            action_arrays.append(
                np.stack(
                    [
                        np.stack([value.detach().cpu().numpy() for value in arms[arm]], axis=0)
                        for arm in ("base10", "base1", "snap10", "snap1")
                    ],
                    axis=0,
                )
            )
        print(f"completed_pair={pair_index + 1}/{fixture_manifest['pair_count']}", flush=True)

    admitted = [record for record in results if record["accepted"]]
    admitted_pairs = sorted({record["pair_id"] for record in admitted})
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.archive, actions=np.concatenate(action_arrays, axis=2))
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "ANALYZED",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "fixture_manifest": str(args.fixture_manifest.resolve()),
        "fixture_manifest_sha256": file_sha256(args.fixture_manifest),
        "base": {
            "checkpoint": str(args.base_checkpoint.resolve()),
            "revision": args.base_revision,
            "model_sha256": checkpoint_sha256(args.base_checkpoint),
        },
        "snap": {
            "checkpoint": str(args.snap_checkpoint.resolve()),
            "revision": args.snap_revision,
            "model_sha256": checkpoint_sha256(args.snap_checkpoint),
        },
        "noise_seeds": args.noise_seeds,
        "teacher_response_threshold": args.teacher_response_threshold,
        "record_count": len(results),
        "semantic_pair_count": fixture_manifest["pair_count"],
        "admitted_record_count": len(admitted),
        "admitted_semantic_pair_count": len(admitted_pairs),
        "bootstrap_unit": "semantic_pair_after_mean_over_noise_seeds",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "action_archive": str(args.archive.resolve()),
        "action_archive_sha256": file_sha256(args.archive),
        "summary": aggregate(admitted, args.bootstrap_repeats, args.bootstrap_seed),
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "record_count": output["record_count"],
                "admitted_record_count": output["admitted_record_count"],
                "admitted_semantic_pair_count": output["admitted_semantic_pair_count"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
