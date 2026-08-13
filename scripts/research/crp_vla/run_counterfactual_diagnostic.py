#!/usr/bin/env python
"""Run a strict matched-noise teacher/student counterfactual diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_crp.paired_inputs import evaluate_condition_pair, tensor_sha256
from lerobot.policies.smolvla_crp.response_metrics import compute_response_metrics
from lerobot.utils.constants import OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--student-revision", required=True)
    parser.add_argument("--factual-batch", type=Path, required=True)
    parser.add_argument("--counterfactual-batch", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--teacher-steps", type=int, default=10)
    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--student-target-time", type=float)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_batch(path: Path, device: str) -> dict:
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict) or OBS_STATE not in batch:
        raise ValueError(f"{path} is not a processed SmolVLA tensor batch")
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }


def load_policy(checkpoint: str, revision: str, device: str) -> SmolVLAPolicy:
    config = SmolVLAConfig.from_pretrained(checkpoint, revision=revision)
    config.device = device
    if Path(checkpoint).is_dir():
        config.load_vlm_weights = False
    return SmolVLAPolicy.from_pretrained(checkpoint, config=config, revision=revision, strict=False)


def scalar_metrics(metrics: dict) -> dict[str, float | list[float]]:
    return {
        "global": {key: float(value.detach().cpu()) for key, value in metrics["global"].items()},
        "horizon": {key: value.detach().cpu().tolist() for key, value in metrics["horizon"].items()},
    }


def main() -> None:
    args = parse_args()
    factual = load_batch(args.factual_batch, args.device)
    counterfactual = load_batch(args.counterfactual_batch, args.device)
    teacher = load_policy(args.teacher_checkpoint, args.teacher_revision, args.device)
    student = load_policy(args.student_checkpoint, args.student_revision, args.device)
    if (
        teacher.config.chunk_size != student.config.chunk_size
        or teacher.config.max_action_dim != student.config.max_action_dim
    ):
        raise ValueError("Teacher/student action shapes are incompatible")

    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed)
    noise = torch.randn(
        (factual[OBS_STATE].shape[0], teacher.config.chunk_size, teacher.config.max_action_dim),
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    teacher_factual, teacher_counterfactual = evaluate_condition_pair(
        teacher, factual, counterfactual, noise, num_steps=args.teacher_steps
    )
    student_factual, student_counterfactual = evaluate_condition_pair(
        student,
        factual,
        counterfactual,
        noise,
        num_steps=args.student_steps,
        target_time=args.student_target_time,
    )
    metrics = compute_response_metrics(
        teacher_factual, teacher_counterfactual, student_factual, student_counterfactual
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_prefix.with_suffix(".npz")
    metadata_path = args.output_prefix.with_suffix(".json")
    np.savez_compressed(
        archive_path,
        teacher_factual=teacher_factual.detach().cpu().numpy(),
        teacher_counterfactual=teacher_counterfactual.detach().cpu().numpy(),
        student_factual=student_factual.detach().cpu().numpy(),
        student_counterfactual=student_counterfactual.detach().cpu().numpy(),
    )
    metadata = {
        "schema_version": 1,
        "status": "ANALYZED",
        "teacher": {
            "checkpoint": args.teacher_checkpoint,
            "revision": args.teacher_revision,
            "num_steps": args.teacher_steps,
        },
        "student": {
            "checkpoint": args.student_checkpoint,
            "revision": args.student_revision,
            "num_steps": args.student_steps,
            "target_time": args.student_target_time,
        },
        "noise_seed": args.noise_seed,
        "noise_sha256": tensor_sha256(noise),
        "factual_batch": str(args.factual_batch.resolve()),
        "counterfactual_batch": str(args.counterfactual_batch.resolve()),
        "action_archive": str(archive_path.resolve()),
        "metrics": scalar_metrics(metrics),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
