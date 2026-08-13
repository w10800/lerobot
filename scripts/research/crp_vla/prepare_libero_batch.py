#!/usr/bin/env python
"""Create a provenance-recorded, preprocessed LIBERO batch for CUDA gates."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--task-override", help="Condition text; all non-language inputs remain unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
        }
    if isinstance(value, dict):
        return {key: describe(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return {"kind": type(value).__name__, "repr": repr(value)}


def main() -> None:
    args = parse_args()
    if args.episode < 0 or args.sample_index < 0:
        raise ValueError("episode and sample-index must be non-negative")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    metadata = LeRobotDatasetMetadata(
        args.dataset,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        args.dataset,
        root=args.dataset_root,
        episodes=[args.episode],
        delta_timestamps=delta_timestamps,
        revision=args.dataset_revision,
        return_uint8=True,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} is outside dataset length {len(dataset)}")

    preprocessor, _ = make_pre_post_processors(
        config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    raw = dataset[args.sample_index]
    source_task = raw.get("task")
    if args.task_override is not None:
        raw["task"] = args.task_override
    # Match ``lerobot_train._preprocess_dataset_batch``. Dataset loading keeps
    # images as compact uint8 tensors, while SmolVLA expects float32 in [0, 1].
    for camera_key in metadata.camera_keys:
        value = raw.get(camera_key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            raw[camera_key] = value.to(dtype=torch.float32) / 255.0
    processed = preprocessor(raw)
    batch = {
        key: value.detach().to(device="cpu").contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }
    present_image_keys = [key for key in config.image_features if key in batch]
    if not present_image_keys:
        raise ValueError(f"No configured image feature survived preprocessing: {config.image_features}")
    for image_key in present_image_keys:
        image = batch[image_key]
        if not isinstance(image, torch.Tensor) or not image.is_floating_point():
            raise TypeError(f"Expected floating-point image tensor for {image_key}, got {type(image)}")
        if image.numel() and (image.min().item() < 0.0 or image.max().item() > 1.0):
            raise ValueError(f"Expected {image_key} in [0, 1] after preprocessing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "FIXTURE",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "episode": args.episode,
        "sample_index_within_selection": args.sample_index,
        "source_task": source_task,
        "effective_task": args.task_override if args.task_override is not None else source_task,
        "dataset_fps": metadata.fps,
        "delta_timestamps": delta_timestamps,
        "batch_file": str(args.output.resolve()),
        "batch_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "batch": describe(batch),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
