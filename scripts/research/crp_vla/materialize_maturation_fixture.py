#!/usr/bin/env python
"""Freeze the ordinary factual fixture and exact data-order hashes for D-013."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from materialize_balanced_pair_fixtures import (
    build_episode_task_indices,
    describe_batch,
    preprocess_raw,
)

REGISTERED_STEPS = (1_000, 3_000, 5_000, 10_000, 20_000, 30_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="lerobot/libero")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def select_latest_episodes(
    episodes: Any, episode_task_indices: dict[int, int], expected_tasks: int = 40
) -> dict[int, tuple[int, int]]:
    selections: dict[int, tuple[int, int]] = {}
    for row in episodes:
        episode = int(row["episode_index"])
        task = episode_task_indices.get(episode)
        if task is None:
            continue
        candidate = (episode, int(row["length"]))
        if task not in selections or episode > selections[task][0]:
            selections[task] = candidate
    if set(selections) != set(range(expected_tasks)):
        missing = sorted(set(range(expected_tasks)) - set(selections))
        extra = sorted(set(selections) - set(range(expected_tasks)))
        raise ValueError(f"Expected task indices 0..{expected_tasks - 1}; missing={missing}, extra={extra}")
    return selections


def hash_sampler_prefixes(
    sampler: Any,
    batch_size: int,
    registered_steps: tuple[int, ...] = REGISTERED_STEPS,
) -> dict[int, str]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not registered_steps or tuple(sorted(set(registered_steps))) != registered_steps:
        raise ValueError("registered_steps must be strictly increasing")
    targets = {step * batch_size: step for step in registered_steps}
    maximum = max(targets)
    hasher = hashlib.sha256()
    output = {}
    consumed = 0
    while consumed < maximum:
        for index in sampler:
            hasher.update(np.asarray([int(index)], dtype="<i8").tobytes())
            consumed += 1
            if consumed in targets:
                output[targets[consumed]] = hasher.hexdigest()
            if consumed == maximum:
                break
    return output


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise ValueError("Repository must be clean before fixture materialization")

    from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    metadata = LeRobotDatasetMetadata(
        args.dataset_repo,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    preprocessor, _ = make_pre_post_processors(
        config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    episode_task_indices = build_episode_task_indices(args.dataset_root)
    selections = select_latest_episodes(metadata.episodes, episode_task_indices)
    full_dataset = LeRobotDataset(
        args.dataset_repo,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        revision=args.dataset_revision,
        return_uint8=True,
    )
    sampler = EpisodeAwareSampler(
        full_dataset.meta.episodes["dataset_from_index"],
        full_dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=full_dataset.episodes,
        drop_n_last_frames=getattr(config, "drop_n_last_frames", 0),
        shuffle=True,
        seed=args.seed,
        absolute_to_relative_idx=full_dataset.absolute_to_relative_idx,
    )
    order_hashes = hash_sampler_prefixes(sampler, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    example = None
    for task_index in sorted(selections):
        episode, length = selections[task_index]
        dataset = LeRobotDataset(
            args.dataset_repo,
            root=args.dataset_root,
            episodes=[episode],
            delta_timestamps=delta_timestamps,
            revision=args.dataset_revision,
            return_uint8=True,
        )
        sample_index = length // 2
        raw = dataset[sample_index]
        batch = preprocess_raw(raw, raw["task"], metadata.camera_keys, preprocessor)
        path = args.output_dir / f"task{task_index:02d}.pt"
        torch.save(batch, path)
        records.append(
            {
                "task_index": task_index,
                "task": raw["task"],
                "episode": episode,
                "episode_length": length,
                "sample_index_within_episode": sample_index,
                "absolute_frame_index": int(raw["index"]),
                "fixture": str(path.resolve()),
                "fixture_sha256": file_sha256(path),
            }
        )
        example = batch

    manifest = {
        "schema_version": 1,
        "status": "MATURATION_FACTUAL_FIXTURE_FROZEN",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_sha256": file_sha256(args.checkpoint / "model.safetensors"),
        "dataset_repo": args.dataset_repo,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_revision": args.dataset_revision,
        "dataset_fps": metadata.fps,
        "dataset_frame_count": full_dataset.num_frames,
        "dataset_episode_count": full_dataset.num_episodes,
        "delta_timestamps": delta_timestamps,
        "selection_rule": "highest episode index and middle valid frame for each task index 0..39",
        "task_count": len(records),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "registered_steps": list(REGISTERED_STEPS),
        "data_order_prefix_sha256": {str(step): digest for step, digest in order_hashes.items()},
        "records": records,
        "example_batch": describe_batch(example),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "task_count": len(records),
                "data_order_30k_sha256": order_hashes[30_000],
                "manifest": str(args.manifest.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
