#!/usr/bin/env python
"""Materialize matched processed fixtures for the balanced diagnostic catalog."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def load_catalog(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pair_ids = [record["pair_id"] for record in records]
    if not records or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Catalog must contain at least one record and unique pair IDs")
    return records


def select_episode(episodes: Any, source_task: str) -> tuple[int, int]:
    matches = [
        (int(row["episode_index"]), int(row["length"]))
        for row in episodes
        if source_task in row["tasks"]
    ]
    if not matches:
        raise ValueError(f"No episode contains exact source task {source_task!r}")
    return min(matches)


def preprocess_raw(raw: dict, task: str, camera_keys: list[str], preprocessor) -> dict:
    candidate = dict(raw)
    candidate["task"] = task
    for camera_key in camera_keys:
        value = candidate.get(camera_key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            candidate[camera_key] = value.to(dtype=torch.float32) / 255.0
    processed = preprocessor(candidate)
    return {
        key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }


def non_condition_hash(batch: dict) -> str:
    digest = hashlib.sha256()
    condition_keys = {
        "task",
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    for key, value in sorted(batch.items()):
        if key in condition_keys:
            continue
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def describe_batch(batch: dict) -> dict:
    return {
        key: (
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
            if isinstance(value, torch.Tensor)
            else {"type": type(value).__name__, "value": value}
        )
        for key, value in sorted(batch.items())
    }


def main() -> None:
    args = parse_args()
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla_crp.paired_inputs import assert_matched_batches

    records = load_catalog(args.catalog)
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "model.safetensors")

    dataset_repo = records[0]["dataset_repo"]
    dataset_revision = records[0]["dataset_revision"]
    if any(
        record["dataset_repo"] != dataset_repo or record["dataset_revision"] != dataset_revision
        for record in records
    ):
        raise ValueError("Catalog mixes dataset repositories or revisions")

    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    metadata = LeRobotDatasetMetadata(
        dataset_repo,
        root=args.dataset_root,
        revision=dataset_revision,
    )
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    preprocessor, _ = make_pre_post_processors(
        config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    selections = {
        source_task: select_episode(metadata.episodes, source_task)
        for source_task in sorted({record["source_task"] for record in records})
    }
    datasets = {
        episode: LeRobotDataset(
            dataset_repo,
            root=args.dataset_root,
            episodes=[episode],
            delta_timestamps=delta_timestamps,
            revision=dataset_revision,
            return_uint8=True,
        )
        for episode, _ in sorted(set(selections.values()))
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for record in records:
        episode, episode_length = selections[record["source_task"]]
        sample_index = episode_length // 2
        raw = datasets[episode][sample_index]
        if raw["task"] != record["source_task"]:
            raise ValueError(
                f"Selected frame task mismatch for {record['pair_id']}: "
                f"{raw['task']!r} != {record['source_task']!r}"
            )
        factual = preprocess_raw(
            raw,
            record["factual_prompt"],
            metadata.camera_keys,
            preprocessor,
        )
        counterfactual = preprocess_raw(
            raw,
            record["counterfactual_prompt"],
            metadata.camera_keys,
            preprocessor,
        )
        assert_matched_batches(factual, counterfactual)
        factual_hash = non_condition_hash(factual)
        counterfactual_hash = non_condition_hash(counterfactual)
        if factual_hash != counterfactual_hash:
            raise RuntimeError(f"Non-condition hash mismatch for {record['pair_id']}")

        factual_path = args.output_dir / f"{record['pair_id']}.factual.pt"
        counterfactual_path = args.output_dir / f"{record['pair_id']}.counterfactual.pt"
        torch.save(factual, factual_path)
        torch.save(counterfactual, counterfactual_path)
        outputs.append(
            {
                **record,
                "episode": episode,
                "episode_length": episode_length,
                "sample_index_within_selection": sample_index,
                "absolute_frame_index": int(raw["index"]),
                "non_condition_sha256": factual_hash,
                "factual_batch": str(factual_path.resolve()),
                "factual_batch_sha256": file_sha256(factual_path),
                "counterfactual_batch": str(counterfactual_path.resolve()),
                "counterfactual_batch_sha256": file_sha256(counterfactual_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "FIXTURES_MATERIALIZED",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": file_sha256(args.catalog),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_fps": metadata.fps,
        "delta_timestamps": delta_timestamps,
        "pair_count": len(outputs),
        "unique_source_task_count": len(selections),
        "records": outputs,
        "example_batch": describe_batch(factual),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pair_count": manifest["pair_count"],
                "unique_source_task_count": manifest["unique_source_task_count"],
                "manifest": str(args.manifest.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
