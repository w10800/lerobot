#!/usr/bin/env python
"""Build a provenance manifest over frozen-teacher response archives."""

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalization_hash(checkpoint: Path) -> tuple[str, list[str]]:
    paths = [
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing normalization artifacts: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), [str(path.resolve()) for path in paths]


def batch_manifest(batch_path: str) -> dict:
    path = Path(batch_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-response-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    norm_hash, norm_files = normalization_hash(args.checkpoint)
    records = []
    for path in sorted(args.inputs):
        diagnostic = json.loads(path.read_text())
        factual = batch_manifest(diagnostic["factual_batch"])
        counterfactual = batch_manifest(diagnostic["counterfactual_batch"])
        for key in ("dataset", "dataset_revision", "episode", "sample_index_within_selection"):
            if factual[key] != counterfactual[key]:
                raise ValueError(f"Unmatched batch manifests for {path}: {key}")
        teacher_norm = diagnostic["metrics"]["global"]["teacher_response_norm"]
        records.append(
            {
                "pair_id": (
                    f"{diagnostic['task_group']}:{diagnostic['task_id']}:{diagnostic['intervention_type']}"
                ),
                "dataset": factual["dataset"],
                "dataset_revision": factual["dataset_revision"],
                "episode": factual["episode"],
                "frame_selector": factual["sample_index_within_selection"],
                "factual_prompt": factual["effective_task"],
                "counterfactual_prompt": counterfactual["effective_task"],
                "noise_seed": diagnostic["noise_seed"],
                "noise_sha256": diagnostic["noise_sha256"],
                "teacher_checkpoint_revision": diagnostic["teacher"]["revision"],
                "teacher_checkpoint_sha256": diagnostic["teacher"]["model_sha256"],
                "teacher_nfe": diagnostic["teacher"]["num_steps"],
                "normalization_version_sha256": norm_hash,
                "action_archive": diagnostic["action_archive"],
                "action_archive_sha256": diagnostic["action_archive_sha256"],
                "teacher_response_norm": teacher_norm,
                "accepted": teacher_norm >= args.teacher_response_threshold,
                "diagnostic_manifest": str(path.resolve()),
                "diagnostic_manifest_sha256": file_sha256(path),
            }
        )
    output = {
        "schema_version": 1,
        "status": "CACHE_MANIFEST",
        "teacher_response_threshold": args.teacher_response_threshold,
        "normalization_version_sha256": norm_hash,
        "normalization_files": norm_files,
        "record_count": len(records),
        "accepted_count": sum(record["accepted"] for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
