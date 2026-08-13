#!/usr/bin/env python
"""Verify deterministic SmolVLA inference at 10/5/2/1 NFE.

The input is a preprocessed tensor batch saved with ``torch.save``. It must
contain the observation tensors accepted by ``SmolVLAPolicy.predict_action_chunk``.
The script fixes the initial action noise across every NFE and repeat.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument("--revision", required=True, help="Immutable HF revision or local checkpoint label")
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 5, 2, 1])
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("Use at least two repeats for a determinism check")
    if any(step < 1 for step in args.steps):
        raise ValueError("All NFE values must be positive")

    config = SmolVLAConfig.from_pretrained(args.checkpoint, revision=args.revision)
    config.device = args.device
    if Path(args.checkpoint).is_dir():
        # The local safetensors file contains the full VLM; initialize architecture only,
        # then load the complete checkpoint instead of downloading base VLM weights first.
        config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config=config, revision=args.revision, strict=False
    )
    raw_batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    if not isinstance(raw_batch, dict) or OBS_STATE not in raw_batch:
        raise ValueError(f"Expected a processed tensor batch containing {OBS_STATE!r}")
    batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
    }

    batch_size = batch[OBS_STATE].shape[0]
    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed)
    noise = torch.randn(
        (batch_size, config.chunk_size, config.max_action_dim),
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    records = []
    for num_steps in args.steps:
        hashes = []
        latencies_ms = []
        for _ in range(args.repeats):
            policy.config.num_steps = num_steps
            policy.reset()
            synchronize(args.device)
            started = time.perf_counter()
            actions = policy.predict_action_chunk(dict(batch), noise=noise.clone())
            synchronize(args.device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            hashes.append(tensor_hash(actions))
        deterministic = len(set(hashes)) == 1
        records.append(
            {
                "num_steps": num_steps,
                "action_sha256": hashes[0],
                "all_repeat_hashes": hashes,
                "deterministic_exact": deterministic,
                "latency_ms_median": statistics.median(latencies_ms),
                "latency_ms_all": latencies_ms,
            }
        )
        if not deterministic:
            raise RuntimeError(f"Exact determinism failed at {num_steps} NFE")

    manifest = {
        "schema_version": 1,
        "status": "SMOKE",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "batch_file": str(args.batch_file.resolve()),
        "batch_sha256": hashlib.sha256(args.batch_file.read_bytes()).hexdigest(),
        "noise_seed": args.noise_seed,
        "noise_sha256": tensor_hash(noise),
        "device": args.device,
        "torch_version": torch.__version__,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
