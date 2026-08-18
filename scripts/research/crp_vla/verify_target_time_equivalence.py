#!/usr/bin/env python
"""Verify that zero-init target-time conditioning preserves 1-NFE output."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    config = SmolVLAConfig.from_pretrained(args.checkpoint, revision=args.revision)
    config.device = args.device
    config.num_steps = 1
    config.training_objective = "snapflow"
    config.use_target_time_embedding = True
    if Path(args.checkpoint).is_dir():
        config.load_vlm_weights = False
    config.__post_init__()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config=config, revision=args.revision, strict=False
    )
    batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict) or OBS_STATE not in batch:
        raise ValueError("Expected a processed SmolVLA tensor batch")
    batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed)
    noise = torch.randn(
        (batch[OBS_STATE].shape[0], config.chunk_size, config.max_action_dim),
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        policy.reset()
        local_velocity_output = policy.predict_action_chunk(dict(batch), noise=noise.clone())
        policy.reset()
        global_velocity_output = policy.predict_action_chunk(
            dict(batch), noise=noise.clone(), target_time=0.0
        )
    max_abs_error = torch.max(torch.abs(local_velocity_output - global_velocity_output)).item()
    passed = max_abs_error <= args.atol
    result = {
        "schema_version": 1,
        "status": "SMOKE",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "noise_seed": args.noise_seed,
        "atol": args.atol,
        "max_abs_error": max_abs_error,
        "exact_equal": torch.equal(local_velocity_output, global_velocity_output),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"Zero-init target-time equivalence failed: max_abs_error={max_abs_error}")


if __name__ == "__main__":
    main()
