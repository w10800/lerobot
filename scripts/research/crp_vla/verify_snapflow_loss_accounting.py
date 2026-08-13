#!/usr/bin/env python
"""Verify SnapFlow scalar-loss accounting on one fixed processed batch."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    config.load_vlm_weights = False
    config.training_objective = "snapflow"
    config.use_target_time_embedding = True
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config=config,
        revision=args.checkpoint_revision,
        strict=False,
    )
    raw_batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    if not isinstance(raw_batch, dict) or OBS_STATE not in raw_batch or ACTION not in raw_batch:
        raise ValueError("Expected a processed training batch with observation.state and action")
    batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
    }
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    prepared_action = policy.prepare_action(batch)
    noise = torch.randn(
        prepared_action.shape,
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    time = torch.rand(
        prepared_action.shape[0],
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        loss, loss_dict = policy.forward(dict(batch), noise=noise, time=time)
    residual = float(loss_dict["loss/accounting_residual"])
    status = "PASS" if residual < args.tolerance else "FAIL"
    output = {
        "schema_version": 1,
        "status": status,
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "batch_file": str(args.batch_file.resolve()),
        "batch_sha256": hashlib.sha256(args.batch_file.read_bytes()).hexdigest(),
        "seed": args.seed,
        "device": args.device,
        "tolerance": args.tolerance,
        "loss": float(loss.detach().cpu()),
        "loss_components": {
            key: value for key, value in loss_dict.items() if key.startswith("loss/")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
