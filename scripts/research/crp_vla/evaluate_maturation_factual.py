#!/usr/bin/env python
"""Evaluate one registered maturation checkpoint on the frozen factual fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_STATE

NFES = (10, 2, 1)
NOISE_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency-warmups", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_batch(path: Path, expected_sha256: str, device: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Fixture hash mismatch: {path}")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict) or OBS_STATE not in batch or ACTION not in batch:
        raise ValueError(f"Invalid factual fixture: {path}")
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }
    return ensure_action_batch_dimension(batch)


def ensure_action_batch_dimension(batch: dict[str, Any]) -> dict[str, Any]:
    """Align dataset action fields with already-batched processed observations."""
    if batch[OBS_STATE].ndim < 2 or batch[OBS_STATE].shape[0] != 1:
        raise ValueError(f"Expected one processed observation, got {batch[OBS_STATE].shape}")
    if batch[ACTION].ndim == 2:
        batch[ACTION] = batch[ACTION].unsqueeze(0)
    if batch[ACTION].ndim != 3 or batch[ACTION].shape[0] != 1:
        raise ValueError(f"Expected one action chunk, got {batch[ACTION].shape}")
    action_pad = batch.get("action_is_pad")
    if isinstance(action_pad, torch.Tensor) and action_pad.ndim == 1:
        batch["action_is_pad"] = action_pad.unsqueeze(0)
    return batch


def fixed_noise_time(policy: Any, batch: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=batch[OBS_STATE].device).manual_seed(seed)
    noise = torch.randn(
        (batch[OBS_STATE].shape[0], policy.config.chunk_size, policy.config.max_action_dim),
        generator=generator,
        device=batch[OBS_STATE].device,
        dtype=torch.float32,
    )
    time_value = torch.rand(
        batch[OBS_STATE].shape[0],
        generator=generator,
        device=batch[OBS_STATE].device,
        dtype=torch.float32,
    )
    return noise, time_value


def mse_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_dim: int,
    execution_horizon: int = 10,
    prefix: str = "normalized",
) -> dict[str, float]:
    predicted = predicted[..., :action_dim].float()
    target = target[..., :action_dim].float()
    common = min(predicted.shape[1], target.shape[1])
    predicted = predicted[:, :common]
    target = target[:, :common]
    return {
        f"full_{prefix}_mse": float(torch.mean((predicted - target) ** 2).cpu()),
        f"executed_prefix_{prefix}_mse": float(
            torch.mean((predicted[:, :execution_horizon] - target[:, :execution_horizon]) ** 2).cpu()
        ),
    }


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    policy_dir = checkpoint / "pretrained_model"
    return {
        path.name: file_sha256(path)
        for path in sorted(policy_dir.glob("policy_*processor*"))
        if path.is_file()
    }


def checkpoint_files(checkpoint: Path) -> dict[str, dict[str, str]]:
    required = {
        "model": checkpoint / "pretrained_model" / "model.safetensors",
        "optimizer": checkpoint / "training_state" / "optimizer_state.safetensors",
        "scheduler": checkpoint / "training_state" / "scheduler_state.json",
        "rng": checkpoint / "training_state" / "rng_state.safetensors",
        "training_step": checkpoint / "training_state" / "training_step.json",
        "config": checkpoint / "pretrained_model" / "train_config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint provenance files missing: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in required.items()
    }


def latency_benchmark(
    policy: Any,
    batch: dict[str, Any],
    nfe: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    noise, _ = fixed_noise_time(policy, batch, 20_260_814 + nfe)
    kwargs = {"target_time": 0.0} if nfe == 1 else {}
    policy.config.num_steps = nfe
    values = []
    for index in range(warmups + repeats):
        policy.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            policy.predict_action_chunk(batch, noise=noise, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if index >= warmups:
            values.append(elapsed)
    return {
        "warmups": warmups,
        "repeats": repeats,
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.latency_warmups < 0 or args.latency_repeats < 1:
        raise ValueError("Invalid latency warmup/repeat counts")
    fixture = json.loads(args.fixture_manifest.read_text())
    if fixture.get("status") != "MATURATION_FACTUAL_FIXTURE_FROZEN" or fixture.get("task_count") != 40:
        raise ValueError("Expected the frozen 40-task maturation fixture")
    policy_dir = args.checkpoint / "pretrained_model"

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(policy_dir)
    config.device = args.device
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(policy_dir, config=config, strict=False)
    _, postprocessor = make_pre_post_processors(config, str(policy_dir))
    policy.eval()
    action_dim = int(config.action_feature.shape[0])
    records = []
    repeatability = None
    for fixture_record in fixture["records"]:
        batch = load_batch(
            Path(fixture_record["fixture"]), fixture_record["fixture_sha256"], args.device
        )
        target = batch[ACTION]
        for seed in NOISE_SEEDS:
            noise, time_value = fixed_noise_time(policy, batch, seed)
            with torch.inference_mode():
                loss, loss_dict = policy(batch, noise=noise, time=time_value)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss at task {fixture_record['task_index']}")
            for nfe in NFES:
                policy.reset()
                policy.config.num_steps = nfe
                kwargs = {"target_time": 0.0} if nfe == 1 else {}
                with torch.inference_mode():
                    predicted = policy.predict_action_chunk(batch, noise=noise, **kwargs)
                if not torch.isfinite(predicted).all():
                    raise FloatingPointError(
                        f"Non-finite predicted action at task {fixture_record['task_index']} nfe={nfe}"
                    )
                raw_predicted = postprocessor(predicted)
                raw_target = postprocessor(target)
                metrics = {
                    **mse_metrics(predicted, target, action_dim),
                    **mse_metrics(raw_predicted, raw_target, action_dim, prefix="raw_7d"),
                }
                records.append(
                    {
                        "task_index": fixture_record["task_index"],
                        "task": fixture_record["task"],
                        "noise_seed": seed,
                        "nfe": nfe,
                        "total_validation_loss": float(loss.cpu()),
                        "fm_loss": float(loss_dict["loss/fm"]),
                        "shortcut_loss": float(loss_dict["loss/shortcut"]),
                        "accounted_total": float(loss_dict["loss/accounted_total"]),
                        "accounting_residual": float(loss_dict["loss/accounting_residual"]),
                        **metrics,
                    }
                )
                if repeatability is None and nfe == 1:
                    policy.reset()
                    with torch.inference_mode():
                        repeated = policy.predict_action_chunk(batch, noise=noise, target_time=0.0)
                    repeatability = {
                        "exact": bool(torch.equal(predicted, repeated)),
                        "max_abs_difference": float(torch.max(torch.abs(predicted - repeated)).cpu()),
                    }
    if not repeatability or not repeatability["exact"]:
        raise RuntimeError(f"Fixed-noise inference is not exactly repeatable: {repeatability}")

    latency_batch = load_batch(
        Path(fixture["records"][0]["fixture"]),
        fixture["records"][0]["fixture_sha256"],
        args.device,
    )
    latency = {
        str(nfe): latency_benchmark(
            policy, latency_batch, nfe, args.latency_warmups, args.latency_repeats
        )
        for nfe in NFES
    }
    summary = {}
    for nfe in NFES:
        subset = [record for record in records if record["nfe"] == nfe]
        summary[str(nfe)] = {
            key: float(np.mean([record[key] for record in subset]))
            for key in (
                "total_validation_loss",
                "fm_loss",
                "shortcut_loss",
                "accounted_total",
                "accounting_residual",
                "full_normalized_mse",
                "executed_prefix_normalized_mse",
                "full_raw_7d_mse",
                "executed_prefix_raw_7d_mse",
            )
        }
    output = {
        "schema_version": 1,
        "status": "FACTUAL_CHECKPOINT_EVALUATED",
        "checkpoint_step": args.checkpoint_step,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_files": checkpoint_files(args.checkpoint),
        "processor_manifest": processor_manifest(args.checkpoint),
        "fixture_manifest": str(args.fixture_manifest.resolve()),
        "fixture_manifest_sha256": file_sha256(args.fixture_manifest),
        "task_count": 40,
        "noise_seeds": list(NOISE_SEEDS),
        "record_count": len(records),
        "repeatability": repeatability,
        "summary": summary,
        "latency": latency,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": args.checkpoint_step, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
