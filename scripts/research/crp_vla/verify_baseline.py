#!/usr/bin/env python
"""Verify deterministic SmolVLA inference at 10/5/2/1 NFE.

The input is a preprocessed tensor batch saved with ``torch.save``. It must
contain the observation tensors accepted by ``SmolVLAPolicy.predict_action_chunk``.
The script fixes the initial action noise across every NFE and repeat.
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
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
    parser.add_argument("--repeats", type=int, default=2, help="Determinism repeats per NFE")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--order-seed", type=int, default=0)
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


def distribution(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "median": statistics.median(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "all": values,
    }


def run_sample(policy, batch: dict, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
    policy.config.num_steps = num_steps
    policy.reset()
    return policy.predict_action_chunk(dict(batch), noise=noise.clone())


def profile_components(
    policy,
    batch: dict,
    noise: torch.Tensor,
    num_steps: int,
    repeats: int,
    device: str,
) -> dict:
    """Measure denoising calls separately from prefix encoding and fixed overhead."""
    total_values = []
    denoise_values = []
    fixed_values = []
    original_denoise_step = policy.model.denoise_step
    try:
        for _ in range(repeats):
            events = []

            def timed_denoise_step(*args, _events=events, **kwargs):
                if device.startswith("cuda"):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = original_denoise_step(*args, **kwargs)
                    end.record()
                    _events.append((start, end))
                    return output
                started = time.perf_counter()
                output = original_denoise_step(*args, **kwargs)
                _events.append((started, time.perf_counter()))
                return output

            policy.model.denoise_step = timed_denoise_step
            synchronize(device)
            started = time.perf_counter()
            run_sample(policy, batch, noise, num_steps)
            synchronize(device)
            total_ms = (time.perf_counter() - started) * 1000.0
            if device.startswith("cuda"):
                denoise_ms = sum(start.elapsed_time(end) for start, end in events)
            else:
                denoise_ms = sum((end - start) * 1000.0 for start, end in events)
            total_values.append(total_ms)
            denoise_values.append(denoise_ms)
            fixed_values.append(max(0.0, total_ms - denoise_ms))
    finally:
        policy.model.denoise_step = original_denoise_step
        policy.reset()
    return {
        "total_ms": distribution(total_values),
        "denoising_ms": distribution(denoise_values),
        "prefix_and_fixed_overhead_ms": distribution(fixed_values),
        "method": "denoise_step CUDA events; total wall time minus denoising is prefix plus fixed overhead",
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("Use at least two repeats for a determinism check")
    if args.warmups < 1 or args.latency_repeats < 1 or args.profile_repeats < 1:
        raise ValueError("warmups, latency-repeats, and profile-repeats must be positive")
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
    records_by_step = {}
    for num_steps in args.steps:
        hashes = []
        for _ in range(args.repeats):
            actions = run_sample(policy, batch, noise, num_steps)
            hashes.append(tensor_hash(actions))
        deterministic = len(set(hashes)) == 1
        records_by_step[num_steps] = {
            "num_steps": num_steps,
            "action_sha256": hashes[0],
            "all_repeat_hashes": hashes,
            "deterministic_exact": deterministic,
        }
        if not deterministic:
            raise RuntimeError(f"Exact determinism failed at {num_steps} NFE")

    for num_steps in args.steps:
        for _ in range(args.warmups):
            run_sample(policy, batch, noise, num_steps)
        synchronize(args.device)

    measurement_order = [step for step in args.steps for _ in range(args.latency_repeats)]
    random.Random(args.order_seed).shuffle(measurement_order)
    latency_by_step = {step: [] for step in args.steps}
    for num_steps in measurement_order:
        synchronize(args.device)
        started = time.perf_counter()
        run_sample(policy, batch, noise, num_steps)
        synchronize(args.device)
        latency_by_step[num_steps].append((time.perf_counter() - started) * 1000.0)

    for num_steps in args.steps:
        latency = distribution(latency_by_step[num_steps])
        records_by_step[num_steps]["latency_ms"] = latency
        # Keep the legacy fields so older consumers remain compatible.
        records_by_step[num_steps]["latency_ms_median"] = latency["median"]
        records_by_step[num_steps]["latency_ms_all"] = latency["all"]
        records_by_step[num_steps]["component_profile"] = profile_components(
            policy,
            batch,
            noise,
            num_steps,
            args.profile_repeats,
            args.device,
        )

    records = [records_by_step[step] for step in args.steps]

    manifest = {
        "schema_version": 2,
        "status": "SMOKE",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "batch_file": str(args.batch_file.resolve()),
        "batch_sha256": hashlib.sha256(args.batch_file.read_bytes()).hexdigest(),
        "noise_seed": args.noise_seed,
        "noise_sha256": tensor_hash(noise),
        "warmups_per_nfe": args.warmups,
        "latency_repeats_per_nfe": args.latency_repeats,
        "profile_repeats_per_nfe": args.profile_repeats,
        "measurement_order_seed": args.order_seed,
        "measurement_order": measurement_order,
        "device": args.device,
        "torch_version": torch.__version__,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
