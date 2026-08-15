#!/usr/bin/env python
"""Run the single frozen Task 12 student-state action-preservation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prepare_task12_probe import STATUS as PROTOCOL_STATUS
from safetensors.torch import load_file
from task9_common import file_sha256, load_json
from trajectory_instrumentation import load_replan_trace

TENSOR_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.language.attention_mask",
    "observation.language.tokens",
    "observation.state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task11-trace-manifest", type=Path, required=True)
    parser.add_argument("--phase-b-query-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def normalize_mean_std(
    action: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    return (np.asarray(action, dtype=np.float32) - mean) / (std + eps)


def tensor_digest(parameters: list[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_action_stats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    path = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    tensors = load_file(str(path))
    return tensors["action.mean"].cpu().numpy(), tensors["action.std"].cpu().numpy()


def select_denormalized_targets(
    origin: str, base_actions: np.ndarray, snap_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the probe target, original-Snap anchor, and pool kind."""
    if origin == "snap1_20k":
        return base_actions, snap_actions, "preservation"
    if origin == "base10":
        return snap_actions, snap_actions, "anchor"
    raise ValueError(f"Unexpected Task 12 origin arm: {origin}")


def build_trace_index(trace_manifest: Path) -> dict[tuple[str, str, int], Path]:
    manifest = load_json(trace_manifest)
    index: dict[tuple[str, str, int], Path] = {}
    for record in manifest["records"]:
        top_path = Path(record["trace_path"])
        if file_sha256(top_path) != record["trace_hash"]:
            raise RuntimeError(f"Task 11 trace hash mismatch: {top_path}")
        top = load_json(top_path)
        for position, replan in enumerate(top["replan_records"]):
            key = (record["case_id"], record["arm"], position)
            if key in index:
                raise RuntimeError(f"Duplicate replan identity: {key}")
            index[key] = Path(replan["path"])
    return index


def load_samples(
    protocol: dict[str, Any],
    query_root: Path,
    trace_index: dict[tuple[str, str, int], Path],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    split_by_case = dict.fromkeys(protocol["split"]["train_case_ids"], "train")
    split_by_case.update(dict.fromkeys(protocol["split"]["heldout_case_ids"], "heldout"))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_paths = sorted(query_root.glob("shard-*/status.json"))
    if len(status_paths) != 40:
        raise RuntimeError(f"Expected 40 Phase B query shards, found {len(status_paths)}")
    for status_path in status_paths:
        status = load_json(status_path)
        if status.get("status") != "TASK12_PHASE_B_QUERY_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete query shard: {status_path}")
        shard_dir = status_path.parent
        metrics_path = shard_dir / "same_state_metrics.jsonl"
        actions_path = shard_dir / "branch_actions.npz"
        if (
            file_sha256(metrics_path) != status["metrics_sha256"]
            or file_sha256(actions_path) != status["actions_sha256"]
        ):
            raise RuntimeError(f"Phase B query hash mismatch: {shard_dir}")
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        with np.load(actions_path) as arrays:
            base_actions = arrays["base10"]
            snap_actions = arrays["snap1_20k"]
        if len(rows) != len(base_actions) or len(rows) != len(snap_actions):
            raise RuntimeError(f"Query row/action count mismatch: {shard_dir}")
        for row in rows:
            case_id = row["case_id"]
            split = split_by_case.get(case_id)
            if split is None:
                raise RuntimeError(f"Unexpected diagnostic case: {case_id}")
            origin = row["origin_arm"]
            array_index = int(row["array_index"])
            replan_index = int(row["replan_index"])
            trace_path = trace_index[(case_id, origin, replan_index)]
            target_denorm, self_denorm, kind = select_denormalized_targets(
                origin, base_actions[array_index], snap_actions[array_index]
            )
            pools[f"{split}_{kind}"].append(
                {
                    "case_id": case_id,
                    "origin_arm": origin,
                    "replan_index": replan_index,
                    "trace_path": trace_path,
                    "target": normalize_mean_std(target_denorm[:10], mean, std),
                    "self_target": normalize_mean_std(self_denorm[:10], mean, std),
                }
            )
    required = {f"{split}_{kind}" for split in ("train", "heldout") for kind in ("preservation", "anchor")}
    if set(pools) != required or any(not pools[key] for key in required):
        raise RuntimeError(f"Incomplete training pools: {sorted(pools)}")
    return dict(pools)


def load_model_input(sample: dict[str, Any], device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    metadata, payload = load_replan_trace(sample["trace_path"], device="cpu")
    identity = metadata["identity"]
    expected = (sample["case_id"], sample["origin_arm"], sample["replan_index"])
    observed = (identity["case_id"], identity["arm_id"], int(identity["replan_index"]))
    if observed != expected:
        raise RuntimeError(f"Trace identity mismatch: {observed} != {expected}")
    observation = payload["normalized_observation"]
    batch = {key: observation[key].to(device) for key in TENSOR_KEYS}
    noise = payload["policy_output"]["noise_tensor"].to(device)
    return batch, noise


def merge_inputs(
    loaded: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return (
        {key: torch.cat([item[0][key] for item in loaded], dim=0) for key in TENSOR_KEYS},
        torch.cat([item[1] for item in loaded], dim=0),
    )


def predict_with_grad(policy: Any, batch: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.model.sample_actions(
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state,
        noise=noise,
        target_time=0.0,
    )
    return actions[:, :10, : policy.config.action_feature.shape[0]]


def evaluate_pool(policy: Any, samples: list[dict[str, Any]], device: str) -> dict[str, Any]:
    by_case_target: dict[str, list[float]] = defaultdict(list)
    by_case_self: dict[str, list[float]] = defaultdict(list)
    policy.eval()
    with torch.inference_mode():
        for sample in samples:
            batch, noise = load_model_input(sample, device)
            prediction = predict_with_grad(policy, batch, noise).float()
            target = torch.from_numpy(sample["target"]).to(device).unsqueeze(0)
            self_target = torch.from_numpy(sample["self_target"]).to(device).unsqueeze(0)
            by_case_target[sample["case_id"]].append(float(torch.mean((prediction - target) ** 2).item()))
            by_case_self[sample["case_id"]].append(float(torch.mean((prediction - self_target) ** 2).item()))
    target_case_means = [float(np.mean(values)) for values in by_case_target.values()]
    self_case_means = [float(np.mean(values)) for values in by_case_self.values()]
    return {
        "case_count": len(by_case_target),
        "state_count": len(samples),
        "target_mse_case_mean": float(np.mean(target_case_means)),
        "target_mse_case_median": float(np.median(target_case_means)),
        "original_snap_mse_case_mean": float(np.mean(self_case_means)),
        "original_snap_mse_case_median": float(np.median(self_case_means)),
    }


def copy_processor_contract(source: Path, destination: Path) -> dict[str, str]:
    for path in sorted(source.glob("policy_*processor*")):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    tokenizer = source / "tokenizer"
    if tokenizer.is_dir():
        shutil.copytree(tokenizer, destination / "tokenizer")
    return {
        path.name: file_sha256(path)
        for path in sorted(destination.glob("policy_*processor*"))
        if path.is_file()
    }


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before the admitted Task 12 probe")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Task 12 probe: {args.output_root}")
    protocol = load_json(args.protocol)
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("decision") != "D-019":
        raise RuntimeError("Task 12 preservation-probe protocol is not frozen")
    checkpoint = args.selected_checkpoint.resolve()
    expected_checkpoint = protocol["selected_checkpoint"]
    if file_sha256(checkpoint / "model.safetensors") != expected_checkpoint["model_sha256"]:
        raise RuntimeError("Selected 20k checkpoint hash changed")
    settings = protocol["training"]
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    trace_index = build_trace_index(args.task11_trace_manifest)
    mean, std = load_action_stats(checkpoint)
    pools = load_samples(protocol, args.phase_b_query_root, trace_index, mean, std)
    args.output_root.mkdir(parents=True)

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(checkpoint)
    original_vlm_name = config.vlm_model_name
    config.device = args.device
    config.load_vlm_weights = False
    config.vlm_model_name = str(args.offline_vlm_model_dir.resolve())
    config.num_steps = 1
    config.n_action_steps = 10
    config.compile_model = False
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Task 12 probe has no trainable parameters")
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in policy.parameters())

    preflight_rng = np.random.default_rng(seed)
    combined_train = pools["train_preservation"] + pools["train_anchor"]
    preflight_indices = preflight_rng.choice(
        len(combined_train),
        size=min(
            int(protocol["preflight"]["same_state_self_replay_samples"]),
            len(combined_train),
        ),
        replace=False,
    )
    policy.eval()
    maximum_roundtrip_error = 0.0
    with torch.inference_mode():
        for index in preflight_indices:
            sample = combined_train[int(index)]
            batch, noise = load_model_input(sample, args.device)
            prediction = predict_with_grad(policy, batch, noise).float()[0]
            self_target = torch.from_numpy(sample["self_target"]).to(args.device)
            maximum_roundtrip_error = max(
                maximum_roundtrip_error,
                float(torch.max(torch.abs(prediction - self_target)).item()),
            )
    allowed = float(protocol["preflight"]["maximum_raw_action_roundtrip_abs_error"])
    if maximum_roundtrip_error > allowed:
        raise RuntimeError(f"Same-state replay roundtrip error {maximum_roundtrip_error} exceeds {allowed}")

    policy.train()
    zero_samples = [pools["train_preservation"][0], pools["train_anchor"][0]]
    zero_batch, zero_noise = merge_inputs([load_model_input(sample, args.device) for sample in zero_samples])
    zero_targets = torch.stack([torch.from_numpy(sample["target"]) for sample in zero_samples]).to(
        args.device
    )
    before_zero_hash = tensor_digest(trainable)
    zero_optimizer = torch.optim.AdamW(trainable, lr=0.0, weight_decay=0.0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        zero_prediction = predict_with_grad(policy, zero_batch, zero_noise).float()
        zero_loss = torch.mean((zero_prediction - zero_targets) ** 2)
    if not torch.isfinite(zero_loss):
        raise RuntimeError("Non-finite zero-learning-rate preflight loss")
    zero_loss.backward()
    zero_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(trainable, float(settings["gradient_clip_norm"]))
    )
    if not math.isfinite(zero_gradient_norm) or zero_gradient_norm <= 0:
        raise RuntimeError("Task 12 probe gradient path is not finite and nonzero")
    zero_optimizer.step()
    zero_optimizer.zero_grad(set_to_none=True)
    after_zero_hash = tensor_digest(trainable)
    if before_zero_hash != after_zero_hash:
        raise RuntimeError("Zero-learning-rate preflight changed trainable tensors")
    del zero_optimizer, zero_batch, zero_noise, zero_targets, zero_prediction, zero_loss

    heldout_before = {
        "preservation": evaluate_pool(policy, pools["heldout_preservation"], args.device),
        "anchor": evaluate_pool(policy, pools["heldout_anchor"], args.device),
    }
    policy.train()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        betas=tuple(float(value) for value in settings["betas"]),
        eps=float(settings["eps"]),
        weight_decay=float(settings["weight_decay"]),
    )
    training_rng = np.random.default_rng(seed)
    log_rows: list[dict[str, Any]] = []
    started_ns = time.time_ns()
    for step in range(1, int(settings["steps"]) + 1):
        preservation = pools["train_preservation"][
            int(training_rng.integers(len(pools["train_preservation"])))
        ]
        anchor = pools["train_anchor"][int(training_rng.integers(len(pools["train_anchor"])))]
        samples = [preservation, anchor]
        batch, noise = merge_inputs([load_model_input(sample, args.device) for sample in samples])
        targets = torch.stack([torch.from_numpy(sample["target"]) for sample in samples]).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            prediction = predict_with_grad(policy, batch, noise).float()
            preservation_loss = torch.mean((prediction[0] - targets[0]) ** 2)
            anchor_loss = torch.mean((prediction[1] - targets[1]) ** 2)
            loss = preservation_loss + anchor_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Task 12 probe loss at step {step}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(settings["gradient_clip_norm"]))
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite Task 12 probe gradient at step {step}")
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == int(settings["steps"]):
            row = {
                "step": step,
                "loss": float(loss.item()),
                "preservation_loss": float(preservation_loss.item()),
                "anchor_loss": float(anchor_loss.item()),
                "gradient_norm": gradient_norm,
            }
            log_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    heldout_after = {
        "preservation": evaluate_pool(policy, pools["heldout_preservation"], args.device),
        "anchor": evaluate_pool(policy, pools["heldout_anchor"], args.device),
    }
    checkpoint_dir = args.output_root / "checkpoint_000500" / "pretrained_model"
    checkpoint_dir.mkdir(parents=True)
    policy.config.vlm_model_name = original_vlm_name
    policy.config.device = "cuda"
    policy.save_pretrained(checkpoint_dir)
    processor_manifest = copy_processor_contract(checkpoint, checkpoint_dir)
    source_processors = {
        path.name: file_sha256(path)
        for path in sorted(checkpoint.glob("policy_*processor*"))
        if path.is_file()
    }
    if processor_manifest != source_processors:
        raise RuntimeError("Saved Task 12 processor contract differs from selected 20k")
    training_log = args.output_root / "training_metrics.jsonl"
    training_log.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in log_rows))
    result = {
        "schema_version": 1,
        "status": "TASK12_STUDENT_STATE_PRESERVATION_PROBE_COMPLETE",
        "classification": "EXPLORATORY",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "protocol_sha256": file_sha256(args.protocol),
        "source_checkpoint_sha256": expected_checkpoint["model_sha256"],
        "trained_checkpoint": str(checkpoint_dir.resolve()),
        "trained_model_sha256": file_sha256(checkpoint_dir / "model.safetensors"),
        "processor_manifest": processor_manifest,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "pool_counts": {key: len(value) for key, value in sorted(pools.items())},
        "preflight": {
            "same_state_sample_count": len(preflight_indices),
            "maximum_raw_action_roundtrip_abs_error": maximum_roundtrip_error,
            "zero_lr_tensor_sha256_before": before_zero_hash,
            "zero_lr_tensor_sha256_after": after_zero_hash,
            "zero_lr_gradient_norm": zero_gradient_norm,
        },
        "training": settings,
        "training_log_sha256": file_sha256(training_log),
        "heldout_before": heldout_before,
        "heldout_after": heldout_after,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "task11_reconfirmation": False,
        "libero_cf_training": False,
        "checkpoint_selection": False,
    }
    (args.output_root / "TASK12_PRESERVATION_PROBE_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "model_sha256": result["trained_model_sha256"]}))


if __name__ == "__main__":
    main()
