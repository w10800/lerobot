#!/usr/bin/env python
"""Run same-state Base-10/Snap-1 queries for one frozen diagnostic task shard."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from task9_common import file_sha256, load_json
from task12_common import COUNTERFACTUAL_HORIZONS
from trajectory_instrumentation import load_replan_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--task11-trace-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def query(policy: Any, batch: dict[str, Any], noise: torch.Tensor, *, nfe: int, target: float | None) -> torch.Tensor:
    policy.reset()
    policy.config.num_steps = nfe
    kwargs = {} if target is None else {"target_time": target}
    with torch.inference_mode():
        return policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone(), **kwargs).detach()


def main() -> None:
    args = parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("status") != "TASK12_PASS_BRANCH_PROTOCOL_FROZEN":
        raise RuntimeError("Task 12 protocol is not frozen")
    execution = load_json(args.execution_manifest)
    shard = next((item for item in execution["shards"] if item["shard_id"] == args.shard_id), None)
    if shard is None:
        raise ValueError(f"Unknown Task 11 shard {args.shard_id}")
    diagnostic_ids = set(load_json(args.diagnostic_subset)["case_ids"])
    case_ids = sorted(set(shard["case_ids"]) & diagnostic_ids)
    if len(case_ids) != 5:
        raise RuntimeError(f"Task 12 diagnostic shard must contain five cases, got {len(case_ids)}")
    traces = load_json(args.task11_trace_manifest)
    trace_by_key = {(item["case_id"], item["arm"]): item for item in traces["records"]}
    output = args.output_root.resolve() / "phase_b_queries" / args.shard_id
    if output.exists():
        raise RuntimeError(f"Task 12 Phase B shard output already exists: {output}")
    output.mkdir(parents=True)

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def load_policy(checkpoint: Path) -> tuple[Any, Any]:
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = args.device
        config.load_vlm_weights = False
        config.vlm_model_name = str(args.offline_vlm_model_dir.resolve())
        config.n_action_steps = 10
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=False)
        policy.eval()
        _, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        return policy, postprocessor

    base_policy, base_post = load_policy(args.base_checkpoint)
    snap_policy, snap_post = load_policy(args.selected_checkpoint)
    rows: list[dict[str, Any]] = []
    base_actions: list[np.ndarray] = []
    snap_actions: list[np.ndarray] = []
    started_ns = time.time_ns()
    for case_index, case_id in enumerate(case_ids, start=1):
        for origin_arm in ("base10", "snap1_20k"):
            trace_record = trace_by_key[(case_id, origin_arm)]
            if file_sha256(Path(trace_record["trace_path"])) != trace_record["trace_hash"]:
                raise RuntimeError(f"Task 11 trace hash mismatch: {case_id}/{origin_arm}")
            top = load_json(Path(trace_record["trace_path"]))
            for replan in top["replan_records"]:
                metadata, payload = load_replan_trace(Path(replan["path"]), device=args.device)
                batch = payload["normalized_observation"]
                noise = payload["policy_output"]["noise_tensor"]
                archived = payload["policy_output"]["raw_policy_output"]
                base_chunk = query(base_policy, batch, noise, nfe=10, target=None)
                snap_chunk = query(snap_policy, batch, noise, nfe=1, target=0.0)
                self_chunk = base_chunk if origin_arm == "base10" else snap_chunk
                if not torch.equal(self_chunk, archived):
                    maximum = float((self_chunk.float() - archived.float()).abs().max().item())
                    raise RuntimeError(
                        f"Task 12 self-replay mismatch {case_id}/{origin_arm}/"
                        f"{metadata['identity']['replan_index']}: max_abs={maximum}"
                    )
                base_denorm = base_post(base_chunk[0, :10].detach().clone()).float().cpu().numpy()
                snap_denorm = snap_post(snap_chunk[0, :10].detach().clone()).float().cpu().numpy()
                base_raw = base_chunk[0].float().cpu().numpy()
                snap_raw = snap_chunk[0].float().cpu().numpy()
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "task_id": shard["task_id"],
                    "origin_arm": origin_arm,
                    "replan_index": int(metadata["identity"]["replan_index"]),
                    "simulator_timestep": int(metadata["identity"]["simulator_timestep"]),
                    "self_replay_exact": True,
                    "state_payload_sha256": metadata["integrity"]["simulator_state_sha256"],
                    "noise_sha256": file_sha256(Path(replan["path"]) / "sha256_manifest.txt"),
                }
                for horizon in COUNTERFACTUAL_HORIZONS:
                    row[f"raw_prefix_l2_h{horizon}"] = float(
                        np.linalg.norm(base_raw[:horizon] - snap_raw[:horizon])
                    )
                    row[f"denorm_prefix_l2_h{horizon}"] = float(
                        np.linalg.norm(base_denorm[:horizon] - snap_denorm[:horizon])
                    )
                difference = base_denorm - snap_denorm
                row["denorm_translation_l2_h10"] = float(np.linalg.norm(difference[:, :3]))
                row["denorm_rotation_l2_h10"] = float(np.linalg.norm(difference[:, 3:6]))
                row["denorm_gripper_l2_h10"] = float(np.linalg.norm(difference[:, 6]))
                row["array_index"] = len(rows)
                rows.append(row)
                base_actions.append(base_denorm)
                snap_actions.append(snap_denorm)
        print(
            f"shard={args.shard_id} diagnostic_case={case_index}/5 same_state_queries=complete",
            flush=True,
        )
    with (output / "same_state_metrics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    np.savez_compressed(
        output / "branch_actions.npz",
        base10=np.stack(base_actions),
        snap1_20k=np.stack(snap_actions),
    )
    status = {
        "schema_version": 1,
        "status": "TASK12_PHASE_B_QUERY_SHARD_COMPLETE",
        "shard_id": args.shard_id,
        "task_id": shard["task_id"],
        "case_count": len(case_ids),
        "replan_state_count": len(rows),
        "policy_query_count": len(rows) * 2,
        "self_replay_exact_count": len(rows),
        "protocol_sha256": file_sha256(args.protocol),
        "metrics_sha256": file_sha256(output / "same_state_metrics.jsonl"),
        "actions_sha256": file_sha256(output / "branch_actions.npz"),
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "new_rollout": False,
        "new_training": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
