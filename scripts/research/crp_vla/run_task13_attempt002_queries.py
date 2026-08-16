#!/usr/bin/env python
"""Run frozen same-state queries for Task 13 recovery Attempt002.

The two policies are loaded once.  Results are committed task-by-task so a
process interruption cannot silently turn a partial shard into admitted data.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_replay_v2 import case_slug
from task9_common import file_sha256, load_json
from task13_recovery_common import require_complete_attempt002
from trajectory_instrumentation import load_replan_trace

HORIZONS = (1, 3, 5, 10)
ARMS = ("base10", "snap1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--dev-b2-manifest", type=Path, required=True)
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def query(
    policy: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    *,
    nfe: int,
    target: float | None,
) -> torch.Tensor:
    policy.reset()
    policy.config.num_steps = nfe
    kwargs = {} if target is None else {"target_time": target}
    with torch.inference_mode():
        return policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone(), **kwargs).detach()


def shard_name(suite: str, task_id: int) -> str:
    return f"{suite}-task{task_id:02d}"


def validate_existing(path: Path) -> bool:
    status_path = path / "status.json"
    if not path.exists():
        return False
    if not status_path.exists():
        raise RuntimeError(f"Incomplete query shard already exists: {path}")
    status = load_json(status_path)
    if status.get("status") != "TASK13_ATTEMPT002_QUERY_SHARD_COMPLETE":
        raise RuntimeError(f"Invalid existing query shard: {path}")
    if file_sha256(path / "same_state_metrics.jsonl") != status["metrics_sha256"]:
        raise RuntimeError(f"Existing query metrics hash drift: {path}")
    if file_sha256(path / "branch_actions.npz") != status["actions_sha256"]:
        raise RuntimeError(f"Existing query actions hash drift: {path}")
    return True


def main() -> None:
    args = parse_args()
    run = load_json(args.run_manifest)
    require_complete_attempt002(run)
    design = load_json(args.dev_b2_manifest)
    metrics = load_json(args.frozen_metrics)
    if design.get("status") != "TASK13_ATTEMPT002_DEV_B2_FROZEN" or len(design.get("cases", [])) != 200:
        raise RuntimeError("Dev-B2 manifest is not the frozen 200-case Attempt002 design")
    if metrics.get("status") != "TASK13_METRICS_FROZEN":
        raise RuntimeError("Task 13 metrics are not frozen")
    if metrics.get("state_sampling") != (
        "all replan states from both primary arms; aggregate state metrics to case means before primary contrasts"
    ):
        raise RuntimeError("Task 13 state-sampling rule drift")
    if run.get("design_sha256") != file_sha256(args.dev_b2_manifest):
        raise RuntimeError("Attempt002 run/design hash mismatch")

    result_index = {
        (
            str(row["suite"]),
            int(row["task_id"]),
            int(row["init_state_id"]),
            int(row["env_seed"]),
            str(row["arm"]),
        ): row
        for row in run["results"]
    }
    by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in design["cases"]:
        by_task[(str(case["suite"]), int(case["task_id"]))].append(case)
    if len(by_task) != 40 or {len(value) for value in by_task.values()} != {5}:
        raise RuntimeError("Dev-B2 task balance drift")

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

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_policy, base_post = load_policy(args.base_checkpoint)
    snap_policy, snap_post = load_policy(args.snap_checkpoint)
    complete_states = 0
    for task_index, ((suite, task_id), cases) in enumerate(sorted(by_task.items()), start=1):
        output = output_root / shard_name(suite, task_id)
        if validate_existing(output):
            complete_states += int(load_json(output / "status.json")["replan_state_count"])
            print(f"query_task={task_index}/40 shard={output.name} status=verified_existing", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        base_actions: list[np.ndarray] = []
        snap_actions: list[np.ndarray] = []
        started_ns = time.time_ns()
        for case in sorted(cases, key=lambda value: int(value["init_state_id"])):
            slug = case_slug(case)
            for origin_arm in ARMS:
                result = result_index[
                    (
                        suite,
                        task_id,
                        int(case["init_state_id"]),
                        int(case["env_seed"]),
                        origin_arm,
                    )
                ]
                trace_manifest_path = Path(result["trace_manifest"]["path"])
                if file_sha256(trace_manifest_path) != result["trace_manifest"]["sha256"]:
                    raise RuntimeError(f"Attempt002 trace manifest hash mismatch: {slug}/{origin_arm}")
                top = load_json(trace_manifest_path)
                if not top.get("replan_records"):
                    raise RuntimeError(f"Attempt002 trace has no replan states: {slug}/{origin_arm}")
                for replan in top["replan_records"]:
                    metadata, payload = load_replan_trace(Path(replan["path"]), device=args.device)
                    batch = payload["normalized_observation"]
                    noise = payload["policy_output"]["noise_tensor"]
                    archived = payload["policy_output"]["raw_policy_output"]
                    base_chunk = query(base_policy, batch, noise, nfe=10, target=None)
                    snap_chunk = query(snap_policy, batch, noise, nfe=1, target=0.0)
                    self_chunk = base_chunk if origin_arm == "base10" else snap_chunk
                    if not torch.equal(self_chunk.cpu(), archived.cpu()):
                        maximum = float((self_chunk.float().cpu() - archived.float().cpu()).abs().max().item())
                        raise RuntimeError(
                            f"Attempt002 self-replay mismatch {slug}/{origin_arm}/"
                            f"{metadata['identity']['replan_index']}: max_abs={maximum}"
                        )
                    base_denorm = base_post(base_chunk[0, :10].detach().clone()).float().cpu().numpy()
                    snap_denorm = snap_post(snap_chunk[0, :10].detach().clone()).float().cpu().numpy()
                    base_raw = base_chunk[0].float().cpu().numpy()
                    snap_raw = snap_chunk[0].float().cpu().numpy()
                    difference = base_denorm - snap_denorm
                    row: dict[str, Any] = {
                        "case_id": slug,
                        "design_case_id": case["case_id"],
                        "task_id": f"{suite}:{task_id}",
                        "origin_arm": origin_arm,
                        "replan_index": int(metadata["identity"]["replan_index"]),
                        "simulator_timestep": int(metadata["identity"]["simulator_timestep"]),
                        "self_replay_exact": True,
                        "state_payload_sha256": metadata["integrity"]["simulator_state_sha256"],
                        "noise_sha256": file_sha256(Path(replan["path"]) / "sha256_manifest.txt"),
                        "denorm_translation_l2_h10": float(np.linalg.norm(difference[:, :3])),
                        "denorm_rotation_l2_h10": float(np.linalg.norm(difference[:, 3:6])),
                        "denorm_gripper_l2_h10": float(np.linalg.norm(difference[:, 6])),
                        "snap_action_norm_h10": float(np.linalg.norm(snap_denorm)),
                        "array_index": len(rows),
                    }
                    for horizon in HORIZONS:
                        row[f"raw_prefix_l2_h{horizon}"] = float(
                            np.linalg.norm(base_raw[:horizon] - snap_raw[:horizon])
                        )
                        row[f"denorm_prefix_l2_h{horizon}"] = float(
                            np.linalg.norm(base_denorm[:horizon] - snap_denorm[:horizon])
                        )
                    rows.append(row)
                    base_actions.append(base_denorm)
                    snap_actions.append(snap_denorm)
        metrics_path = output / "same_state_metrics.jsonl"
        with metrics_path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        actions_path = output / "branch_actions.npz"
        np.savez_compressed(actions_path, base10=np.stack(base_actions), snap1=np.stack(snap_actions))
        status = {
            "schema_version": 1,
            "status": "TASK13_ATTEMPT002_QUERY_SHARD_COMPLETE",
            "shard_id": output.name,
            "task_id": f"{suite}:{task_id}",
            "case_count": len(cases),
            "replan_state_count": len(rows),
            "policy_query_count": len(rows) * 2,
            "self_replay_exact_count": len(rows),
            "run_manifest_sha256": file_sha256(args.run_manifest),
            "dev_b2_manifest_sha256": file_sha256(args.dev_b2_manifest),
            "frozen_metrics_sha256": file_sha256(args.frozen_metrics),
            "metrics_sha256": file_sha256(metrics_path),
            "actions_sha256": file_sha256(actions_path),
            "started_ns": started_ns,
            "finished_ns": time.time_ns(),
            "new_rollout": False,
            "new_training": False,
        }
        (output / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        complete_states += len(rows)
        print(
            f"query_task={task_index}/40 shard={output.name} states={len(rows)} total_states={complete_states}",
            flush=True,
        )
    print(json.dumps({"status": "TASK13_ATTEMPT002_QUERIES_COMPLETE", "states": complete_states}))


if __name__ == "__main__":
    main()
