#!/usr/bin/env python
"""Execute isolated Base/Snap branches from Snap-1 student-visited states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_libero_paired_four_arm_pilot import configure_standard_libero
from run_replay_v2 import task_semantics
from task9_common import file_sha256, load_json
from trajectory_instrumentation import (
    capture_libero_state,
    execute_counterfactual_branches,
    load_replan_trace,
    reset_and_restore_libero_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--task11-trace-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


class BranchAdapter:
    def __init__(
        self,
        env: Any,
        case: dict[str, Any],
        object_names: list[str],
        env_postprocessor: Any,
        action_key: str,
    ) -> None:
        self.env = env
        self.case = case
        self.object_names = object_names
        self.env_postprocessor = env_postprocessor
        self.action_key = action_key
        self.observation: Any = None

    def restore(self, state: dict[str, Any]) -> dict[str, Any]:
        self.observation = reset_and_restore_libero_state(
            self.env,
            state,
            env_seed=int(self.case["noise_seed"]),
            init_state_id=int(self.case["initial_state_id"]),
        )
        return state

    def capture(self) -> dict[str, Any]:
        return capture_libero_state(self.env, self.observation, self.object_names)

    def step(self, action: np.ndarray) -> None:
        tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
        env_action = self.env_postprocessor({self.action_key: tensor})[self.action_key]
        self.observation, _, _, _, _ = self.env.step(env_action.numpy()[0])


def main() -> None:
    args = parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("status") != "TASK12_PASS_BRANCH_PROTOCOL_FROZEN":
        raise RuntimeError("Task 12 protocol is not frozen")
    execution = load_json(args.execution_manifest)
    shard = next((item for item in execution["shards"] if item["shard_id"] == args.shard_id), None)
    if shard is None:
        raise ValueError(f"Unknown Task 11 shard {args.shard_id}")
    manifest = load_json(args.confirmation1200_manifest)
    cases = {item["case_id"]: item for item in manifest["cases"]}
    diagnostic_ids = set(load_json(args.diagnostic_subset)["case_ids"])
    case_ids = sorted(set(shard["case_ids"]) & diagnostic_ids)
    if len(case_ids) != 5:
        raise RuntimeError("Task 12 branch shard must contain five diagnostic cases")
    query_root = args.output_root.resolve() / "phase_b_queries" / args.shard_id
    query_status = load_json(query_root / "status.json")
    if query_status.get("status") != "TASK12_PHASE_B_QUERY_SHARD_COMPLETE":
        raise RuntimeError("Task 12 same-state query shard is incomplete")
    if file_sha256(query_root / "same_state_metrics.jsonl") != query_status["metrics_sha256"]:
        raise RuntimeError("Task 12 same-state metrics changed before branching")
    if file_sha256(query_root / "branch_actions.npz") != query_status["actions_sha256"]:
        raise RuntimeError("Task 12 branch actions changed before branching")
    with (query_root / "same_state_metrics.jsonl").open() as stream:
        query_rows = [json.loads(line) for line in stream]
    with np.load(query_root / "branch_actions.npz") as archive:
        action_arrays = {key: archive[key].copy() for key in archive.files}
    trace_records = load_json(args.task11_trace_manifest)["records"]
    snap_trace = {
        item["case_id"]: item
        for item in trace_records
        if item["arm"] == "snap1_20k" and item["case_id"] in case_ids
    }
    output = args.output_root.resolve() / "phase_b_branches" / args.shard_id
    if output.exists():
        raise RuntimeError(f"Task 12 branch output already exists: {output}")
    output.mkdir(parents=True)
    configure_standard_libero(args.libero_root, output / "libero_standard_config")

    import libero.libero as libero_module
    from libero.libero import benchmark

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.utils.constants import ACTION

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    suite_name, task_text = str(shard["task_id"]).split(":", 1)
    task_id = int(task_text)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    config = LiberoEnvConfig(
        task=suite_name, task_ids=[task_id], observation_height=256, observation_width=256
    )
    _, env_postprocessor = config.get_env_processors()
    rows: list[dict[str, Any]] = []
    started_ns = time.time_ns()
    for case_index, case_id in enumerate(case_ids, start=1):
        case = cases[case_id]
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=int(case["maximum_episode_length"]),
            camera_name=config.camera_name,
            obs_type=config.obs_type,
            render_mode=config.render_mode,
            observation_width=config.observation_width,
            observation_height=config.observation_height,
            init_states=config.init_states,
            episode_index=int(case["initial_state_id"]),
            n_envs=1,
            num_steps_wait=10,
            camera_name_mapping=config.camera_name_mapping,
            control_freq=config.fps,
            control_mode=config.control_mode,
            is_libero_plus=config.is_libero_plus,
            hard_reset=config.hard_reset,
        )
        try:
            bddl = Path(env._task_bddl_file)
            semantics = task_semantics(bddl)
            top_record = snap_trace[case_id]
            if file_sha256(Path(top_record["trace_path"])) != top_record["trace_hash"]:
                raise RuntimeError(f"Task 11 Snap trace changed: {case_id}")
            top = load_json(Path(top_record["trace_path"]))
            snap_queries = {
                int(row["replan_index"]): row
                for row in query_rows
                if row["case_id"] == case_id and row["origin_arm"] == "snap1_20k"
            }
            if len(snap_queries) != len(top["replan_records"]):
                raise RuntimeError(f"Task 12 Snap replan query coverage mismatch: {case_id}")
            for replan in top["replan_records"]:
                metadata, payload = load_replan_trace(Path(replan["path"]), device="cpu")
                replan_index = int(metadata["identity"]["replan_index"])
                query_row = snap_queries[replan_index]
                array_index = int(query_row["array_index"])
                frozen_state = payload["simulator_state"]
                adapter = BranchAdapter(
                    env,
                    case,
                    semantics["objects_of_interest"],
                    env_postprocessor,
                    ACTION,
                )

                result = execute_counterfactual_branches(
                    frozen_state=frozen_state,
                    action_chunks={
                        "base10": action_arrays["base10"][array_index],
                        "snap1_20k": action_arrays["snap1_20k"][array_index],
                    },
                    restore_state=adapter.restore,
                    capture_state=adapter.capture,
                    step_action=adapter.step,
                )
                if not result["identical_initial_state"]:
                    raise RuntimeError("Task 12 counterfactual restoration identity failed")
                rows.append(
                    {
                        "case_id": case_id,
                        "task_id": shard["task_id"],
                        "origin_arm": "snap1_20k",
                        "replan_index": replan_index,
                        "simulator_timestep": int(metadata["identity"]["simulator_timestep"]),
                        "initial_state_sha256": result["initial_state_sha256"],
                        "restoration_hashes": result["restoration_hashes"],
                        "identical_initial_state": True,
                        "horizons": result["horizons"],
                        "transition_divergence": result["transition_divergence"],
                        "combined_weighted_scalar_used": False,
                    }
                )
        finally:
            env.close()
        print(
            f"shard={args.shard_id} diagnostic_case={case_index}/5 isolated_branches=complete",
            flush=True,
        )
    rows_path = output / "branch_transition_metrics.jsonl"
    with rows_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    status = {
        "schema_version": 1,
        "status": "TASK12_PHASE_B_BRANCH_SHARD_COMPLETE",
        "shard_id": args.shard_id,
        "task_id": shard["task_id"],
        "case_count": len(case_ids),
        "student_replan_state_count": len(rows),
        "isolated_branch_count": len(rows) * 2,
        "transition_metric_count": len(rows) * 4,
        "identical_restoration_count": len(rows) * 2,
        "metrics_sha256": file_sha256(rows_path),
        "protocol_sha256": file_sha256(args.protocol),
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "new_training": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
