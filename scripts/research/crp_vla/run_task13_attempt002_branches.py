#!/usr/bin/env python
"""Execute frozen isolated branches from Attempt002 Snap-visited states."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_libero_paired_four_arm_pilot import configure_standard_libero
from run_replay_v2 import case_slug, task_semantics
from task9_common import file_sha256, load_json
from task13_recovery_common import require_complete_attempt002
from trajectory_instrumentation import (
    capture_libero_state,
    execute_counterfactual_branches,
    load_replan_trace,
    reset_and_restore_libero_state,
)

HORIZONS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--dev-b2-manifest", type=Path, required=True)
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def shard_name(suite: str, task_id: int) -> str:
    return f"{suite}-task{task_id:02d}"


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
            env_seed=int(self.case["env_seed"]),
            init_state_id=int(self.case["init_state_id"]),
        )
        return state

    def capture(self) -> dict[str, Any]:
        return capture_libero_state(self.env, self.observation, self.object_names)

    def step(self, action: np.ndarray) -> None:
        tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
        env_action = self.env_postprocessor({self.action_key: tensor})[self.action_key]
        self.observation, _, _, _, _ = self.env.step(env_action.numpy()[0])


def validate_query_shard(path: Path) -> dict[str, Any]:
    status = load_json(path / "status.json")
    if status.get("status") != "TASK13_ATTEMPT002_QUERY_SHARD_COMPLETE":
        raise RuntimeError(f"Incomplete Attempt002 query shard: {path}")
    if file_sha256(path / "same_state_metrics.jsonl") != status["metrics_sha256"]:
        raise RuntimeError(f"Attempt002 query metric hash mismatch: {path}")
    if file_sha256(path / "branch_actions.npz") != status["actions_sha256"]:
        raise RuntimeError(f"Attempt002 query action hash mismatch: {path}")
    return status


def validate_existing(path: Path) -> bool:
    status_path = path / "status.json"
    if not path.exists():
        return False
    if not status_path.exists():
        raise RuntimeError(f"Incomplete branch shard already exists: {path}")
    status = load_json(status_path)
    metrics_path = path / "branch_transition_metrics.jsonl"
    if status.get("status") != "TASK13_ATTEMPT002_BRANCH_SHARD_COMPLETE":
        raise RuntimeError(f"Invalid existing branch shard: {path}")
    if file_sha256(metrics_path) != status["metrics_sha256"]:
        raise RuntimeError(f"Existing branch metrics hash drift: {path}")
    return True


def main() -> None:
    args = parse_args()
    run = load_json(args.run_manifest)
    require_complete_attempt002(run)
    design = load_json(args.dev_b2_manifest)
    metrics = load_json(args.frozen_metrics)
    if design.get("status") != "TASK13_ATTEMPT002_DEV_B2_FROZEN" or len(design.get("cases", [])) != 200:
        raise RuntimeError("Dev-B2 manifest is not the frozen 200-case Attempt002 design")
    if metrics.get("status") != "TASK13_METRICS_FROZEN" or metrics.get("horizons") != list(HORIZONS):
        raise RuntimeError("Task 13 frozen transition horizons drift")
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

    configure_standard_libero(args.libero_root, args.output_root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.utils.constants import ACTION

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name, _ in by_task}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total_states = 0
    for task_index, ((suite_name, task_id), cases) in enumerate(sorted(by_task.items()), start=1):
        shard = shard_name(suite_name, task_id)
        output = output_root / shard
        if validate_existing(output):
            total_states += int(load_json(output / "status.json")["student_replan_state_count"])
            print(f"branch_task={task_index}/40 shard={shard} status=verified_existing", flush=True)
            continue
        query_root = args.query_root.resolve() / shard
        query_status = validate_query_shard(query_root)
        with (query_root / "same_state_metrics.jsonl").open() as stream:
            query_rows = [json.loads(line) for line in stream]
        with np.load(query_root / "branch_actions.npz") as archive:
            action_arrays = {key: archive[key].copy() for key in archive.files}
        output.mkdir(parents=True, exist_ok=False)
        suite = suites[suite_name]
        config = LiberoEnvConfig(
            task=suite_name,
            task_ids=[task_id],
            observation_height=256,
            observation_width=256,
        )
        _, env_postprocessor = config.get_env_processors()
        rows: list[dict[str, Any]] = []
        started_ns = time.time_ns()
        for case in sorted(cases, key=lambda value: int(value["init_state_id"])):
            slug = case_slug(case)
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=suite_name,
                episode_length=int(case["max_steps"]),
                camera_name=config.camera_name,
                obs_type=config.obs_type,
                render_mode=config.render_mode,
                observation_width=config.observation_width,
                observation_height=config.observation_height,
                init_states=config.init_states,
                episode_index=int(case["init_state_id"]),
                n_envs=1,
                num_steps_wait=10,
                camera_name_mapping=config.camera_name_mapping,
                control_freq=config.fps,
                control_mode=config.control_mode,
                is_libero_plus=config.is_libero_plus,
                hard_reset=config.hard_reset,
            )
            try:
                semantics = task_semantics(Path(env._task_bddl_file))
                for origin_arm in ("base10", "snap1"):
                    result = result_index[
                        (
                            suite_name,
                            task_id,
                            int(case["init_state_id"]),
                            int(case["env_seed"]),
                            origin_arm,
                        )
                    ]
                    trace_manifest_path = Path(result["trace_manifest"]["path"])
                    if file_sha256(trace_manifest_path) != result["trace_manifest"]["sha256"]:
                        raise RuntimeError(f"Attempt002 trace hash mismatch: {slug}/{origin_arm}")
                    top = load_json(trace_manifest_path)
                    origin_queries = {
                        int(row["replan_index"]): row
                        for row in query_rows
                        if row["case_id"] == slug and row["origin_arm"] == origin_arm
                    }
                    if len(origin_queries) != len(top["replan_records"]):
                        raise RuntimeError(f"Attempt002 query coverage mismatch: {slug}/{origin_arm}")
                    for replan in top["replan_records"]:
                        metadata, payload = load_replan_trace(Path(replan["path"]), device="cpu")
                        replan_index = int(metadata["identity"]["replan_index"])
                        query_row = origin_queries[replan_index]
                        array_index = int(query_row["array_index"])
                        frozen_state = payload["simulator_state"]
                        adapter = BranchAdapter(
                            env,
                            case,
                            semantics["objects_of_interest"],
                            env_postprocessor,
                            ACTION,
                        )
                        branch_result = execute_counterfactual_branches(
                            frozen_state=frozen_state,
                            action_chunks={
                                "base10": action_arrays["base10"][array_index],
                                "snap1": action_arrays["snap1"][array_index],
                            },
                            restore_state=adapter.restore,
                            capture_state=adapter.capture,
                            step_action=adapter.step,
                            horizons=HORIZONS,
                        )
                        if not branch_result["identical_initial_state"]:
                            raise RuntimeError("Attempt002 counterfactual restoration identity failed")
                        rows.append(
                            {
                                "case_id": slug,
                                "design_case_id": case["case_id"],
                                "task_id": f"{suite_name}:{task_id}",
                                "origin_arm": origin_arm,
                                "replan_index": replan_index,
                                "simulator_timestep": int(metadata["identity"]["simulator_timestep"]),
                                "initial_state_sha256": branch_result["initial_state_sha256"],
                                "restoration_hashes": branch_result["restoration_hashes"],
                                "identical_initial_state": True,
                                "horizon_zero_exact": len(
                                    set(branch_result["restoration_hashes"].values())
                                )
                                == 1,
                                "horizons": branch_result["horizons"],
                                "transition_divergence": branch_result["transition_divergence"],
                                "combined_weighted_scalar_used": False,
                            }
                        )
            finally:
                env.close()
        rows_path = output / "branch_transition_metrics.jsonl"
        with rows_path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        status = {
            "schema_version": 1,
            "status": "TASK13_ATTEMPT002_BRANCH_SHARD_COMPLETE",
            "shard_id": shard,
            "task_id": f"{suite_name}:{task_id}",
            "case_count": len(cases),
            "replan_state_count": len(rows),
            "student_replan_state_count": sum(row["origin_arm"] == "snap1" for row in rows),
            "base_replan_state_count": sum(row["origin_arm"] == "base10" for row in rows),
            "isolated_branch_count": len(rows) * 2,
            "identical_restoration_count": len(rows) * 2,
            "horizon_zero_exact_count": len(rows),
            "query_metrics_sha256": query_status["metrics_sha256"],
            "query_actions_sha256": query_status["actions_sha256"],
            "run_manifest_sha256": file_sha256(args.run_manifest),
            "dev_b2_manifest_sha256": file_sha256(args.dev_b2_manifest),
            "frozen_metrics_sha256": file_sha256(args.frozen_metrics),
            "metrics_sha256": file_sha256(rows_path),
            "started_ns": started_ns,
            "finished_ns": time.time_ns(),
            "new_training": False,
        }
        (output / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        total_states += len(rows)
        print(
            f"branch_task={task_index}/40 shard={shard} states={len(rows)} total_states={total_states}",
            flush=True,
        )
    print(json.dumps({"status": "TASK13_ATTEMPT002_BRANCHES_COMPLETE", "states": total_states}))


if __name__ == "__main__":
    main()
