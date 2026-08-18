#!/usr/bin/env python
"""Run the six-arm Attempt002 engineering dry run for Task14R."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256, load_capsule, structured_hash
from run_libero_paired_four_arm_pilot import (
    add_batch_dimension,
    configure_standard_libero,
    git_repository_value,
    observation_hashes,
)
from run_replay_v2 import case_slug, contact_pairs, task_semantics
from task14r_common import (
    TASK14R_EVIDENCE_LABELS,
    TASK14R_NAME,
    TASK14R_PHASE_A_ARMS,
    clipping_record,
    compare_closed_loop_steps,
    compose_online_action_chunk,
)
from trajectory_instrumentation import capture_libero_state, physical_state_hash

BASE_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
SNAP_SHA256 = "3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-revision", required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(child) for child in value]
    return value


def query_policy(
    policy: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    *,
    nfe: int,
    target_time: float | None,
) -> torch.Tensor:
    policy.config.num_steps = nfe
    kwargs = {} if target_time is None else {"target_time": target_time}
    with torch.inference_mode():
        return policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone(), **kwargs).detach()


def write_steps(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(json_ready(row), sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "step_count": len(rows)}


def main() -> None:
    args = parse_args()
    if args.execution_horizon != 10:
        raise ValueError("Task14R Phase A freezes execution_horizon=10")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before the admitted engineering dry run")
    design = json.loads(args.engineering_manifest.read_text())
    if design.get("status") != "TASK14R_PHASE_A_ENGINEERING_MANIFEST_FROZEN":
        raise RuntimeError("Task14R Phase A manifest is not frozen")
    if design.get("evidence_labels") != list(TASK14R_EVIDENCE_LABELS):
        raise RuntimeError("Task14R engineering evidence labels drift")
    if design.get("registered_arms") != list(TASK14R_PHASE_A_ARMS):
        raise RuntimeError("Task14R six-arm order drift")
    if int(design.get("case_count", 0)) != 15:
        raise RuntimeError("Task14R Phase A requires exactly 15 harmful Attempt002 cases")

    base_sha = file_sha256(args.base_checkpoint / "model.safetensors")
    snap_sha = file_sha256(args.snap_checkpoint / "model.safetensors")
    if base_sha != BASE_SHA256 or snap_sha != SNAP_SHA256:
        raise RuntimeError("Task14R checkpoint hash mismatch")
    configure_standard_libero(args.libero_root, args.output_root / "libero_standard_config")

    import libero.libero as libero_module
    from libero.libero import benchmark

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)

    def load_policy(checkpoint: Path, revision: str) -> dict[str, Any]:
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = args.device
        config.load_vlm_weights = False
        config.vlm_model_name = str(args.offline_vlm_model_dir.resolve())
        config.n_action_steps = args.execution_horizon
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, revision=revision, strict=False)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        return {"policy": policy, "preprocessor": preprocessor, "postprocessor": postprocessor}

    policies = {
        "base": load_policy(args.base_checkpoint, args.base_revision),
        "snap": load_policy(args.snap_checkpoint, args.snap_revision),
    }
    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in sorted({row["suite"] for row in design["cases"]})}
    args.output_root.mkdir(parents=True, exist_ok=False)
    run_started_ns = time.time_ns()
    case_audits = []
    all_results = []

    for case_index, case in enumerate(design["cases"]):
        suite_name = str(case["suite"])
        task_id = int(case["task_id"])
        init_state_id = int(case["init_state_id"])
        env_seed = int(case["env_seed"])
        max_steps = int(case["max_steps"])
        suite = suites[suite_name]
        prompt = suite.get_task(task_id).language
        slug = case_slug(case)
        capsule_dir = Path(case["capsule_path"])
        if file_sha256(capsule_dir / "sha256_manifest.txt") != str(case["capsule_sha256"]):
            raise RuntimeError(f"Attempt002 capsule hash mismatch: {slug}")
        capsule_metadata, capsule = load_capsule(capsule_dir, device=args.device)
        noise_schedule = np.asarray(capsule["noise_schedule"])
        if array_sha256(noise_schedule) != str(case["noise_schedule_sha256"]):
            raise RuntimeError(f"Attempt002 noise schedule mismatch: {slug}")

        config = LiberoEnvConfig(
            task=suite_name,
            task_ids=[task_id],
            observation_height=256,
            observation_width=256,
        )
        env_preprocessor, env_postprocessor = config.get_env_processors()
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=max_steps,
            camera_name=config.camera_name,
            obs_type=config.obs_type,
            render_mode=config.render_mode,
            observation_width=config.observation_width,
            observation_height=config.observation_height,
            init_states=config.init_states,
            episode_index=init_state_id,
            n_envs=1,
            num_steps_wait=10,
            camera_name_mapping=config.camera_name_mapping,
            control_freq=config.fps,
            control_mode=config.control_mode,
            is_libero_plus=config.is_libero_plus,
            hard_reset=config.hard_reset,
        )
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        arm_results: dict[str, dict[str, Any]] = {}
        arm_query_records: dict[str, list[dict[str, Any]]] = {}
        try:
            semantics = task_semantics(Path(env._task_bddl_file))
            for arm in TASK14R_PHASE_A_ARMS:
                policies["base"]["policy"].reset()
                policies["snap"]["policy"].reset()
                env.init_state_id = init_state_id
                env.reset(seed=env_seed)
                if env._env is None:
                    raise RuntimeError("LIBERO simulator is unavailable")
                raw = env._env.set_init_state(np.asarray(capsule["simulator"]["state"]).copy())
                restored = np.asarray(env._env.get_sim_state())
                if array_sha256(restored) != array_sha256(np.asarray(capsule["simulator"]["state"])):
                    raise RuntimeError(f"Task14R exact initial restoration failed: {slug}/{arm}")
                observation = copy.deepcopy(capsule.get("raw_initial_observation", env._format_raw_obs(raw)))
                initial_state = capture_libero_state(env, observation, semantics["objects_of_interest"])
                initial_physical_hash = physical_state_hash(initial_state)
                queue: deque[dict[str, Any]] = deque()
                rows: list[dict[str, Any]] = []
                query_records: list[dict[str, Any]] = []
                success = False
                termination_reason = "MAX_STEPS"
                replan_index = -1

                for step in range(max_steps):
                    if not queue:
                        replan_index += 1
                        if replan_index >= len(noise_schedule):
                            raise RuntimeError(f"Frozen noise schedule exhausted: {slug}/{arm}")
                        raw_batch = preprocess_observation(add_batch_dimension(copy.deepcopy(observation)))
                        raw_batch["task"] = [prompt]
                        base_batch = policies["base"]["preprocessor"](
                            env_preprocessor(copy.deepcopy(raw_batch))
                        )
                        snap_batch = policies["snap"]["preprocessor"](
                            env_preprocessor(copy.deepcopy(raw_batch))
                        )
                        base_input_hash = structured_hash(base_batch)
                        snap_input_hash = structured_hash(snap_batch)
                        if base_input_hash != snap_input_hash:
                            raise RuntimeError(
                                f"Base/Snap online processor mismatch: {slug}/{arm}/{replan_index}"
                            )
                        noise = torch.from_numpy(noise_schedule[replan_index].copy()).to(args.device)
                        base_chunk = query_policy(
                            policies["base"]["policy"],
                            base_batch,
                            noise,
                            nfe=10,
                            target_time=None,
                        )
                        snap_chunk = query_policy(
                            policies["snap"]["policy"],
                            snap_batch,
                            noise,
                            nfe=1,
                            target_time=0.0,
                        )
                        base_denorm = (
                            policies["base"]["postprocessor"](
                                base_chunk[0, : args.execution_horizon].detach().clone()
                            )
                            .detach()
                            .cpu()
                        )
                        snap_denorm = (
                            policies["snap"]["postprocessor"](
                                snap_chunk[0, : args.execution_horizon].detach().clone()
                            )
                            .detach()
                            .cpu()
                        )
                        composed = compose_online_action_chunk(base_denorm, snap_denorm, arm)
                        query_record = {
                            "replan_index": replan_index,
                            "simulator_timestep": step,
                            "observation_sha256": structured_hash(observation),
                            "base_processed_observation_sha256": base_input_hash,
                            "snap_processed_observation_sha256": snap_input_hash,
                            "same_online_observation": base_input_hash == snap_input_hash,
                            "noise_sha256": array_sha256(noise.detach().cpu().numpy()),
                            "base_raw_chunk_sha256": array_sha256(base_chunk.detach().cpu().float().numpy()),
                            "snap_raw_chunk_sha256": array_sha256(snap_chunk.detach().cpu().float().numpy()),
                            "base_denormalized_chunk_sha256": array_sha256(base_denorm.numpy()),
                            "snap_denormalized_chunk_sha256": array_sha256(snap_denorm.numpy()),
                            "composed_chunk_sha256": array_sha256(composed.numpy()),
                            "online_policy_query_count": 2,
                        }
                        query_records.append(query_record)
                        for chunk_index in range(args.execution_horizon):
                            queue.append(
                                {
                                    "chunk_index": chunk_index,
                                    "base": base_denorm[chunk_index].numpy().copy(),
                                    "snap": snap_denorm[chunk_index].numpy().copy(),
                                    "composed": composed[chunk_index].numpy().copy(),
                                    "query": query_record,
                                }
                            )

                    queued = queue.popleft()
                    action_before = np.asarray(queued["composed"], dtype=np.float32)
                    pre_state = capture_libero_state(env, observation, semantics["objects_of_interest"])
                    observation_digest = structured_hash(observation)
                    env_action = env_postprocessor({ACTION: torch.from_numpy(action_before).unsqueeze(0)})[
                        ACTION
                    ]
                    environment_command = env_action.detach().cpu().float().numpy()[0]
                    clip = clipping_record(environment_command)
                    observation_after, _, terminated, truncated, info = env.step(environment_command)
                    success = bool(info["is_success"])
                    if success:
                        termination_reason = "SUCCESS"
                    elif terminated:
                        termination_reason = "ENV_TERMINATED"
                    elif truncated:
                        termination_reason = "TRUNCATED"
                    else:
                        termination_reason = "MAX_STEPS"
                    post_state = capture_libero_state(
                        env, observation_after, semantics["objects_of_interest"]
                    )
                    rows.append(
                        {
                            "schema_version": "task14r.phase_a.step.v1",
                            "task_name": TASK14R_NAME,
                            "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
                            "case_id": slug,
                            "arm": arm,
                            "step": step,
                            "replan_index": replan_index,
                            "chunk_index": int(queued["chunk_index"]),
                            "base_action": np.asarray(queued["base"]).tolist(),
                            "snap_action": np.asarray(queued["snap"]).tolist(),
                            "composed_action_before_clipping": action_before.tolist(),
                            "composed_action_after_clipping": clip["after"],
                            "environment_command_before_controller_clipping": (environment_command.tolist()),
                            "controller_input_after_clipping": clip["after"],
                            "actually_executed_environment_command": environment_command.tolist(),
                            "clipping_changed": clip["changed"],
                            "clipping_changed_count": clip["changed_count"],
                            "observation_sha256": observation_digest,
                            "observation_hashes": observation_hashes(observation),
                            "base_processed_observation_sha256": queued["query"][
                                "base_processed_observation_sha256"
                            ],
                            "snap_processed_observation_sha256": queued["query"][
                                "snap_processed_observation_sha256"
                            ],
                            "noise_sha256": queued["query"]["noise_sha256"],
                            "physical_state_before_sha256": physical_state_hash(pre_state),
                            "physical_state_after_sha256": physical_state_hash(post_state),
                            "simulator_state_blob_before_sha256": pre_state["state_blob_sha256"],
                            "simulator_state_blob_after_sha256": post_state["state_blob_sha256"],
                            "contact_pairs_after": contact_pairs(env._env),
                            "success": success,
                            "termination_reason": (termination_reason if terminated or truncated else None),
                        }
                    )
                    observation = observation_after
                    if terminated or truncated:
                        break
                if rows and rows[-1]["termination_reason"] is None:
                    rows[-1]["termination_reason"] = termination_reason
                trace = write_steps(args.output_root / "traces" / slug / f"{arm}.steps.jsonl.gz", rows)
                result = {
                    "status": "COMPLETED",
                    "task_name": TASK14R_NAME,
                    "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
                    "case_id": slug,
                    "design_case_id": case["case_id"],
                    "suite": suite_name,
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "env_seed": env_seed,
                    "arm": arm,
                    "success": success,
                    "steps_run": len(rows),
                    "termination_reason": termination_reason,
                    "initial_physical_state_sha256": initial_physical_hash,
                    "action_stream_sha256": array_sha256(
                        np.asarray([row["composed_action_before_clipping"] for row in rows])
                    ),
                    "postclip_action_stream_sha256": array_sha256(
                        np.asarray([row["composed_action_after_clipping"] for row in rows])
                    ),
                    "physical_state_stream_sha256": structured_hash(
                        [row["physical_state_after_sha256"] for row in rows]
                    ),
                    "observation_stream_sha256": structured_hash([row["observation_sha256"] for row in rows]),
                    "query_count": len(query_records) * 2,
                    "all_queries_use_same_online_observation": all(
                        bool(row["same_online_observation"]) for row in query_records
                    ),
                    "noise_sequence_sha256": structured_hash([row["noise_sha256"] for row in query_records]),
                    "trace": trace,
                }
                arm_rows[arm] = rows
                arm_results[arm] = result
                arm_query_records[arm] = query_records
                all_results.append(result)
                print(
                    f"case={case_index + 1}/15 arm={arm} success={int(success)} steps={len(rows)}",
                    flush=True,
                )
        finally:
            env.close()

        noop = compare_closed_loop_steps(arm_rows["snap_repeat"], arm_rows["noop"])
        full = compare_closed_loop_steps(arm_rows["base_repeat"], arm_rows["full_swap"])
        initial_hashes = {arm: row["initial_physical_state_sha256"] for arm, row in arm_results.items()}
        historical = {
            "base_repeat_matches_attempt002_outcome": arm_results["base_repeat"]["success"]
            == bool(case["attempt002_base_success"]),
            "snap_repeat_matches_attempt002_outcome": arm_results["snap_repeat"]["success"]
            == bool(case["attempt002_snap_success"]),
        }
        diagnostics = {
            "policy_noise_consistent_across_arms": len(
                {row["noise_sequence_sha256"] for row in arm_results.values()}
            )
            == 1,
            "action_queue_alignment_noop": noop["queue_alignment_identity"],
            "action_queue_alignment_full_swap": full["queue_alignment_identity"],
            "base_and_snap_queried_same_live_observation": all(
                row["all_queries_use_same_online_observation"] for row in arm_results.values()
            ),
            "reset_physical_state_complete": len(set(initial_hashes.values())) == 1,
            "controller_clipping_consistent_noop": arm_results["snap_repeat"]["postclip_action_stream_sha256"]
            == arm_results["noop"]["postclip_action_stream_sha256"],
            "controller_clipping_consistent_full_swap": arm_results["base_repeat"][
                "postclip_action_stream_sha256"
            ]
            == arm_results["full_swap"]["postclip_action_stream_sha256"],
            "simulator_nondeterminism_suspected_noop": noop["action_identity"]
            and not noop["physical_state_identity"],
            "simulator_nondeterminism_suspected_full_swap": full["action_identity"]
            and not full["physical_state_identity"],
        }
        case_status = (
            "TASK14R_PHASE_A_CASE_IDENTITY_PASSED"
            if noop["closed_loop_identity"] and full["closed_loop_identity"] and all(historical.values())
            else "TASK14R_PHASE_A_CASE_IDENTITY_FAILED"
        )
        case_audit = {
            "status": case_status,
            "case_id": slug,
            "design_case_id": case["case_id"],
            "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
            "noop_vs_snap_repeat": noop,
            "full_swap_vs_base_repeat": full,
            "historical_repeatability": historical,
            "diagnostics": diagnostics,
            "initial_physical_state_hashes": initial_hashes,
            "arm_results": arm_results,
        }
        case_path = args.output_root / "case_audits" / f"{slug}.json"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text(json.dumps(case_audit, indent=2, sort_keys=True) + "\n")
        case_audits.append(case_audit)

        partial = {
            "status": "TASK14R_PHASE_A_RUNNING_PARTIAL",
            "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
            "completed_case_count": len(case_audits),
            "completed_rollout_count": len(all_results),
            "case_audits": case_audits,
        }
        (args.output_root / "run.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n"
        )

    passed = sum(row["status"] == "TASK14R_PHASE_A_CASE_IDENTITY_PASSED" for row in case_audits)
    final = {
        "schema_version": "task14r.phase_a.run.v1",
        "task_name": TASK14R_NAME,
        "phase": "PHASE_A_ATTEMPT002_ENGINEERING_DRY_RUN",
        "status": (
            "TASK14R_PHASE_A_ENGINEERING_GATES_PASSED"
            if passed == 15
            else "TASK14R_PHASE_A_ENGINEERING_GATES_FAILED"
        ),
        "evidence_labels": list(TASK14R_EVIDENCE_LABELS),
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "libero_commit": git_repository_value(args.libero_root, "rev-parse", "HEAD"),
        "engineering_manifest": str(args.engineering_manifest.resolve()),
        "engineering_manifest_sha256": file_sha256(args.engineering_manifest),
        "base_checkpoint_sha256": base_sha,
        "snap_checkpoint_sha256": snap_sha,
        "run_started_ns": run_started_ns,
        "run_finished_ns": time.time_ns(),
        "case_count": len(case_audits),
        "rollout_count": len(all_results),
        "passed_identity_case_count": passed,
        "failed_identity_case_count": len(case_audits) - passed,
        "arm_counts": dict(
            sorted(
                defaultdict(
                    int, {arm: sum(row["arm"] == arm for row in all_results) for arm in TASK14R_PHASE_A_ARMS}
                ).items()
            )
        ),
        "case_audits": case_audits,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": final["status"], "passed_cases": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
