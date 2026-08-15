#!/usr/bin/env python
"""Replay one maturation checkpoint on the frozen replay-v2 development capsules."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256, classify_failure, file_sha256, load_capsule, structured_hash
from run_libero_paired_four_arm_pilot import (
    configure_standard_libero,
    git_repository_value,
    observation_hashes,
)
from run_replay_v2 import (
    body_poses,
    case_slug,
    collision_evidence,
    contact_pairs,
    json_ready,
    make_event_state,
    update_events,
    write_trace,
)
from trajectory_instrumentation import (
    ReplanTraceRecorder,
    capture_libero_state,
    objective_event_snapshot,
)

SNAP_ARMS = {
    "snap10": (10, None),
    "snap2": (2, None),
    "snap1": (1, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(checkpoint.glob("policy_*processor*"))
        if path.is_file()
    }


def checkpoint_trace_root(output: Path, checkpoint_step: int) -> Path:
    return output.parent / "traces" / f"step_{checkpoint_step:06d}"


def validate_checkpoint_processors(metadata: dict[str, Any], checkpoint_processors: dict[str, str]) -> None:
    expected = metadata.get("policy_processor_versions", {}).get("snap")
    if not isinstance(expected, dict) or checkpoint_processors != expected:
        raise ValueError("Checkpoint processor manifest does not match frozen Snap processor manifest")


def validate_capsule_metadata(metadata: dict[str, Any], case: dict[str, Any]) -> None:
    expected = {
        "phase": "development",
        "suite": case["suite"],
        "task_id": int(case["task_id"]),
        "init_state_id": int(case["init_state_id"]),
        "env_seed": int(case["env_seed"]),
        "maximum_episode_length": int(case["max_steps"]),
    }
    mismatches = {
        key: {"capsule": metadata.get(key), "design": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Capsule/design mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise ValueError("Repository must be clean before admitted maturation replay")
    design = json.loads(args.design.read_text())
    cases = design.get("cases", [])
    if design.get("phase") != "development" or len(cases) != 40:
        raise ValueError("Expected the frozen 40-case replay-v2 development design")
    configure_standard_libero(args.libero_root, args.output.parent / "libero_standard_config")
    repository_commit = git_value("rev-parse", "HEAD")
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")

    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    if Path(libero_module.get_assets_path()).resolve() != expected_assets:
        raise RuntimeError("LIBERO assets did not resolve to the pinned checkout")

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    config.load_vlm_weights = False
    config.n_action_steps = 10
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=config, strict=False)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    checkpoint_processors = processor_manifest(args.checkpoint)
    checkpoint_sha = file_sha256(args.checkpoint / "model.safetensors")
    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in sorted({case["suite"] for case in cases})}
    results = []
    for case_index, case in enumerate(cases):
        slug = case_slug(case)
        capsule_dir = args.capsule_root / slug
        metadata, payload = load_capsule(capsule_dir, args.device)
        validate_capsule_metadata(metadata, case)
        validate_checkpoint_processors(metadata, checkpoint_processors)
        suite_name = case["suite"]
        task_id = int(case["task_id"])
        init_state_id = int(case["init_state_id"])
        env_seed = int(case["env_seed"])
        max_steps = int(case["max_steps"])
        suite = suites[suite_name]
        prompt = suite.get_task(task_id).language
        if prompt != metadata["instruction"]:
            raise ValueError(f"Instruction mismatch for {slug}")
        env_config = LiberoEnvConfig(
            task=suite_name,
            task_ids=[task_id],
            observation_height=256,
            observation_width=256,
        )
        env_preprocessor, env_postprocessor = env_config.get_env_processors()
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=max_steps,
            camera_name=env_config.camera_name,
            obs_type=env_config.obs_type,
            render_mode=env_config.render_mode,
            observation_width=env_config.observation_width,
            observation_height=env_config.observation_height,
            init_states=env_config.init_states,
            episode_index=init_state_id,
            n_envs=1,
            num_steps_wait=10,
            camera_name_mapping=env_config.camera_name_mapping,
            control_freq=env_config.fps,
            control_mode=env_config.control_mode,
            is_libero_plus=env_config.is_libero_plus,
            hard_reset=env_config.hard_reset,
        )
        try:
            env.init_state_id = init_state_id
            env.reset(seed=env_seed)
            if env._env is None:
                raise RuntimeError("LIBERO inner environment was not initialized")
            evaluator_path = Path(type(env).step.__code__.co_filename)
            if file_sha256(evaluator_path) != metadata["success_evaluator_sha256"]:
                raise RuntimeError(f"Success evaluator mismatch for {slug}")
            initial_state = payload["simulator"]["state"]
            expected_state_sha = array_sha256(initial_state)
            semantics = metadata["task_semantics"]
            canonical_observation = payload["raw_initial_observation"]
            canonical_input_sha = structured_hash(payload["canonical_policy_batch"])
            if canonical_input_sha != metadata["canonical_input_sha256"]:
                raise RuntimeError(f"Canonical input hash mismatch for {slug}")
            noise_schedule = payload["noise_schedule"]
            if array_sha256(noise_schedule) != metadata["noise_schedule_sha256"]:
                raise RuntimeError(f"Noise schedule hash mismatch for {slug}")
            env._env.set_init_state(initial_state)
            initial_poses = body_poses(env._env, semantics["objects_of_interest"])

            for arm, (nfe, target_time) in SNAP_ARMS.items():
                policy.reset()
                policy.config.num_steps = nfe
                env.init_state_id = init_state_id
                env.reset(seed=env_seed)
                env._env.set_init_state(initial_state)
                if array_sha256(np.asarray(env._env.get_sim_state())) != expected_state_sha:
                    raise RuntimeError(f"Exact simulator-state restore failed for {slug}/{arm}")
                observation = copy.deepcopy(canonical_observation)
                action_queue: deque[torch.Tensor] = deque()
                predicted_chunks = []
                executed_actions = []
                trace_rows = []
                events = make_event_state(initial_poses, semantics)
                replan_index = -1
                success = False
                termination_reason = "MAX_STEPS"
                arm_start = time.perf_counter_ns()
                replan_recorder = ReplanTraceRecorder(
                    checkpoint_trace_root(args.output, args.checkpoint_step) / slug / f"{arm}.replans",
                    identity_base={
                        "case_id": slug,
                        "task_id": f"{suite_name}:{task_id}",
                        "arm_id": arm,
                        "rollout_id": f"maturation-{args.checkpoint_step}:{slug}:{arm}",
                    },
                    language_condition={
                        "instruction": prompt,
                        "reference": metadata["bddl_path"],
                    },
                    processor_contract=checkpoint_processors,
                    processor_hash=structured_hash(checkpoint_processors),
                    execution_horizon=int(metadata["execution_horizon"]),
                    nfe=nfe,
                    noise_seed=int(metadata["noise_seed"]),
                )
                for step in range(max_steps):
                    latency_ms = None
                    if not action_queue:
                        if replan_recorder.pending is not None:
                            boundary_state = capture_libero_state(
                                env, observation, semantics["objects_of_interest"]
                            )
                            replan_recorder.finish(
                                objective_event_snapshot(
                                    boundary_state,
                                    contacts=contact_pairs(env._env),
                                    success=success,
                                    termination_reason=None,
                                )
                            )
                        replan_index += 1
                        if replan_index == 0:
                            batch = copy.deepcopy(payload["canonical_policy_batch"])
                        else:
                            from run_libero_paired_four_arm_pilot import add_batch_dimension

                            from lerobot.envs.utils import preprocess_observation

                            batch = preprocess_observation(add_batch_dimension(observation))
                            batch["task"] = [prompt]
                            batch = preprocessor(env_preprocessor(batch))
                        input_sha = structured_hash(batch)
                        noise = torch.from_numpy(noise_schedule[replan_index].copy()).to(args.device)
                        kwargs = {} if target_time is None else {"target_time": target_time}
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        started = time.perf_counter_ns()
                        with torch.inference_mode():
                            chunk = policy.predict_action_chunk(batch, noise=noise, **kwargs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                        predicted_chunks.append(chunk[0].detach().cpu().float().numpy())
                        denormalized_chunk = postprocessor(
                            chunk[0, : metadata["execution_horizon"]].detach().clone()
                        )
                        replan_state = capture_libero_state(
                            env, observation, semantics["objects_of_interest"]
                        )
                        replan_recorder.start(
                            replan_index=replan_index,
                            simulator_timestep=step,
                            raw_observation=observation,
                            normalized_observation=batch,
                            simulator_state=replan_state,
                            raw_policy_output=chunk.detach().cpu(),
                            denormalized_action=denormalized_chunk.detach().cpu(),
                            noise_tensor=noise.detach().cpu(),
                            action_mask=batch.get("action_is_pad"),
                            event_state=objective_event_snapshot(
                                replan_state,
                                contacts=contact_pairs(env._env),
                                success=success,
                                termination_reason=None,
                            ),
                        )
                        action_queue.extend(chunk[0, : metadata["execution_horizon"]])
                    action = postprocessor(action_queue.popleft().unsqueeze(0))
                    action = env_postprocessor({ACTION: action})[ACTION]
                    action_numpy = action.detach().cpu().float().numpy()[0]
                    executed_actions.append(action_numpy)
                    replan_recorder.append_executed_action(action_numpy)
                    observation_before = observation_hashes(observation)
                    observation, _, terminated, truncated, info = env.step(action_numpy)
                    success = bool(info["is_success"])
                    pairs = contact_pairs(env._env)
                    poses = body_poses(env._env, semantics["objects_of_interest"])
                    update_events(events, step, action_numpy, pairs, poses, semantics, success)
                    robot_state = observation.get("robot_state", {})
                    trace_rows.append(
                        {
                            "timestamp_ns": time.time_ns(),
                            "elapsed_ms": (time.perf_counter_ns() - arm_start) / 1_000_000,
                            "step": step,
                            "replan_index": replan_index,
                            "observation_hashes": observation_before,
                            "processed_policy_input_sha256": input_sha,
                            "predicted_full_action_chunk_index": replan_index,
                            "actually_executed_action": action_numpy.tolist(),
                            "end_effector_pose": json_ready(robot_state.get("eef")),
                            "gripper_command": float(action_numpy[-1]),
                            "gripper_state": json_ready(robot_state.get("gripper")),
                            "object_poses": poses,
                            "contact_pairs": pairs,
                            "collision": collision_evidence(pairs),
                            "success_predicates": {"libero_check_success": success},
                            "policy_nfe": nfe,
                            "policy_latency_ms": latency_ms,
                        }
                    )
                    if terminated or truncated:
                        termination_reason = (
                            "SUCCESS" if success else "ENV_TERMINATED" if terminated else "TRUNCATED"
                        )
                        break
                final_state = capture_libero_state(env, observation, semantics["objects_of_interest"])
                replan_recorder.finish(
                    objective_event_snapshot(
                        final_state,
                        contacts=contact_pairs(env._env),
                        success=success,
                        termination_reason=termination_reason,
                    )
                )
                event_record = {key: value for key, value in events.items() if key != "initial_object_poses"}
                trace_manifest = write_trace(
                    checkpoint_trace_root(args.output, args.checkpoint_step) / slug,
                    arm,
                    trace_rows,
                    predicted_chunks,
                    executed_actions,
                    replan_recorder.records,
                )
                result = {
                    "status": "COMPLETED",
                    "checkpoint_step": args.checkpoint_step,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": prompt,
                    "init_state_id": init_state_id,
                    "env_seed": env_seed,
                    "arm": arm,
                    "nfe": nfe,
                    "success": success,
                    "steps_run": len(executed_actions),
                    "success_timestep": len(executed_actions) if success else None,
                    "termination_reason": termination_reason,
                    "capsule_sha256": file_sha256(capsule_dir / "sha256_manifest.txt"),
                    "initial_sim_state_sha256": expected_state_sha,
                    "canonical_input_sha256": canonical_input_sha,
                    "noise_schedule_sha256": metadata["noise_schedule_sha256"],
                    "evaluator_sha256": metadata["success_evaluator_sha256"],
                    "latency_ms_per_replan": [
                        row["policy_latency_ms"] for row in trace_rows if row["policy_latency_ms"] is not None
                    ],
                    "events": event_record,
                    "failure_classification": classify_failure(event_record, success),
                    "trace_manifest": trace_manifest,
                }
                results.append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                partial = {
                    "status": "RUNNING_PARTIAL",
                    "checkpoint_step": args.checkpoint_step,
                    "completed_rollout_count": len(results),
                    "planned_rollout_count": len(cases) * len(SNAP_ARMS),
                    "results": results,
                }
                args.output.with_suffix(".partial.json").write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"step={args.checkpoint_step} case={case_index + 1}/40 arm={arm} "
                    f"success={int(success)} steps={len(executed_actions)}",
                    flush=True,
                )
        finally:
            env.close()
    output = {
        "schema_version": 1,
        "status": "MATURATION_REPLAY_COMPLETED",
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "libero_commit": libero_commit,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "processor_manifest": checkpoint_processors,
        "design_sha256": file_sha256(args.design),
        "case_count": len(cases),
        "rollout_count": len(results),
        "arm_specs": SNAP_ARMS,
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "rollouts": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
