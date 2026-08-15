#!/usr/bin/env python
"""Execute one frozen Task 11 task shard without interim outcome aggregation."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256, structured_hash
from run_libero_paired_four_arm_pilot import (
    add_batch_dimension,
    configure_standard_libero,
    observation_hashes,
)
from run_replay_v2 import (
    body_poses,
    contact_pairs,
    make_event_state,
    task_semantics,
    update_events,
    write_trace,
)
from task9_common import canonical_json_sha256, effective_processor_contract, file_sha256, load_json
from task11_common import (
    BASE_ARM,
    SNAP_ARM,
    TASK10_MANIFEST_SHA256,
    validate_execution_manifest,
    validate_result_record,
    write_hashed_json,
)
from trajectory_instrumentation import (
    ReplanTraceRecorder,
    capture_libero_state,
    objective_event_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attempt-id", default="attempt-0001")
    parser.add_argument("--retry-authorization", type=Path)
    return parser.parse_args()


def append_infrastructure_log(path: Path, line: str) -> None:
    with path.open("a") as stream:
        stream.write(line.rstrip() + "\n")


def load_retry_authorization(
    path: Path | None, case_id: str, arm: str, attempt_id: str
) -> dict[str, Any] | None:
    if attempt_id == "attempt-0001":
        if path is not None:
            raise ValueError("Retry authorization supplied for first attempt")
        return None
    if path is None:
        raise RuntimeError("A named retry attempt requires an immutable retry authorization")
    record = load_json(path)
    allowed = {
        (str(item["case_id"]), str(item["arm"]), str(item["attempt_id"])) for item in record["allowed"]
    }
    if (case_id, arm, attempt_id) not in allowed:
        raise RuntimeError(f"Retry authorization does not admit {case_id}/{arm}/{attempt_id}")
    if record.get("outcome_accessed_before_authorization") is not False:
        raise RuntimeError("Retry authorization must be method-blind and outcome-unseen")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "record": record}


def existing_completed_result(
    result_arm_root: Path,
    *,
    case: dict[str, Any],
    arm: str,
    base_model_hash: str,
) -> dict[str, Any] | None:
    completed = []
    for path in sorted(result_arm_root.glob("attempt-*/result.json")):
        record = load_json(path)
        validate_result_record(
            record,
            case=case,
            arm=arm,
            base_model_hash=base_model_hash,
            verify_files=True,
        )
        completed.append(record)
    if len(completed) > 1:
        raise RuntimeError(f"Multiple completed attempts exist for {case['case_id']}/{arm}")
    return completed[0] if completed else None


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    execution = load_json(args.execution_manifest)
    manifest = load_json(args.confirmation1200_manifest)
    validate_execution_manifest(execution, manifest)
    if file_sha256(args.confirmation1200_manifest) != TASK10_MANIFEST_SHA256:
        raise RuntimeError("Task 11 frozen manifest changed")
    if (
        execution["repository_commit"]
        != __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ):
        raise RuntimeError("Repository HEAD differs from Task 11 execution freeze")
    if __import__("subprocess").check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Repository must remain clean during formal execution")
    shard_entry = next((item for item in execution["shards"] if item["shard_id"] == args.shard_id), None)
    if shard_entry is None:
        raise ValueError(f"Unknown frozen shard: {args.shard_id}")
    shard_path = Path(shard_entry["shard_manifest_path"])
    if file_sha256(shard_path) != shard_entry["shard_manifest_sha256"]:
        raise RuntimeError("Shard manifest hash mismatch")
    shard = load_json(shard_path)
    cases_by_id = {str(case["case_id"]): case for case in manifest["cases"]}
    cases = [cases_by_id[case_id] for case_id in shard["case_ids"]]
    if len(cases) != 30 or {str(case["task_id"]) for case in cases} != {str(shard["task_id"])}:
        raise RuntimeError("Shard no longer maps to one exact 30-case task")

    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")
    selected_hash = file_sha256(args.selected_checkpoint / "model.safetensors")
    if base_hash != execution["base_model_sha256"] or selected_hash != execution["selected_model_sha256"]:
        raise RuntimeError("Task 11 model hash drift")
    base_contract = effective_processor_contract(args.base_checkpoint)
    snap_contract = effective_processor_contract(args.selected_checkpoint)
    if (
        base_contract["sha256"] != execution["processor_hash"]
        or snap_contract["sha256"] != execution["processor_hash"]
    ):
        raise RuntimeError("Task 11 processor contract drift")

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def load_policy(checkpoint: Path) -> dict[str, Any]:
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = args.device
        config.load_vlm_weights = False
        config.vlm_model_name = str(args.offline_vlm_model_dir.resolve())
        config.n_action_steps = 10
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=False)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        return {"policy": policy, "preprocessor": preprocessor, "postprocessor": postprocessor}

    bundles = {BASE_ARM: load_policy(args.base_checkpoint), SNAP_ARM: load_policy(args.selected_checkpoint)}
    configure_standard_libero(args.libero_root, output_root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    if Path(libero_module.get_assets_path()).resolve() != expected_assets:
        raise RuntimeError("LIBERO assets drift")
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    suite_name, task_text = str(shard["task_id"]).split(":", 1)
    task_id = int(task_text)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    prompt = suite.get_task(task_id).language
    if any(case["task_instruction"] != prompt for case in cases):
        raise RuntimeError("Frozen language condition drift")
    env_config = LiberoEnvConfig(
        task=suite_name, task_ids=[task_id], observation_height=256, observation_width=256
    )
    env_preprocessor, env_postprocessor = env_config.get_env_processors()
    shard_started_ns = time.time_ns()
    completed_or_validated = 0
    skipped_valid = 0
    infra_log = output_root / "CONFIRMATION1200_EXECUTION_LOG.md"
    failure_log = output_root / "CONFIRMATION1200_FAILURE_AND_RETRY_LOG.md"
    append_infrastructure_log(
        infra_log,
        f"- `{args.shard_id}` started at unix ns `{shard_started_ns}` with attempt namespace `{args.attempt_id}`.",
    )

    for case_index, case in enumerate(cases):
        init_state_id = int(case["initial_state_id"])
        env_seed = int(case["noise_seed"])
        max_steps = int(case["maximum_episode_length"])
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
                raise RuntimeError("LIBERO inner environment unavailable")
            source_state = np.asarray(suite.get_task_init_states(task_id)[init_state_id]).copy()
            if array_sha256(source_state) != case["initial_state_hash"]:
                raise RuntimeError("Frozen initial-state identity mismatch")
            initial_sim_state = np.asarray(env._env.get_sim_state()).copy()
            if array_sha256(initial_sim_state) != case["simulator_state_hash"]:
                raise RuntimeError("Frozen simulator-state identity mismatch")
            canonical_raw = env._env.set_init_state(initial_sim_state)
            canonical_observation = env._format_raw_obs(copy.deepcopy(canonical_raw))
            qpos = np.asarray(env._env.sim.data.qpos).copy()
            qvel = np.asarray(env._env.sim.data.qvel).copy()
            qpos_qvel_hash = canonical_json_sha256({"qpos": array_sha256(qpos), "qvel": array_sha256(qvel)})
            if qpos_qvel_hash != case["qpos_qvel_hash"]:
                raise RuntimeError("Frozen qpos/qvel identity mismatch")
            evaluator_path = Path(type(env).step.__code__.co_filename)
            if file_sha256(evaluator_path) != case["evaluator_hash"]:
                raise RuntimeError("Frozen evaluator hash mismatch")
            observed_hash = canonical_json_sha256(observation_hashes(canonical_observation))
            render_variance = observed_hash != case["observation_hash"]
            initial_batch = preprocess_observation(add_batch_dimension(copy.deepcopy(canonical_observation)))
            initial_batch["task"] = [prompt]
            base_batch = bundles[BASE_ARM]["preprocessor"](env_preprocessor(copy.deepcopy(initial_batch)))
            snap_batch = bundles[SNAP_ARM]["preprocessor"](env_preprocessor(copy.deepcopy(initial_batch)))
            if structured_hash(base_batch) != structured_hash(snap_batch):
                raise RuntimeError("Paired processor output mismatch")
            replans = (max_steps + int(case["execution_horizon"]) - 1) // int(case["execution_horizon"])
            generator = torch.Generator(device="cpu").manual_seed(env_seed)
            noise_schedule = torch.randn(
                (replans, 1, 50, 32), generator=generator, dtype=torch.float32
            ).numpy()
            if (
                list(noise_schedule.shape) != case["noise_shape"]
                or array_sha256(noise_schedule) != case["noise_schedule_hash"]
            ):
                raise RuntimeError("Frozen noise schedule mismatch")
            bddl_path = Path(env._task_bddl_file)
            semantics = task_semantics(bddl_path)
            initial_poses = body_poses(env._env, semantics["objects_of_interest"])

            for arm in (BASE_ARM, SNAP_ARM):
                result_arm_root = output_root / "execution" / "results" / case["case_id"] / arm
                completed = existing_completed_result(
                    result_arm_root, case=case, arm=arm, base_model_hash=base_hash
                )
                if completed is not None:
                    skipped_valid += 1
                    completed_or_validated += 1
                    continue
                first_attempt_root = result_arm_root / "attempt-0001"
                selected_attempt_id = args.attempt_id
                if args.attempt_id != "attempt-0001" and not first_attempt_root.exists():
                    # Arms not reached by an earlier stopped shard are still first attempts.
                    selected_attempt_id = "attempt-0001"
                attempt_root = result_arm_root / selected_attempt_id
                retry = load_retry_authorization(
                    args.retry_authorization if selected_attempt_id != "attempt-0001" else None,
                    str(case["case_id"]),
                    arm,
                    selected_attempt_id,
                )
                if attempt_root.exists():
                    raise RuntimeError(
                        f"Attempt namespace already exists without a valid completed result: {attempt_root}"
                    )
                attempt_root.mkdir(parents=True)
                bundle = bundles[arm]
                policy = bundle["policy"]
                nfe = 10 if arm == BASE_ARM else 1
                target_time = None if arm == BASE_ARM else 0.0
                attempt_started_ns = time.time_ns()
                try:
                    policy.reset()
                    policy.config.num_steps = nfe
                    env.init_state_id = init_state_id
                    env.reset(seed=env_seed)
                    env._env.set_init_state(initial_sim_state)
                    if array_sha256(np.asarray(env._env.get_sim_state())) != case["simulator_state_hash"]:
                        raise RuntimeError("Paired arm simulator restoration mismatch")
                    observation = copy.deepcopy(canonical_observation)
                    action_queue: deque[torch.Tensor] = deque()
                    trace_rows: list[dict[str, Any]] = []
                    predicted_chunks: list[np.ndarray] = []
                    executed_actions: list[np.ndarray] = []
                    events = make_event_state(initial_poses, semantics)
                    success = False
                    termination_reason = "MAX_STEPS"
                    replan_index = -1
                    last_input_hash = structured_hash(base_batch)
                    recorder = ReplanTraceRecorder(
                        output_root
                        / "execution"
                        / "traces"
                        / case["case_id"]
                        / arm
                        / selected_attempt_id
                        / "replans",
                        identity_base={
                            "case_id": str(case["case_id"]),
                            "task_id": str(case["task_id"]),
                            "arm_id": arm,
                            "rollout_id": f"task11:{case['case_id']}:{arm}:{selected_attempt_id}",
                        },
                        language_condition={"instruction": prompt, "reference": str(bddl_path.resolve())},
                        processor_contract=base_contract,
                        processor_hash=base_contract["sha256"],
                        execution_horizon=int(case["execution_horizon"]),
                        nfe=nfe,
                        noise_seed=env_seed,
                    )
                    arm_started = time.perf_counter_ns()
                    for step in range(max_steps):
                        latency_ms = None
                        if not action_queue:
                            if recorder.pending is not None:
                                boundary = capture_libero_state(
                                    env, observation, semantics["objects_of_interest"]
                                )
                                recorder.finish(
                                    objective_event_snapshot(
                                        boundary,
                                        contacts=contact_pairs(env._env),
                                        success=success,
                                        termination_reason=None,
                                    )
                                )
                            replan_index += 1
                            if replan_index == 0:
                                batch = copy.deepcopy(base_batch if arm == BASE_ARM else snap_batch)
                            else:
                                batch = preprocess_observation(add_batch_dimension(observation))
                                batch["task"] = [prompt]
                                batch = bundle["preprocessor"](env_preprocessor(batch))
                            last_input_hash = structured_hash(batch)
                            noise = torch.from_numpy(noise_schedule[replan_index].copy()).to(args.device)
                            kwargs = {} if target_time is None else {"target_time": target_time}
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            policy_started = time.perf_counter_ns()
                            with torch.inference_mode():
                                chunk = policy.predict_action_chunk(batch, noise=noise, **kwargs)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            latency_ms = (time.perf_counter_ns() - policy_started) / 1_000_000
                            predicted_chunks.append(chunk[0].detach().cpu().float().numpy())
                            denormalized = bundle["postprocessor"](
                                chunk[0, : int(case["execution_horizon"])].detach().clone()
                            )
                            replan_state = capture_libero_state(
                                env, observation, semantics["objects_of_interest"]
                            )
                            recorder.start(
                                replan_index=replan_index,
                                simulator_timestep=step,
                                raw_observation=observation,
                                normalized_observation=batch,
                                simulator_state=replan_state,
                                raw_policy_output=chunk.detach().cpu(),
                                denormalized_action=denormalized.detach().cpu(),
                                noise_tensor=noise.detach().cpu(),
                                action_mask=batch.get("action_is_pad"),
                                event_state=objective_event_snapshot(
                                    replan_state,
                                    contacts=contact_pairs(env._env),
                                    success=success,
                                    termination_reason=None,
                                ),
                            )
                            action_queue.extend(chunk[0, : int(case["execution_horizon"])])
                        action = bundle["postprocessor"](action_queue.popleft().unsqueeze(0))
                        action = env_postprocessor({ACTION: action})[ACTION]
                        action_numpy = action.detach().cpu().float().numpy()[0]
                        executed_actions.append(action_numpy)
                        recorder.append_executed_action(action_numpy)
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
                                "elapsed_ms": (time.perf_counter_ns() - arm_started) / 1_000_000,
                                "step": step,
                                "replan_index": replan_index,
                                "observation_hashes": observation_before,
                                "processed_policy_input_sha256": last_input_hash,
                                "actually_executed_action": action_numpy.tolist(),
                                "end_effector_pose": __import__("run_replay_v2").json_ready(
                                    robot_state.get("eef")
                                ),
                                "gripper_command": float(action_numpy[-1]),
                                "gripper_state": __import__("run_replay_v2").json_ready(
                                    robot_state.get("gripper")
                                ),
                                "object_poses": poses,
                                "contact_pairs": pairs,
                                "success_predicates": {"libero_check_success": success},
                                "termination_reason": (
                                    "SUCCESS"
                                    if success
                                    else "ENV_TERMINATED"
                                    if terminated
                                    else "TRUNCATED"
                                    if truncated
                                    else None
                                ),
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
                    recorder.finish(
                        objective_event_snapshot(
                            final_state,
                            contacts=contact_pairs(env._env),
                            success=success,
                            termination_reason=termination_reason,
                        )
                    )
                    trace_manifest = write_trace(
                        output_root / "execution" / "traces" / case["case_id"] / arm / selected_attempt_id,
                        arm,
                        trace_rows,
                        predicted_chunks,
                        executed_actions,
                        recorder.records,
                    )
                    result_path = attempt_root / "result.json"
                    record = {
                        "schema_version": 1,
                        "status": "COMPLETED",
                        "case_id": str(case["case_id"]),
                        "task_id": str(case["task_id"]),
                        "task_name": str(case["task_name"]),
                        "arm": arm,
                        "model_id": "Base-10" if arm == BASE_ARM else "20k Snap-1",
                        "model_hash": base_hash if arm == BASE_ARM else selected_hash,
                        "manifest_hash": TASK10_MANIFEST_SHA256,
                        "shard_id": args.shard_id,
                        "shard_manifest_hash": shard_entry["shard_manifest_sha256"],
                        "initial_state_hash": case["initial_state_hash"],
                        "simulator_state_hash": case["simulator_state_hash"],
                        "qpos_qvel_hash": case["qpos_qvel_hash"],
                        "canonical_observation_hash_frozen": case["observation_hash"],
                        "canonical_observation_hash_execution": observed_hash,
                        "egl_render_variance_observed": render_variance,
                        "processor_hash": base_contract["sha256"],
                        "evaluator_hash": case["evaluator_hash"],
                        "noise_schedule_hash": case["noise_schedule_hash"],
                        "execution_horizon": int(case["execution_horizon"]),
                        "maximum_episode_length": max_steps,
                        "success": success,
                        "termination_reason": termination_reason,
                        "steps_run": len(executed_actions),
                        "trace_path": trace_manifest["path"],
                        "trace_hash": trace_manifest["sha256"],
                        "attempt_id": selected_attempt_id,
                        "retry_lineage": retry,
                        "attempt_started_ns": attempt_started_ns,
                        "attempt_finished_ns": time.time_ns(),
                        "diagnostic_derived_feature_used": False,
                        "new_model_trained": False,
                        "checkpoint_modified": False,
                    }
                    write_hashed_json(result_path, record)
                    admitted = load_json(result_path)
                    validate_result_record(
                        admitted,
                        case=case,
                        arm=arm,
                        base_model_hash=base_hash,
                        verify_files=True,
                    )
                    completed_or_validated += 1
                    policy.reset()
                except Exception as error:
                    failure = {
                        "status": "FAILED",
                        "case_id": case["case_id"],
                        "arm": arm,
                        "attempt_id": selected_attempt_id,
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                        "outcome_aggregation_accessed": False,
                        "automatic_retry": False,
                        "timestamp_ns": time.time_ns(),
                    }
                    (attempt_root / "failure.json").write_text(
                        json.dumps(failure, indent=2, sort_keys=True) + "\n"
                    )
                    append_infrastructure_log(
                        failure_log,
                        f"- `{args.shard_id}` `{case['case_id']}` `{arm}` `{selected_attempt_id}`: "
                        f"`{type(error).__name__}`; automatic retry `NO`; outcome aggregation accessed `NO`.",
                    )
                    raise
            print(
                f"shard={args.shard_id} case={case_index + 1}/30 infrastructure_status=COMPLETED arms=2/2",
                flush=True,
            )
        finally:
            env.close()

    shard_status = {
        "schema_version": 1,
        "status": "TASK11_SHARD_COMPLETED",
        "shard_id": args.shard_id,
        "task_id": shard["task_id"],
        "case_count": 30,
        "completed_or_integrity_validated_results": completed_or_validated,
        "skipped_hash_valid_completed_results": skipped_valid,
        "planned_results": 60,
        "shard_manifest_sha256": shard_entry["shard_manifest_sha256"],
        "repository_commit": execution["repository_commit"],
        "started_ns": shard_started_ns,
        "finished_ns": time.time_ns(),
        "interim_outcome_aggregation_performed": False,
    }
    status_path = output_root / "execution" / "shard_status" / f"{args.shard_id}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(shard_status, indent=2, sort_keys=True) + "\n")
    append_infrastructure_log(
        infra_log,
        f"- `{args.shard_id}` completed with `60/60` hash-valid results at unix ns `{shard_status['finished_ns']}`; no outcome aggregation performed.",
    )
    print(json.dumps({"status": "TASK11_SHARD_COMPLETED", "shard": args.shard_id, "results": 60}))


if __name__ == "__main__":
    main()
