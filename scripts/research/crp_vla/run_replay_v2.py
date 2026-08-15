#!/usr/bin/env python
"""Capture replayable LIBERO capsules and run the registered six arms in one process."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import platform
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import (
    ARM_SPECS,
    ARMS,
    array_sha256,
    classify_failure,
    file_sha256,
    load_capsule,
    structured_hash,
    validate_case_records,
    write_capsule,
)
from run_libero_paired_four_arm_pilot import (
    add_batch_dimension,
    configure_standard_libero,
    git_repository_value,
    observation_hashes,
)
from trajectory_instrumentation import (
    ReplanTraceRecorder,
    capture_libero_state,
    objective_event_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-revision", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def command_value(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def environment_fingerprint(libero_commit: str) -> dict[str, Any]:
    cuda_device = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_device,
        "nvidia_smi": command_value(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "mujoco": package_version("mujoco"),
        "mujoco_py": package_version("mujoco-py"),
        "robosuite": package_version("robosuite"),
        "libero_commit": libero_commit,
        "renderer": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
        },
    }


def camera_parameters(inner_env: Any) -> dict[str, Any]:
    sim = inner_env.sim
    model = sim.model
    names = []
    for index in range(int(model.ncam)):
        name = model.camera_id2name(index)
        names.append(
            {
                "id": index,
                "name": name,
                "fovy": float(model.cam_fovy[index]),
                "pos": np.asarray(model.cam_pos[index]).tolist(),
                "quat": np.asarray(model.cam_quat[index]).tolist(),
            }
        )
    return {"cameras": names}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    return value


def task_semantics(bddl_path: Path) -> dict[str, Any]:
    text = bddl_path.read_text()
    interest_match = re.search(r"\(:obj_of_interest\s+(.*?)\)\s*\(:init", text, flags=re.DOTALL)
    objects = re.findall(r"[A-Za-z0-9_]+", interest_match.group(1)) if interest_match else []
    goal_match = re.search(r"\(:goal\s+(.*?)\)\s*\)\s*$", text, flags=re.DOTALL)
    goal = " ".join(goal_match.group(1).split()) if goal_match else None
    return {
        "objects_of_interest": objects,
        "manipulated_object": objects[0] if objects else None,
        "receptacle_or_fixture": objects[1] if len(objects) > 1 else None,
        "goal_expression": goal,
    }


def body_poses(inner_env: Any, names_of_interest: list[str]) -> dict[str, dict[str, list[float]]]:
    sim = inner_env.sim
    poses = {}
    for index in range(1, int(sim.model.nbody)):
        name = sim.model.body_id2name(index)
        if not name:
            continue
        if names_of_interest and not any(token.lower() in name.lower() for token in names_of_interest):
            continue
        poses[name] = {
            "pos": np.asarray(sim.data.body_xpos[index]).tolist(),
            "quat": np.asarray(sim.data.body_xquat[index]).tolist(),
        }
    return poses


def contact_pairs(inner_env: Any) -> list[dict[str, Any]]:
    sim = inner_env.sim
    pairs = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        left = sim.model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}"
        right = sim.model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}"
        pairs.append(
            {
                "left": left,
                "right": right,
                "distance": float(contact.dist),
                "position": np.asarray(contact.pos).tolist(),
            }
        )
    return pairs


def collision_evidence(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    robot_tokens = ("robot", "panda")
    end_effector_tokens = ("gripper", "finger")
    environment_tokens = ("table", "floor", "wall", "cabinet", "counter")
    matched = []
    for pair in pairs:
        left = pair["left"].lower()
        right = pair["right"].lower()
        robot_environment = (
            any(token in left for token in robot_tokens)
            and not any(token in left for token in end_effector_tokens)
            and any(token in right for token in environment_tokens)
        ) or (
            any(token in right for token in robot_tokens)
            and not any(token in right for token in end_effector_tokens)
            and any(token in left for token in environment_tokens)
        )
        if robot_environment:
            matched.append(pair)
    return {
        "observed": bool(matched),
        "definition": "registered name-based robot/environment contact heuristic",
        "pairs": matched,
    }


def candidate_gripper_contact(pairs: list[dict[str, Any]], target_name: str | None) -> dict[str, Any] | None:
    gripper_tokens = ("gripper", "finger")
    excluded = ("robot", "panda", "table", "floor", "wall")
    for pair in pairs:
        for gripper_side, other_side in ((pair["left"], pair["right"]), (pair["right"], pair["left"])):
            matches_target = target_name is not None and target_name.lower() in other_side.lower()
            if (
                any(token in gripper_side.lower() for token in gripper_tokens)
                and not any(token in other_side.lower() for token in excluded)
                and matches_target
            ):
                return {**pair, "candidate_object": other_side}
    return None


def make_event_state(initial_poses: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    target_available = semantics["manipulated_object"] is not None
    contact = {
        "available": target_available,
        "observed": False,
        "target": semantics["manipulated_object"],
    }
    receptacle = {
        "available": semantics["receptacle_or_fixture"] is not None,
        "observed": False,
        "target": semantics["receptacle_or_fixture"],
    }
    return {
        "first_target_contact": contact,
        "first_gripper_close": {"available": True, "observed": False},
        "first_object_lift": {"available": True, "observed": False},
        "first_receptacle_entry": receptacle,
        "collision": {"available": True, "observed": False},
        "object_dropped": {"available": True, "observed": False},
        "initial_object_poses": initial_poses,
    }


def update_events(
    events: dict[str, Any],
    step: int,
    action: np.ndarray,
    pairs: list[dict[str, Any]],
    poses: dict[str, Any],
    semantics: dict[str, Any],
    success: bool,
) -> None:
    if action[-1] > 0 and not events["first_gripper_close"]["observed"]:
        events["first_gripper_close"] = {
            "available": True,
            "observed": True,
            "step": step,
            "command": float(action[-1]),
        }
    candidate = candidate_gripper_contact(pairs, semantics["manipulated_object"])
    if candidate is not None and not events["first_target_contact"]["observed"]:
        events["first_target_contact"] = {
            "available": True,
            "observed": True,
            "step": step,
            "target": semantics["manipulated_object"],
            "contact": candidate,
        }
    collision = collision_evidence(pairs)
    if collision["observed"] and not events["collision"]["observed"]:
        events["collision"] = {"available": True, "step": step, **collision}
    initial = events["initial_object_poses"]
    for name, pose in poses.items():
        if name not in initial:
            continue
        delta_z = float(pose["pos"][2] - initial[name]["pos"][2])
        if delta_z > 0.03 and not events["first_object_lift"]["observed"]:
            events["first_object_lift"] = {
                "available": True,
                "observed": True,
                "step": step,
                "body": name,
                "delta_z": delta_z,
            }
        lifted = events["first_object_lift"]
        if lifted.get("observed") and lifted.get("body") == name and delta_z < 0.01:
            events["object_dropped"] = {
                "available": True,
                "observed": True,
                "step": step,
                "body": name,
                "delta_z": delta_z,
                "lift_step": lifted["step"],
            }
    if success and not events["first_receptacle_entry"]["observed"]:
        events["first_receptacle_entry"] = {
            "available": events["first_receptacle_entry"]["available"],
            "observed": True,
            "step": step,
            "target": semantics["receptacle_or_fixture"],
            "evidence": "LIBERO success predicate first became true",
        }


def write_trace(
    trace_dir: Path,
    arm: str,
    rows: list[dict[str, Any]],
    predicted_chunks: list[np.ndarray],
    executed_actions: list[np.ndarray],
    replan_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    json_path = trace_dir / f"{arm}.trace.jsonl.gz"
    with gzip.open(json_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), sort_keys=True, separators=(",", ":")) + "\n")
    numeric_path = trace_dir / f"{arm}.actions.npz"
    np.savez_compressed(
        numeric_path,
        predicted_chunks=np.asarray(predicted_chunks, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
    )
    manifest_path = trace_dir / f"{arm}.sha256.json"
    manifest = {
        "trace_jsonl_gz": {"path": str(json_path.resolve()), "sha256": file_sha256(json_path)},
        "numeric_actions": {"path": str(numeric_path.resolve()), "sha256": file_sha256(numeric_path)},
        "replan_records": replan_records or [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)}


def case_slug(case: dict[str, Any]) -> str:
    return (
        f"{case['suite']}-task{int(case['task_id']):02d}-"
        f"init{int(case['init_state_id']):02d}-seed{int(case['env_seed'])}"
    )


def main() -> None:
    args = parse_args()
    if args.execution_horizon < 1:
        raise ValueError("execution-horizon must be positive")
    if git_value("status", "--porcelain"):
        raise ValueError("Repository must be clean before an admitted replay-v2 run")
    design = json.loads(args.design.read_text())
    if design.get("phase") != args.phase:
        raise ValueError("Design phase and command phase differ")
    cases = design.get("cases", [])
    expected_cases = 8 if args.phase == "smoke" else 40
    if len(cases) != expected_cases:
        raise ValueError(f"{args.phase} design must contain exactly {expected_cases} cases")
    suites_in_design = {case["suite"] for case in cases}
    if args.phase == "smoke" and any(
        sum(case["suite"] == suite for case in cases) != 2 for suite in suites_in_design
    ):
        raise ValueError("Smoke design must contain exactly two cases per suite")
    if args.phase == "development" and len({(case["suite"], int(case["task_id"])) for case in cases}) != 40:
        raise ValueError("Development design must cover all 40 tasks exactly once")

    phase_root = args.output_root / args.phase
    configure_standard_libero(args.libero_root, phase_root / "libero_standard_config")
    repository_commit = git_value("rev-parse", "HEAD")
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    base_sha = file_sha256(args.base_checkpoint / "model.safetensors")
    snap_sha = file_sha256(args.snap_checkpoint / "model.safetensors")

    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    if Path(libero_module.get_assets_path()).resolve() != expected_assets:
        raise RuntimeError("LIBERO assets did not resolve to the pinned checkout")

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    def load_policy(checkpoint: Path, revision: str):
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = args.device
        config.load_vlm_weights = False
        config.n_action_steps = args.execution_horizon
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, revision=revision, strict=False)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        processor_files = sorted(checkpoint.glob("policy_*processor*"))
        return {
            "policy": policy,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "processor_manifest": {
                path.name: file_sha256(path) for path in processor_files if path.is_file()
            },
        }

    policies = {
        "base": load_policy(args.base_checkpoint, args.base_revision),
        "snap": load_policy(args.snap_checkpoint, args.snap_revision),
    }
    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in sorted(suites_in_design)}
    env_fingerprint = environment_fingerprint(libero_commit)
    results: list[dict[str, Any]] = []
    run_started_ns = time.time_ns()

    for case_index, case in enumerate(cases):
        suite_name = case["suite"]
        task_id = int(case["task_id"])
        init_state_id = int(case["init_state_id"])
        env_seed = int(case["env_seed"])
        max_steps = int(case["max_steps"])
        suite = suites[suite_name]
        prompt = suite.get_task(task_id).language
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
        slug = case_slug(case)
        try:
            env.init_state_id = init_state_id
            env.reset(seed=env_seed)
            if env._env is None:
                raise RuntimeError("LIBERO inner environment was not initialized")
            initial_sim_state = np.asarray(env._env.get_sim_state()).copy()
            canonical_raw = env._env.set_init_state(initial_sim_state)
            canonical_observation = env._format_raw_obs(canonical_raw)
            initial_batch = preprocess_observation(add_batch_dimension(canonical_observation))
            initial_batch["task"] = [prompt]
            base_batch = policies["base"]["preprocessor"](env_preprocessor(copy.deepcopy(initial_batch)))
            snap_batch = policies["snap"]["preprocessor"](env_preprocessor(copy.deepcopy(initial_batch)))
            base_input_sha = structured_hash(base_batch)
            snap_input_sha = structured_hash(snap_batch)
            if base_input_sha != snap_input_sha:
                raise RuntimeError(f"Base/Snap canonical processors disagree for {slug}")

            replans = (max_steps + args.execution_horizon - 1) // args.execution_horizon
            noise_generator = torch.Generator(device="cpu").manual_seed(
                args.noise_seed + case_index * 100_000
            )
            noise_schedule = torch.randn(
                (
                    replans,
                    1,
                    policies["base"]["policy"].config.chunk_size,
                    policies["base"]["policy"].config.max_action_dim,
                ),
                generator=noise_generator,
                dtype=torch.float32,
            ).numpy()
            if (
                policies["base"]["policy"].config.chunk_size != policies["snap"]["policy"].config.chunk_size
                or policies["base"]["policy"].config.max_action_dim
                != policies["snap"]["policy"].config.max_action_dim
            ):
                raise RuntimeError("Base/Snap action-noise shapes differ")
            bddl_path = Path(env._task_bddl_file)
            semantics = task_semantics(bddl_path)
            evaluator_path = Path(type(env).step.__code__.co_filename)
            metadata = {
                "schema_version": 2,
                "protocol": "Replayable Paired Closed-Loop Protocol v2",
                "phase": args.phase,
                "case_index": case_index,
                "suite": suite_name,
                "task_id": task_id,
                "task_name": suite.get_task(task_id).name,
                "instruction": prompt,
                "bddl_path": str(bddl_path.resolve()),
                "bddl_sha256": file_sha256(bddl_path),
                "task_semantics": semantics,
                "init_state_id": init_state_id,
                "env_seed": env_seed,
                "camera_parameters": camera_parameters(env._env),
                "environment_fingerprint": env_fingerprint,
                "normalization_version": policies["base"]["processor_manifest"],
                "policy_processor_versions": {
                    "base": policies["base"]["processor_manifest"],
                    "snap": policies["snap"]["processor_manifest"],
                },
                "noise_seed": args.noise_seed + case_index * 100_000,
                "noise_schedule_sha256": array_sha256(noise_schedule),
                "success_evaluator_path": str(evaluator_path.resolve()),
                "success_evaluator_sha256": file_sha256(evaluator_path),
                "execution_horizon": args.execution_horizon,
                "maximum_episode_length": max_steps,
                "repository_commit": repository_commit,
                "repository_dirty": False,
                "libero_commit": libero_commit,
                "base_checkpoint_revision": args.base_revision,
                "base_checkpoint_sha256": base_sha,
                "snap_checkpoint_revision": args.snap_revision,
                "snap_checkpoint_sha256": snap_sha,
                "canonical_input_sha256": base_input_sha,
            }
            payload = {
                "simulator": {
                    "state": initial_sim_state,
                    "qpos": np.asarray(env._env.sim.data.qpos).copy(),
                    "qvel": np.asarray(env._env.sim.data.qvel).copy(),
                },
                "raw_initial_observation": canonical_observation,
                "canonical_policy_batch": base_batch,
                "noise_schedule": noise_schedule,
            }
            capsule_dir = args.output_root / "capsules" / args.phase / slug
            capsule_record = write_capsule(capsule_dir, metadata, payload)
            capsule_sha = capsule_record["sha256"]
            loaded_metadata, loaded_payload = load_capsule(capsule_dir, args.device)
            if structured_hash(loaded_payload["canonical_policy_batch"]) != base_input_sha:
                raise RuntimeError(f"Capsule canonical tensor round trip failed for {slug}")
            initial_poses = body_poses(env._env, semantics["objects_of_interest"])

            case_results = []
            for arm in ARMS:
                policy_key, nfe, target_time = ARM_SPECS[arm]
                bundle = policies[policy_key]
                policy = bundle["policy"]
                policy.reset()
                policy.config.num_steps = nfe
                env.init_state_id = init_state_id
                env.reset(seed=env_seed)
                env._env.set_init_state(loaded_payload["simulator"]["state"])
                restored_state = np.asarray(env._env.get_sim_state())
                restored_sha = array_sha256(restored_state)
                expected_state_sha = array_sha256(initial_sim_state)
                if restored_sha != expected_state_sha:
                    raise RuntimeError(f"Exact simulator-state restore failed for {slug}/{arm}")
                observation = copy.deepcopy(canonical_observation)
                action_queue: deque[torch.Tensor] = deque()
                trace_rows = []
                predicted_chunks = []
                executed_actions = []
                events = make_event_state(initial_poses, semantics)
                success = False
                termination_reason = "MAX_STEPS"
                replan_index = -1
                last_input_sha = loaded_metadata["canonical_input_sha256"]
                arm_start = time.perf_counter_ns()
                processor_contract = bundle["processor_manifest"]
                replan_recorder = ReplanTraceRecorder(
                    phase_root / "traces" / slug / f"{arm}.replans",
                    identity_base={
                        "case_id": slug,
                        "task_id": f"{suite_name}:{task_id}",
                        "arm_id": arm,
                        "rollout_id": f"{args.phase}:{slug}:{arm}",
                    },
                    language_condition={
                        "instruction": prompt,
                        "reference": str(bddl_path.resolve()),
                    },
                    processor_contract=processor_contract,
                    processor_hash=structured_hash(processor_contract),
                    execution_horizon=args.execution_horizon,
                    nfe=nfe,
                    noise_seed=int(loaded_metadata["noise_seed"]),
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
                            batch = copy.deepcopy(loaded_payload["canonical_policy_batch"])
                        else:
                            batch = preprocess_observation(add_batch_dimension(observation))
                            batch["task"] = [prompt]
                            batch = bundle["preprocessor"](env_preprocessor(batch))
                        last_input_sha = structured_hash(batch)
                        noise = torch.from_numpy(noise_schedule[replan_index].copy()).to(args.device)
                        kwargs = {} if target_time is None else {"target_time": target_time}
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        policy_start = time.perf_counter_ns()
                        with torch.inference_mode():
                            chunk = policy.predict_action_chunk(batch, noise=noise, **kwargs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        latency_ms = (time.perf_counter_ns() - policy_start) / 1_000_000
                        predicted_chunks.append(chunk[0].detach().cpu().float().numpy())
                        denormalized_chunk = bundle["postprocessor"](
                            chunk[0, : args.execution_horizon].detach().clone()
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
                        action_queue.extend(chunk[0, : args.execution_horizon])
                    action = bundle["postprocessor"](action_queue.popleft().unsqueeze(0))
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
                            "processed_policy_input_sha256": last_input_sha,
                            "predicted_full_action_chunk_index": replan_index,
                            "actually_executed_action": action_numpy.tolist(),
                            "end_effector_pose": json_ready(robot_state.get("eef")),
                            "gripper_command": float(action_numpy[-1]),
                            "gripper_state": json_ready(robot_state.get("gripper")),
                            "object_poses": poses,
                            "contact_pairs": pairs,
                            "collision": collision_evidence(pairs),
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
                            "events": {
                                key: value for key, value in events.items() if key != "initial_object_poses"
                            },
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

                trace_manifest = write_trace(
                    phase_root / "traces" / slug,
                    arm,
                    trace_rows,
                    predicted_chunks,
                    executed_actions,
                    replan_recorder.records,
                )
                event_record = {key: value for key, value in events.items() if key != "initial_object_poses"}
                classification = classify_failure(event_record, success)
                result = {
                    "status": "COMPLETED",
                    "phase": args.phase,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": prompt,
                    "init_state_id": init_state_id,
                    "env_seed": env_seed,
                    "arm": arm,
                    "checkpoint": policy_key,
                    "nfe": nfe,
                    "target_time": target_time,
                    "execution_horizon": args.execution_horizon,
                    "success": success,
                    "steps_run": len(executed_actions),
                    "success_timestep": len(executed_actions) if success else None,
                    "termination_reason": termination_reason,
                    "capsule_path": str(capsule_dir.resolve()),
                    "capsule_sha256": capsule_sha,
                    "initial_sim_state_sha256": expected_state_sha,
                    "canonical_input_sha256": base_input_sha,
                    "noise_schedule_sha256": loaded_metadata["noise_schedule_sha256"],
                    "used_noise_prefix_length": len(predicted_chunks),
                    "evaluator_sha256": loaded_metadata["success_evaluator_sha256"],
                    "action_stream_sha256": array_sha256(np.asarray(executed_actions)),
                    "latency_ms_per_replan": [
                        row["policy_latency_ms"] for row in trace_rows if row["policy_latency_ms"] is not None
                    ],
                    "events": event_record,
                    "failure_classification": classification,
                    "trace_manifest": trace_manifest,
                }
                case_results.append(result)
                results.append(result)
                partial = {
                    "schema_version": 2,
                    "status": "RUNNING_PARTIAL",
                    "phase": args.phase,
                    "repository_commit": repository_commit,
                    "design_sha256": file_sha256(args.design),
                    "completed_rollout_count": len(results),
                    "planned_rollout_count": len(cases) * len(ARMS),
                    "results": results,
                }
                phase_root.mkdir(parents=True, exist_ok=True)
                (phase_root / "run.partial.json").write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"phase={args.phase} case={case_index + 1}/{len(cases)} "
                    f"arm={arm} success={int(success)} steps={len(executed_actions)}",
                    flush=True,
                )
                policy.reset()
            validate_case_records(case_results)
        finally:
            env.close()

    final = {
        "schema_version": 2,
        "status": "INFRASTRUCTURE_SMOKE_PASSED" if args.phase == "smoke" else "DEVELOPMENT_COMPLETED",
        "phase": args.phase,
        "formal_gate_unchanged": True,
        "crp_training_hold": True,
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "libero_commit": libero_commit,
        "design": str(args.design.resolve()),
        "design_sha256": file_sha256(args.design),
        "base_checkpoint_sha256": base_sha,
        "snap_checkpoint_sha256": snap_sha,
        "environment_fingerprint": env_fingerprint,
        "run_started_ns": run_started_ns,
        "run_finished_ns": time.time_ns(),
        "case_count": len(cases),
        "rollout_count": len(results),
        "arm_order": list(ARMS),
        "arm_specs": ARM_SPECS,
        "results": results,
    }
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "run_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": final["status"], "rollouts": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
