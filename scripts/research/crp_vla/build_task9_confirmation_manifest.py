#!/usr/bin/env python
"""Build the disjoint Task 9 confirmation manifest without executing actions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256
from run_libero_paired_four_arm_pilot import (
    configure_standard_libero,
    git_repository_value,
    observation_hashes,
)
from task9_common import (
    SELECTED_MODEL_SHA256,
    assert_no_state_overlap,
    canonical_json_sha256,
    deterministic_diagnostic_subset,
    effective_processor_contract,
    file_sha256,
    load_json,
    require_model_hash,
    validate_confirmation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--first-init-state", type=int, default=5)
    parser.add_argument("--states-per-task", type=int, default=15)
    parser.add_argument("--diagnostic-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    files = sorted(checkpoint.glob("policy_*processor*"))
    manifest = {path.name: file_sha256(path) for path in files if path.is_file()}
    if not manifest:
        raise ValueError(f"No processor files at {checkpoint}")
    return manifest


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def environment_metadata(libero_root: Path) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "libero_commit": git_repository_value(libero_root, "rev-parse", "HEAD"),
        "renderer": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
        },
    }


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_processors = processor_manifest(args.base_checkpoint)
    selected_processors = processor_manifest(args.selected_checkpoint)
    base_contract = effective_processor_contract(args.base_checkpoint)
    selected_contract = effective_processor_contract(args.selected_checkpoint)
    if base_contract["sha256"] != selected_contract["sha256"]:
        raise ValueError("Base/selected effective state-action processor contracts differ")
    processor_hash = base_contract["sha256"]

    formal = load_json(args.formal_manifest)
    development = load_json(args.development_manifest)
    formal_state_hashes = {
        value for result in formal["results"] for value in (result.get("initial_sim_state_sha256"),) if value
    }
    dev_state_hashes = {
        value
        for result in development["results"]
        for value in (result.get("initial_sim_state_sha256"),)
        if value
    }
    formal_case_ids = {
        (result["suite"], int(result["task_id"]), int(result["init_state_id"]))
        for result in formal["results"]
    }
    dev_case_ids = {
        (result["suite"], int(result["task_id"]), int(result["init_state_id"]))
        for result in development["results"]
    }

    configure_standard_libero(args.libero_root, output_root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    if Path(libero_module.get_assets_path()).resolve() != expected_assets:
        raise RuntimeError("LIBERO assets did not resolve to the pinned checkout")

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

    runtime = environment_metadata(args.libero_root)
    suite_names = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    factories = benchmark.get_benchmark_dict()
    suites = {name: factories[name]() for name in suite_names}
    cases = []
    case_index = 0
    for suite_name in suite_names:
        suite = suites[suite_name]
        if suite.n_tasks != 10:
            raise ValueError(f"Pinned suite {suite_name} has {suite.n_tasks} tasks, expected 10")
        for task_id in range(suite.n_tasks):
            env_config = LiberoEnvConfig(
                task=suite_name,
                task_ids=[task_id],
                observation_height=256,
                observation_width=256,
            )
            max_steps = int(TASK_SUITE_MAX_STEPS[suite_name])
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
                episode_index=args.first_init_state,
                n_envs=1,
                num_steps_wait=10,
                camera_name_mapping=env_config.camera_name_mapping,
                control_freq=env_config.fps,
                control_mode=env_config.control_mode,
                is_libero_plus=env_config.is_libero_plus,
                hard_reset=env_config.hard_reset,
            )
            try:
                source_states = np.asarray(suite.get_task_init_states(task_id))
                end = args.first_init_state + args.states_per_task
                if len(source_states) < end:
                    raise ValueError(f"Task {suite_name}:{task_id} has only {len(source_states)} states")
                evaluator_path: Path | None = None
                for init_state_id in range(args.first_init_state, end):
                    env_seed = args.seed + case_index * 100_000
                    env.init_state_id = init_state_id
                    env.reset(seed=env_seed)
                    if env._env is None:
                        raise RuntimeError("LIBERO inner environment was not initialized")
                    simulator_state = np.asarray(env._env.get_sim_state()).copy()
                    canonical_raw = env._env.set_init_state(simulator_state)
                    observation = env._format_raw_obs(copy.deepcopy(canonical_raw))
                    qpos = np.asarray(env._env.sim.data.qpos).copy()
                    qvel = np.asarray(env._env.sim.data.qvel).copy()
                    initial_state = np.asarray(source_states[init_state_id]).copy()
                    structured_sim_hash = array_sha256(simulator_state)
                    legacy_sim_hash = raw_array_sha256(simulator_state)
                    initial_state_hash = array_sha256(initial_state)
                    qpos_qvel_hash = canonical_json_sha256(
                        {"qpos": array_sha256(qpos), "qvel": array_sha256(qvel)}
                    )
                    observation_hash = canonical_json_sha256(observation_hashes(observation))
                    evaluator_path = Path(type(env).step.__code__.co_filename)
                    evaluator_hash = file_sha256(evaluator_path)
                    replans = (max_steps + 9) // 10
                    noise_seed = args.seed + case_index * 100_000
                    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                    noise = torch.randn(
                        (replans, 1, 50, 32), generator=generator, dtype=torch.float32
                    ).numpy()
                    key = (suite_name, task_id, init_state_id)
                    formal_overlap = key in formal_case_ids or bool(
                        {structured_sim_hash, legacy_sim_hash, initial_state_hash, qpos_qvel_hash}
                        & formal_state_hashes
                    )
                    dev_overlap = key in dev_case_ids or bool(
                        {structured_sim_hash, legacy_sim_hash, initial_state_hash, qpos_qvel_hash}
                        & dev_state_hashes
                    )
                    case_id = f"confirmation-{suite_name}-task{task_id:02d}-state{init_state_id:02d}"
                    cases.append(
                        {
                            "case_id": case_id,
                            "task_id": f"{suite_name}:{task_id}",
                            "suite": suite_name,
                            "suite_task_id": task_id,
                            "task_name": suite.get_task(task_id).name,
                            "task_instruction": suite.get_task(task_id).language,
                            "initial_state_id": init_state_id,
                            "initial_state_source": f"pinned_libero:{suite_name}:task{task_id}:get_task_init_states[{init_state_id}]",
                            "initial_state_hash": initial_state_hash,
                            "simulator_state_hash": structured_sim_hash,
                            "legacy_simulator_state_sha256": legacy_sim_hash,
                            "qpos_qvel_hash": qpos_qvel_hash,
                            "observation_hash": observation_hash,
                            "processor_hash": processor_hash,
                            "evaluator_hash": evaluator_hash,
                            "runtime_metadata": runtime,
                            "renderer_metadata": runtime["renderer"],
                            "maximum_episode_length": max_steps,
                            "execution_horizon": 10,
                            "noise_seed": noise_seed,
                            "noise_shape": list(noise.shape),
                            "noise_generation": "torch.Generator(cpu).manual_seed(seed); torch.randn(float32)",
                            "noise_schedule_hash": array_sha256(noise),
                            "formal100_overlap_check": formal_overlap,
                            "dev40_overlap_check": dev_overlap,
                        }
                    )
                    case_index += 1
                    if case_index % 25 == 0:
                        print(f"manifest_state={case_index}/600", flush=True)
            finally:
                env.close()

    manifest = {
        "schema_version": 1,
        "status": "CONFIRMATION_CASE_MANIFEST_FROZEN",
        "purpose": "prospective disjoint confirmation; no rollout executed",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty_at_generation": bool(git_value("status", "--porcelain")),
        "selected_checkpoint_step": 20_000,
        "selected_model_sha256": SELECTED_MODEL_SHA256,
        "base_processor_manifest": base_processors,
        "selected_processor_manifest": selected_processors,
        "effective_processor_contract": base_contract,
        "processor_equivalence_evidence": "state/action tensors and effective configs exact; replay-v2 canonical inputs exact on all 40 task prompts",
        "processor_hash": processor_hash,
        "formal100_manifest_sha256": file_sha256(args.formal_manifest),
        "dev40_manifest_sha256": file_sha256(args.development_manifest),
        "task_count": 40,
        "cases_per_task": args.states_per_task,
        "case_count": len(cases),
        "cases": cases,
    }
    validate_confirmation_manifest(manifest)
    assert_no_state_overlap(cases, formal_state_hashes, dev_state_hashes)
    manifest_path = output_root / "confirmation_case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subset = deterministic_diagnostic_subset(cases, per_task=args.diagnostic_per_task, seed=args.seed)
    subset["confirmation_manifest_sha256"] = file_sha256(manifest_path)
    (output_root / "confirmation_diagnostic_subset.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n"
    )
    overlap = {
        "schema_version": 1,
        "status": "ZERO_OVERLAP_VERIFIED",
        "formal100_case_identity_overlap_count": sum(case["formal100_overlap_check"] for case in cases),
        "dev40_case_identity_overlap_count": sum(case["dev40_overlap_check"] for case in cases),
        "confirmation_case_count": len(cases),
        "unique_initial_state_hashes": len({case["initial_state_hash"] for case in cases}),
        "unique_simulator_state_hashes": len({case["simulator_state_hash"] for case in cases}),
        "unique_qpos_qvel_hashes": len({case["qpos_qvel_hash"] for case in cases}),
        "formal100_state_hash_count": len(formal_state_hashes),
        "dev40_state_hash_count": len(dev_state_hashes),
    }
    (output_root / "confirmation_overlap_audit.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n"
    )
    hashes = {
        path.name: file_sha256(path)
        for path in (
            manifest_path,
            output_root / "confirmation_diagnostic_subset.json",
            output_root / "confirmation_overlap_audit.json",
        )
    }
    (output_root / "confirmation_manifest_sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "CONFIRMATION_MANIFEST_FROZEN", **overlap}, sort_keys=True))


if __name__ == "__main__":
    main()
