#!/usr/bin/env python
"""Freeze the Task 10 1,200-case manifest without executing policy actions."""

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
    NONINFERIORITY_MARGIN,
    SELECTED_MODEL_SHA256,
    canonical_json_sha256,
    effective_processor_contract,
    file_sha256,
    load_json,
    require_model_hash,
    validate_confirmation_manifest,
)
from task10_common import (
    CASES_PER_TASK,
    PRIMARY_CASES,
    STATE_IDENTITY_FIELDS,
    confirmation1200_diagnostic_subset,
    validate_confirmation1200_manifest,
    validate_task9_subset,
)

NEW_FIRST_INIT_STATE = 20
NEW_STATES_PER_TASK = 15
TASK10_NOISE_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--task9-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    manifest = {
        path.name: file_sha256(path)
        for path in sorted(checkpoint.glob("policy_*processor*"))
        if path.is_file()
    }
    if not manifest:
        raise ValueError(f"No processor files at {checkpoint}")
    return manifest


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


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


def source_state_sets(manifest: dict[str, Any]) -> tuple[set[tuple[str, int, int]], set[str]]:
    case_ids = {
        (str(result["suite"]), int(result["task_id"]), int(result["init_state_id"]))
        for result in manifest["results"]
    }
    hashes = {
        str(value)
        for result in manifest["results"]
        for value in (
            result.get("initial_state_hash"),
            result.get("initial_sim_state_sha256"),
            result.get("qpos_qvel_hash"),
        )
        if value
    }
    return case_ids, hashes


def reject_or_append(
    *,
    case: dict[str, Any],
    accepted: list[dict[str, Any]],
    formal_hashes: set[str],
    development_hashes: set[str],
    task9_hashes: set[str],
    rejection_path: Path,
) -> None:
    reasons = []
    identities = {str(case[field]) for field in STATE_IDENTITY_FIELDS}
    if identities & formal_hashes:
        reasons.append("formal100_state_identity_overlap")
    if identities & development_hashes:
        reasons.append("dev40_state_identity_overlap")
    if identities & task9_hashes:
        reasons.append("task9_600_state_identity_overlap")
    for field in STATE_IDENTITY_FIELDS:
        if str(case[field]) in {str(item[field]) for item in accepted}:
            reasons.append(f"duplicate_{field}")
    if case["case_id"] in {item["case_id"] for item in accepted}:
        reasons.append("duplicate_case_id")
    if reasons:
        record = {
            "schema_version": 1,
            "status": "CANDIDATE_REJECTED_FAIL_CLOSED",
            "candidate": {
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "initial_state_id": case["initial_state_id"],
                **{field: case[field] for field in STATE_IDENTITY_FIELDS},
            },
            "rejection_reasons": sorted(set(reasons)),
            "replacement_attempted": False,
        }
        rejection_path.write_text(json.dumps([record], indent=2, sort_keys=True) + "\n")
        raise RuntimeError(
            f"Fixed candidate {case['case_id']} rejected; manifest generation failed closed: {record['rejection_reasons']}"
        )
    accepted.append(case)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rejection_path = output_root / "CONFIRMATION1200_REJECTED_CANDIDATES.json"
    rejection_path.write_text("[]\n")

    require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_processors = processor_manifest(args.base_checkpoint)
    selected_processors = processor_manifest(args.selected_checkpoint)
    base_contract = effective_processor_contract(args.base_checkpoint)
    selected_contract = effective_processor_contract(args.selected_checkpoint)
    if base_contract["sha256"] != selected_contract["sha256"]:
        raise ValueError("Base/selected effective state-action processor contracts differ")

    task9 = load_json(args.task9_manifest)
    validate_confirmation_manifest(task9)
    if task9.get("case_count") != 600 or task9.get("cases_per_task") != 15:
        raise ValueError("Input is not the frozen Task 9 40x15 manifest")
    formal = load_json(args.formal_manifest)
    development = load_json(args.development_manifest)
    formal_case_ids, formal_hashes = source_state_sets(formal)
    development_case_ids, development_hashes = source_state_sets(development)
    task9_hashes = {str(case[field]) for case in task9["cases"] for field in STATE_IDENTITY_FIELDS}

    old_cases = []
    for source in task9["cases"]:
        case = copy.deepcopy(source)
        case["task9_600_membership"] = True
        old_cases.append(case)
    accepted_new: list[dict[str, Any]] = []

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
    new_index = 0
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
                episode_index=NEW_FIRST_INIT_STATE,
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
                end = NEW_FIRST_INIT_STATE + NEW_STATES_PER_TASK
                if len(source_states) < end:
                    raise ValueError(f"Task {suite_name}:{task_id} has only {len(source_states)} states")
                for init_state_id in range(NEW_FIRST_INIT_STATE, end):
                    env_seed = TASK10_NOISE_SEED + (600 + new_index) * 100_000
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
                    generator = torch.Generator(device="cpu").manual_seed(env_seed)
                    noise = torch.randn(
                        (replans, 1, 50, 32), generator=generator, dtype=torch.float32
                    ).numpy()
                    key = (suite_name, task_id, init_state_id)
                    identities = {
                        structured_sim_hash,
                        legacy_sim_hash,
                        initial_state_hash,
                        qpos_qvel_hash,
                    }
                    case = {
                        "case_id": f"confirmation-{suite_name}-task{task_id:02d}-state{init_state_id:02d}",
                        "task_id": f"{suite_name}:{task_id}",
                        "suite": suite_name,
                        "suite_task_id": task_id,
                        "task_name": suite.get_task(task_id).name,
                        "task_instruction": suite.get_task(task_id).language,
                        "initial_state_id": init_state_id,
                        "initial_state_source": (
                            f"pinned_libero:{suite_name}:task{task_id}:get_task_init_states[{init_state_id}]"
                        ),
                        "initial_state_hash": initial_state_hash,
                        "simulator_state_hash": structured_sim_hash,
                        "legacy_simulator_state_sha256": legacy_sim_hash,
                        "qpos_qvel_hash": qpos_qvel_hash,
                        "observation_hash": observation_hash,
                        "processor_hash": base_contract["sha256"],
                        "evaluator_hash": evaluator_hash,
                        "runtime_metadata": runtime,
                        "renderer_metadata": runtime["renderer"],
                        "maximum_episode_length": max_steps,
                        "execution_horizon": 10,
                        "noise_seed": env_seed,
                        "noise_shape": list(noise.shape),
                        "noise_generation": "torch.Generator(cpu).manual_seed(seed); torch.randn(float32)",
                        "noise_schedule_hash": array_sha256(noise),
                        "formal100_overlap_check": key in formal_case_ids or bool(identities & formal_hashes),
                        "dev40_overlap_check": key in development_case_ids
                        or bool(identities & development_hashes),
                        "task9_600_membership": False,
                    }
                    reject_or_append(
                        case=case,
                        accepted=accepted_new,
                        formal_hashes=formal_hashes,
                        development_hashes=development_hashes,
                        task9_hashes=task9_hashes,
                        rejection_path=rejection_path,
                    )
                    new_index += 1
                    if new_index % 25 == 0:
                        print(f"new_manifest_state={new_index}/600", flush=True)
            finally:
                env.close()

    cases = sorted(old_cases + accepted_new, key=lambda case: str(case["case_id"]))
    manifest = {
        "schema_version": 2,
        "status": "CONFIRMATION1200_MANIFEST_FROZEN",
        "purpose": "prospective 1200-case disjoint confirmation; no rollout executed",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty_at_generation": bool(git_value("status", "--porcelain")),
        "selected_checkpoint_step": 20_000,
        "selected_model_sha256": SELECTED_MODEL_SHA256,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "success_definition_changed": False,
        "evaluator_changed": False,
        "simulator_termination_changed": False,
        "action_horizon_changed": False,
        "normalization_changed": False,
        "formal_confirmation_started": False,
        "confirmation_outcomes_available_at_freeze": False,
        "task9_manifest_sha256": file_sha256(args.task9_manifest),
        "task9_subset_case_count": len(old_cases),
        "new_case_count": len(accepted_new),
        "new_state_range_per_task": [NEW_FIRST_INIT_STATE, NEW_FIRST_INIT_STATE + NEW_STATES_PER_TASK - 1],
        "base_processor_manifest": base_processors,
        "selected_processor_manifest": selected_processors,
        "effective_processor_contract": base_contract,
        "processor_hash": base_contract["sha256"],
        "formal100_manifest_sha256": file_sha256(args.formal_manifest),
        "dev40_manifest_sha256": file_sha256(args.development_manifest),
        "task_count": 40,
        "cases_per_task": CASES_PER_TASK,
        "case_count": len(cases),
        "cases": cases,
    }
    validate_confirmation1200_manifest(manifest)
    validate_task9_subset(cases, task9["cases"])
    if len(cases) != PRIMARY_CASES:
        raise RuntimeError("Internal primary case count mismatch")

    manifest_path = output_root / "CONFIRMATION1200_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subset = confirmation1200_diagnostic_subset(cases)
    subset["confirmation1200_manifest_sha256"] = file_sha256(manifest_path)
    subset_path = output_root / "CONFIRMATION1200_DIAGNOSTIC_SUBSET.json"
    subset_path.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")

    audit = {
        "schema_version": 1,
        "status": "CONFIRMATION1200_ZERO_OVERLAP_AND_UNIQUENESS_VERIFIED",
        "primary_cases": len(cases),
        "tasks": len({case["task_id"] for case in cases}),
        "cases_per_task": CASES_PER_TASK,
        "task9_subset_cases": sum(bool(case["task9_600_membership"]) for case in cases),
        "new_cases": sum(not bool(case["task9_600_membership"]) for case in cases),
        "formal100_overlap": sum(bool(case["formal100_overlap_check"]) for case in cases),
        "dev40_overlap": sum(bool(case["dev40_overlap_check"]) for case in cases),
        "unique_initial_state_hashes": len({case["initial_state_hash"] for case in cases}),
        "unique_simulator_state_hashes": len({case["simulator_state_hash"] for case in cases}),
        "unique_qpos_qvel_hashes": len({case["qpos_qvel_hash"] for case in cases}),
        "cross_task_duplicate_initial_state_hashes": 0,
        "cross_task_duplicate_simulator_state_hashes": 0,
        "cross_task_duplicate_qpos_qvel_hashes": 0,
        "fixed_candidate_rejections": len(json.loads(rejection_path.read_text())),
        "silent_replacements": 0,
    }
    audit_path = output_root / "confirmation1200_overlap_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
