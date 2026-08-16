#!/usr/bin/env python
"""Build the outcome-blind, state-disjoint 400-case Task 13 Dev-B manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import array_sha256, structured_hash
from run_libero_paired_four_arm_pilot import configure_standard_libero
from task9_common import file_sha256, load_json, require_model_hash
from task13_common import (
    DEV_B_CASES,
    DEV_B_FIRST_STATE,
    DEV_B_SEED,
    DEV_B_STATES_PER_TASK,
    SELECTED_MODEL_SHA256,
    STATE_IDENTITY_FIELDS,
    TASK13_ARMS,
    validate_dev_b_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--task12-query-metrics", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def result_identity_sets(payload: dict[str, Any]) -> tuple[set[tuple[str, int, int]], set[str]]:
    rows = payload.get("results", payload.get("cases", []))
    keys: set[tuple[str, int, int]] = set()
    hashes: set[str] = set()
    for row in rows:
        suite = str(row["suite"])
        task = int(row.get("suite_task_id", row.get("task_id")))
        state = int(row.get("initial_state_id", row.get("init_state_id")))
        keys.add((suite, task, state))
        for field in (
            "initial_state_hash",
            "initial_sim_state_sha256",
            "simulator_state_hash",
            "qpos_qvel_hash",
        ):
            if row.get(field):
                hashes.add(str(row[field]))
    return keys, hashes


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    if root != args.preregistration_root.resolve():
        raise RuntimeError("Dev-B must be written beside the frozen preregistration")
    if not (root / "TASK13_PREREGISTRATION.md").is_file():
        raise RuntimeError("Task 13 preregistration is missing")
    if (root / "DEV_B_MANIFEST.json").exists():
        raise FileExistsError("Refusing to overwrite Dev-B")
    require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")

    old_sources = {
        "formal100": result_identity_sets(load_json(args.formal_manifest)),
        "old_dev40": result_identity_sets(load_json(args.development_manifest)),
        "confirmation1200": result_identity_sets(load_json(args.confirmation1200_manifest)),
    }
    task12_state_hashes = set()
    with args.task12_query_metrics.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("state_payload_sha256"):
                task12_state_hashes.add(str(row["state_payload_sha256"]))

    configure_standard_libero(args.libero_root, root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen = {field: set() for field in STATE_IDENTITY_FIELDS}
    suites = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    factories = benchmark.get_benchmark_dict()
    case_index = 0
    for suite_name in suites:
        suite = factories[suite_name]()
        for task_id in range(suite.n_tasks):
            source_states = np.asarray(suite.get_task_init_states(task_id))
            if len(source_states) < DEV_B_FIRST_STATE + DEV_B_STATES_PER_TASK:
                raise RuntimeError(f"Insufficient pinned states for {suite_name}:{task_id}")
            for init_state_id in range(DEV_B_FIRST_STATE, DEV_B_FIRST_STATE + DEV_B_STATES_PER_TASK):
                env_seed = DEV_B_SEED + case_index * 100_000
                config = LiberoEnvConfig(
                    task=suite_name,
                    task_ids=[task_id],
                    observation_height=256,
                    observation_width=256,
                )
                env = LiberoEnv(
                    task_suite=suite,
                    task_id=task_id,
                    task_suite_name=suite_name,
                    episode_length=int(TASK_SUITE_MAX_STEPS[suite_name]),
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
                try:
                    env.init_state_id = init_state_id
                    env.reset(seed=env_seed)
                    if env._env is None:
                        raise RuntimeError("LIBERO inner environment unavailable")
                    initial_state = np.asarray(source_states[init_state_id]).copy()
                    simulator_state = np.asarray(env._env.get_sim_state()).copy()
                    qpos_qvel = {
                        "qpos": np.asarray(env._env.sim.data.qpos).copy(),
                        "qvel": np.asarray(env._env.sim.data.qvel).copy(),
                    }
                    identities = {
                        "initial_state_hash": array_sha256(initial_state),
                        "simulator_state_hash": array_sha256(simulator_state),
                        "qpos_qvel_hash": structured_hash(qpos_qvel),
                    }
                    key = (suite_name, task_id, init_state_id)
                    reasons = []
                    for source_name, (old_keys, old_hashes) in old_sources.items():
                        if key in old_keys or set(identities.values()) & old_hashes:
                            reasons.append(f"{source_name}_overlap")
                    if set(identities.values()) & task12_state_hashes:
                        reasons.append("task12_mechanism_or_probe_state_overlap")
                    for field, value in identities.items():
                        if value in seen[field]:
                            reasons.append(f"duplicate_{field}")
                    if reasons:
                        rejected.append(
                            {
                                "suite": suite_name,
                                "task_id": task_id,
                                "init_state_id": init_state_id,
                                "reasons": sorted(set(reasons)),
                                "replacement_attempted": False,
                            }
                        )
                        (root / "DEV_B_REJECTED_CANDIDATES.json").write_text(
                            json.dumps(rejected, indent=2) + "\n"
                        )
                        raise RuntimeError(f"Dev-B fixed candidate rejected: {rejected[-1]}")
                    for field, value in identities.items():
                        seen[field].add(value)
                    cases.append(
                        {
                            "case_id": f"dev-b-{suite_name}-task{task_id:02d}-state{init_state_id:02d}",
                            "task_key": f"{suite_name}:{task_id}",
                            "suite": suite_name,
                            "task_id": task_id,
                            "task_name": suite.get_task(task_id).name,
                            "task_instruction": suite.get_task(task_id).language,
                            "init_state_id": init_state_id,
                            "env_seed": env_seed,
                            "max_steps": int(TASK_SUITE_MAX_STEPS[suite_name]),
                            **identities,
                            "outcome_accessed": False,
                        }
                    )
                    case_index += 1
                    if case_index % 20 == 0:
                        print(f"dev_b_state={case_index}/{DEV_B_CASES}", flush=True)
                finally:
                    env.close()

    manifest = {
        "schema_version": 1,
        "status": "TASK13_DEV_B_FROZEN",
        "phase": "development",
        "purpose": "prospective mechanism validation; outcomes unseen",
        "expected_case_count": DEV_B_CASES,
        "case_count": len(cases),
        "task_count": 40,
        "cases_per_task": DEV_B_STATES_PER_TASK,
        "first_state_index": DEV_B_FIRST_STATE,
        "registered_arms": list(TASK13_ARMS),
        "base_model_sha256": base_hash,
        "snap_model_sha256": SELECTED_MODEL_SHA256,
        "selection_rule": "fixed state indices 35-44 for every task; no replacement",
        "outcomes_accessed_before_freeze": False,
        "cases": cases,
    }
    validate_dev_b_manifest(manifest)
    manifest_path = root / "DEV_B_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "DEV_B_REJECTED_CANDIDATES.json").write_text(json.dumps(rejected, indent=2) + "\n")
    audit = {
        "status": "TASK13_DEV_B_ZERO_OVERLAP_VERIFIED",
        "case_count": len(cases),
        "cases_per_task": DEV_B_STATES_PER_TASK,
        "overlap": dict.fromkeys((*old_sources, "task12_mechanism_or_probe"), 0),
        "duplicates": dict.fromkeys(STATE_IDENTITY_FIELDS, 0),
        "task12_archived_state_hashes_checked": len(task12_state_hashes),
        "rejected_candidate_count": len(rejected),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (root / "DEV_B_OVERLAP_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    (root / "DEV_B_OVERLAP_AUDIT.md").write_text(
        "# Task 13 Dev-B Overlap Audit\n\n"
        "**Status: TASK13_DEV_B_ZERO_OVERLAP_VERIFIED**\n\n"
        f"- Cases: `{len(cases)}` (40 tasks x {DEV_B_STATES_PER_TASK}).\n"
        "- old dev40 / formal100 / Confirmation1200 overlap: `0 / 0 / 0`.\n"
        "- Task 12 mechanism/probe archived-state overlap: `0`.\n"
        "- Duplicate initial/simulator/qpos-qvel identities: `0 / 0 / 0`.\n"
        "- Candidate replacement: `NO`.\n"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
