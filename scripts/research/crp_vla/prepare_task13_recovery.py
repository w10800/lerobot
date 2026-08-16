#!/usr/bin/env python
"""Quarantine Task 13 attempt001 and freeze the untouched 200-case Dev-B2."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import array_sha256, structured_hash
from run_libero_paired_four_arm_pilot import configure_standard_libero
from task9_common import file_sha256, load_json, require_model_hash
from task13_common import SELECTED_MODEL_SHA256, TASK13_ARMS
from task13_recovery_common import validate_frozen_cases, validate_task12_registry

DEV_B2_FIRST_STATE = 45
DEV_B2_STATES_PER_TASK = 5
DEV_B2_CASES = 200
DEV_B2_SEED = 20260818


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--used-state-registry", type=Path, required=True)
    parser.add_argument("--attempt001-manifest", type=Path, required=True)
    parser.add_argument("--attempt001-run", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--original-decision-rule", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def overlap_rows(cases: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    task12_blob = {
        row["identity_value"]
        for row in registry["entries"]
        if row["dataset"] in {"task12_mechanism_query", "task12_preservation_probe"}
        and row["identity_type"] == "state_blob_sha256"
    }
    old_by_dataset: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in registry["entries"]:
        old_by_dataset[row["dataset"]][row["identity_type"]].add(row["identity_value"])
    overlaps = []
    for case in cases:
        for dataset in ("old_dev40", "formal100", "task11_confirmation1200", "task13_attempt001_initial"):
            for field in ("initial_state_hash", "simulator_state_hash", "qpos_qvel_hash"):
                aliases = {field}
                if field == "simulator_state_hash":
                    aliases.add("initial_sim_state_sha256")
                if any(str(case[field]) in old_by_dataset[dataset][name] for name in aliases):
                    overlaps.append({"case_id": case["case_id"], "dataset": dataset, "identity_type": field})
        if str(case["simulator_state_hash"]) in task12_blob:
            overlaps.append(
                {
                    "case_id": case["case_id"],
                    "dataset": "task12_mechanism_or_probe",
                    "identity_type": "state_blob_sha256",
                }
            )
    return overlaps


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise RuntimeError("Recovery freeze requires a clean worktree")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    registry = load_json(args.used_state_registry)
    validate_task12_registry(registry)
    checked = int(registry.get("task12_unique_state_payload_hash_count", 0))
    records = int(registry.get("task12_authoritative_query_record_count", 0))
    if checked <= 0 or records <= 0:
        raise RuntimeError("Zero Task 12 archived states cannot pass recovery")
    attempt1 = load_json(args.attempt001_manifest)
    attempt1_run = load_json(args.attempt001_run)
    if attempt1_run.get("status") != "DEVELOPMENT_COMPLETED" or len(attempt1_run.get("results", [])) != 800:
        raise RuntimeError("Attempt001 quarantine requires the immutable complete run")
    if attempt1.get("case_count") != 400:
        raise RuntimeError("Attempt001 manifest cardinality drift")
    attempt1_overlaps = overlap_rows(attempt1["cases"], registry)
    attempt1_status = (
        "ATTEMPT001_STATE_CONTAMINATED"
        if attempt1_overlaps
        else "ATTEMPT001_POSTHOC_DISJOINT_BUT_PROCEDURALLY_INVALID"
    )
    (root / "ATTEMPT001_INVALIDATION_REPORT.md").write_text(
        "# Task 13 Attempt001 Invalidation\n\n"
        "**Status: TASK13_ATTEMPT001_PROCEDURALLY_INVALID_FOR_PRIMARY_REPLICATION**\n\n"
        "Attempt001 completed 800/800 technically valid rollouts, but its mandatory Task 12 archived-state audit checked zero states before outcomes. Its Base-10 336/400, Snap-1 316/400 result is permanently `DESCRIPTIVE_ONLY`; no H1/H2/H3 analysis is admitted as primary.\n"
    )
    (root / "ATTEMPT001_POSTHOC_OVERLAP_AUDIT.md").write_text(
        "# Attempt001 Post-hoc Overlap Audit\n\n"
        f"**Status: {attempt1_status}**\n\n"
        f"- Task 12 raw query records checked: `{records}`.\n"
        f"- Unique Task 12 archived state hashes: `{checked}`.\n"
        f"- Attempt001 overlap records: `{len(attempt1_overlaps)}`.\n"
        "- Prospective primary validity remains: `NO`, regardless of overlap count.\n\n"
        + (json.dumps(attempt1_overlaps, indent=2) if attempt1_overlaps else "No overlap identities found.\n")
    )
    (root / "TASK12_STATE_REGISTRY_AUDIT.md").write_text(
        "# Task 12 State Registry Audit\n\n"
        "**Status: TASK12_AUTHORITATIVE_STATE_REGISTRY_VERIFIED**\n\n"
        f"- Raw same-state records: `{records}`.\n"
        f"- Unique archived-state payload hashes: `{checked}`.\n"
        f"- Duplicate records by state hash: `{records - checked}`.\n"
        "- Identity types remain separate: full state payload, MuJoCo state blob, qpos, qvel, and branch initial state.\n"
        f"- Registry SHA-256: `{file_sha256(args.used_state_registry)}`.\n"
    )
    snapshot = root / "DEV_B2_USED_STATE_REGISTRY_SNAPSHOT.json"
    snapshot.write_bytes(args.used_state_registry.read_bytes())
    if file_sha256(snapshot) != file_sha256(args.used_state_registry):
        raise RuntimeError("Used-state registry snapshot hash drift")

    require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")
    configure_standard_libero(args.libero_root, root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

    cases: list[dict[str, Any]] = []
    seen = defaultdict(set)
    rejected = []
    factories = benchmark.get_benchmark_dict()
    index = 0
    for suite_name in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        suite = factories[suite_name]()
        for task_id in range(suite.n_tasks):
            source_states = np.asarray(suite.get_task_init_states(task_id))
            if len(source_states) < 50:
                raise RuntimeError(f"Task {suite_name}:{task_id} lacks untouched state 49")
            for init_state_id in range(DEV_B2_FIRST_STATE, 50):
                env_seed = DEV_B2_SEED + index * 100_000
                config = LiberoEnvConfig(
                    task=suite_name, task_ids=[task_id], observation_height=256, observation_width=256
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
                    env.reset(seed=env_seed)
                    identities = {
                        "initial_state_hash": array_sha256(np.asarray(source_states[init_state_id]).copy()),
                        "simulator_state_hash": array_sha256(np.asarray(env._env.get_sim_state()).copy()),
                        "qpos_qvel_hash": structured_hash(
                            {
                                "qpos": np.asarray(env._env.sim.data.qpos).copy(),
                                "qvel": np.asarray(env._env.sim.data.qvel).copy(),
                            }
                        ),
                    }
                    case = {
                        "case_id": f"dev-b2-{suite_name}-task{task_id:02d}-state{init_state_id:02d}",
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
                    reasons = [
                        f"duplicate_{field}" for field, value in identities.items() if value in seen[field]
                    ]
                    reasons.extend(item["dataset"] for item in overlap_rows([case], registry))
                    if reasons:
                        rejected.append(
                            {
                                "case_id": case["case_id"],
                                "reasons": sorted(set(reasons)),
                                "replacement_attempted": False,
                            }
                        )
                        (root / "DEV_B2_REJECTED_CANDIDATES.json").write_text(
                            json.dumps(rejected, indent=2) + "\n"
                        )
                        raise RuntimeError(f"Dev-B2 fixed candidate rejected: {rejected[-1]}")
                    for field, value in identities.items():
                        seen[field].add(value)
                    cases.append(case)
                    index += 1
                    if index % 20 == 0:
                        print(f"dev_b2_state={index}/{DEV_B2_CASES}", flush=True)
                finally:
                    env.close()
    manifest = {
        "schema_version": 1,
        "status": "TASK13_ATTEMPT002_DEV_B2_FROZEN",
        "phase": "development",
        "expected_case_count": DEV_B2_CASES,
        "case_count": len(cases),
        "task_count": 40,
        "cases_per_task": DEV_B2_STATES_PER_TASK,
        "first_state_index": DEV_B2_FIRST_STATE,
        "registered_arms": list(TASK13_ARMS),
        "base_model_sha256": base_hash,
        "snap_model_sha256": SELECTED_MODEL_SHA256,
        "used_state_registry_sha256": file_sha256(snapshot),
        "selection_rule": "all fixed untouched state indices 45-49; no replacement",
        "outcomes_accessed_before_freeze": False,
        "cases": cases,
    }
    validate_frozen_cases(cases, DEV_B2_CASES, DEV_B2_STATES_PER_TASK)
    manifest_path = root / "DEV_B2_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "DEV_B2_REJECTED_CANDIDATES.json").write_text("[]\n")
    (root / "DEV_B2_OVERLAP_AUDIT.md").write_text(
        "# Dev-B2 Prospective Overlap Audit\n\n"
        "**Status: DEV_B2_ZERO_OVERLAP_VERIFIED**\n\n"
        "- 200 untouched cases: 40 tasks x states 45-49.\n"
        "- old dev40 / formal100 / Confirmation1200 / Task12 / Attempt001 overlap: `0 / 0 / 0 / 0 / 0`.\n"
        "- Internal initial/simulator/qpos-qvel duplicates: `0 / 0 / 0`.\n"
        f"- Task12 unique archived states checked: `{checked}`.\n"
        "- Replacement attempted: `NO`.\n"
    )
    metrics_out = root / "TASK13_ATTEMPT002_FROZEN_METRICS.json"
    metrics_out.write_bytes(args.original_metrics.read_bytes())
    decision_out = root / "TASK13_ATTEMPT002_DECISION_RULE.md"
    decision_out.write_bytes(args.original_decision_rule.read_bytes())
    prereg = root / "TASK13_ATTEMPT002_PREREGISTRATION.md"
    prereg.write_text(
        "# Task 13 Recovery Attempt002 Preregistration\n\n"
        "**Status: TASK13_ATTEMPT002_PREREGISTERED / OUTCOMES UNSEEN**\n\n"
        "Attempt002 is the sole prospective primary replication. It uses the untouched fixed state block 45-49, Base-10 and original confirmed 20k Snap-1 only. H1/H2/H3, labels, action and transition metrics, h1/h3/h5/h10, bootstraps, task clustering, leave-one-task-out prediction, negative controls, and decision rules are byte-identical to the original pre-outcome Task 13 freeze. N=200 is fixed and cannot be adaptively expanded. Attempt001 remains descriptive-only.\n\n"
        f"- Git commit: `{git_value('rev-parse', 'HEAD')}`.\n"
        f"- Frozen unix ns: `{time.time_ns()}`.\n"
        "- New model training: `NO`.\n"
    )
    hashes = {
        "manifest_sha256": file_sha256(manifest_path),
        "registry_sha256": file_sha256(snapshot),
        "metrics_sha256": file_sha256(metrics_out),
        "decision_rule_sha256": file_sha256(decision_out),
        "preregistration_sha256": file_sha256(prereg),
    }
    (root / "TASK13_ATTEMPT002_FREEZE_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"status": "TASK13_ATTEMPT002_READY_TO_COMMIT", **hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
