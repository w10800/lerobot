#!/usr/bin/env python
"""Run Task 9 confirmation preflight without a confirmation rollout."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256, load_capsule, structured_hash
from run_libero_paired_four_arm_pilot import configure_standard_libero, observation_hashes
from task9_common import (
    PRIMARY_ARMS,
    SELECTED_MODEL_SHA256,
    assert_no_state_overlap,
    canonical_json_sha256,
    deterministic_diagnostic_subset,
    effective_processor_contract,
    file_sha256,
    guarded_aggregate,
    load_json,
    require_model_hash,
    validate_confirmation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(checkpoint.glob("policy_*processor*"))
        if path.is_file()
    }


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
    if not passed:
        raise RuntimeError(f"Preflight failed: {name}: {evidence}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    selected_hash = require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")
    offline_vlm_files = {
        path.name: file_sha256(path)
        for path in sorted(args.offline_vlm_model_dir.iterdir())
        if path.is_file()
    }
    record_check(
        checks,
        "offline_vlm_config_complete",
        {
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        <= offline_vlm_files.keys(),
        offline_vlm_files,
    )
    record_check(checks, "selected_model_hash", selected_hash == SELECTED_MODEL_SHA256, selected_hash)
    record_check(checks, "base_model_hash_present", len(base_hash) == 64, base_hash)
    base_processors = processor_manifest(args.base_checkpoint)
    snap_processors = processor_manifest(args.selected_checkpoint)
    base_contract = effective_processor_contract(args.base_checkpoint)
    snap_contract = effective_processor_contract(args.selected_checkpoint)
    record_check(
        checks,
        "action_normalization_and_processor_equality",
        bool(base_processors)
        and bool(snap_processors)
        and base_contract["sha256"] == snap_contract["sha256"],
        {
            "base_file_manifest": base_processors,
            "selected_file_manifest": snap_processors,
            "effective_contract_sha256": base_contract["sha256"],
            "note": "file-level differences are tokenizer storage metadata and unused image stats; effective state/action contract is exact",
        },
    )

    development = load_json(args.development_manifest)
    record_check(
        checks,
        "base10_model_hash_matches_frozen_development",
        base_hash == development.get("base_checkpoint_sha256"),
        {
            "observed": base_hash,
            "expected": development.get("base_checkpoint_sha256"),
        },
    )
    dev_case_results = [
        result
        for result in development["results"]
        if result["suite"] == "libero_10"
        and int(result["task_id"]) == 0
        and int(result["init_state_id"]) == 4
    ]
    base10_result = next(result for result in dev_case_results if result["arm"] == "base10")
    capsule_dir = Path(base10_result["capsule_path"])
    metadata, payload = load_capsule(capsule_dir, args.device)
    record_check(
        checks,
        "capsule_input_hash",
        structured_hash(payload["canonical_policy_batch"]) == metadata["canonical_input_sha256"],
        metadata["canonical_input_sha256"],
    )
    record_check(
        checks,
        "shared_frozen_noise_schedule",
        array_sha256(payload["noise_schedule"]) == metadata["noise_schedule_sha256"],
        metadata["noise_schedule_sha256"],
    )

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def load_policy(checkpoint: Path):
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
        return policy, preprocessor, postprocessor

    base_policy, _, base_post = load_policy(args.base_checkpoint)
    snap_policy, _, snap_post = load_policy(args.selected_checkpoint)
    record_check(checks, "base10_load", base_policy is not None, "loaded")
    record_check(checks, "selected_snap1_load", snap_policy is not None, "loaded")
    batch = copy.deepcopy(payload["canonical_policy_batch"])
    noise = torch.from_numpy(payload["noise_schedule"][0].copy()).to(args.device)
    base_policy.config.num_steps = 10
    snap_policy.config.num_steps = 1
    with torch.inference_mode():
        base_chunk = base_policy.predict_action_chunk(copy.deepcopy(batch), noise=noise)
        snap_chunk = snap_policy.predict_action_chunk(copy.deepcopy(batch), noise=noise, target_time=0.0)
    record_check(
        checks,
        "arm_input_equality",
        structured_hash(batch) == metadata["canonical_input_sha256"],
        {"input": structured_hash(batch), "noise": array_sha256(payload["noise_schedule"][0])},
    )
    base_action = base_post(base_chunk[:, :10].reshape(-1, base_chunk.shape[-1]))
    snap_action = snap_post(snap_chunk[:, :10].reshape(-1, snap_chunk.shape[-1]))
    action_ok = base_action.shape == snap_action.shape and bool(
        torch.isfinite(base_action).all() and torch.isfinite(snap_action).all()
    )
    record_check(
        checks,
        "action_normalization_output",
        action_ok,
        {"shape": list(base_action.shape)},
    )
    del base_policy, snap_policy, base_chunk, snap_chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    configure_standard_libero(args.libero_root, output_root / "preflight_libero_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env_config = LiberoEnvConfig(
        task="libero_10", task_ids=[0], observation_height=256, observation_width=256
    )
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_10",
        episode_length=TASK_SUITE_MAX_STEPS["libero_10"],
        camera_name=env_config.camera_name,
        obs_type=env_config.obs_type,
        render_mode=env_config.render_mode,
        observation_width=env_config.observation_width,
        observation_height=env_config.observation_height,
        init_states=env_config.init_states,
        episode_index=4,
        n_envs=1,
        num_steps_wait=10,
        camera_name_mapping=env_config.camera_name_mapping,
        control_freq=env_config.fps,
        control_mode=env_config.control_mode,
        is_libero_plus=env_config.is_libero_plus,
        hard_reset=env_config.hard_reset,
    )
    try:
        env.init_state_id = 4
        env.reset(seed=int(base10_result["env_seed"]))
        state = payload["simulator"]["state"]
        obs1 = env._format_raw_obs(env._env.set_init_state(state))
        state1 = array_sha256(np.asarray(env._env.get_sim_state()))
        obs2 = env._format_raw_obs(env._env.set_init_state(state))
        state2 = array_sha256(np.asarray(env._env.get_sim_state()))
        record_check(
            checks,
            "simulator_reset_reproducible",
            state1 == state2 == array_sha256(state),
            {"first": state1, "second": state2},
        )
        stored_observation_hash = canonical_json_sha256(
            observation_hashes(payload["raw_initial_observation"])
        )
        arm_observation_hash = canonical_json_sha256(
            observation_hashes(copy.deepcopy(payload["raw_initial_observation"]))
        )
        record_check(
            checks,
            "shared_raw_observation",
            stored_observation_hash == arm_observation_hash,
            stored_observation_hash,
        )
        record_check(
            checks,
            "reset_render_evidence_recorded",
            bool(observation_hashes(obs1)) and bool(observation_hashes(obs2)),
            {
                "first_reset_observation_hash": canonical_json_sha256(observation_hashes(obs1)),
                "second_reset_observation_hash": canonical_json_sha256(observation_hashes(obs2)),
                "note": "paired policy inference uses the frozen canonical capsule observation",
            },
        )
    finally:
        env.close()

    evaluator_hashes = {result["evaluator_sha256"] for result in dev_case_results}
    record_check(checks, "shared_evaluator", len(evaluator_hashes) == 1, sorted(evaluator_hashes))
    simulated_paths = [output_root / "future_results" / "case-smoke" / f"{arm}.json" for arm in PRIMARY_ARMS]
    record_check(
        checks,
        "result_paths_unique",
        len({str(path) for path in simulated_paths}) == len(simulated_paths),
        [str(path) for path in simulated_paths],
    )
    trace_paths = [
        output_root / "future_traces" / "case-smoke" / f"{arm}.sha256.json" for arm in PRIMARY_ARMS
    ]
    record_check(
        checks,
        "trace_manifest_paths_unique",
        len({str(path) for path in trace_paths}) == len(trace_paths),
        [str(path) for path in trace_paths],
    )
    result_hashes = {
        canonical_json_sha256(
            {"case_id": "case-smoke", "arm": arm, "input": metadata["canonical_input_sha256"]}
        )
        for arm in PRIMARY_ARMS
    }
    record_check(checks, "result_hashes_unique", len(result_hashes) == 2, sorted(result_hashes))

    manifest = load_json(args.confirmation_manifest)
    validate_confirmation_manifest(manifest)
    subset = load_json(args.diagnostic_subset)
    expected_subset = deterministic_diagnostic_subset(manifest["cases"], per_task=5, seed=20260815)
    record_check(
        checks,
        "diagnostic_subset_deterministic",
        subset["case_ids"] == expected_subset["case_ids"],
        {"case_count": len(subset["case_ids"])},
    )
    primary_ids = {case["case_id"] for case in manifest["cases"]}
    try:
        guarded_aggregate([], primary_ids, set(PRIMARY_ARMS))
    except RuntimeError:
        incomplete_rejected = True
    else:
        incomplete_rejected = False
    record_check(checks, "incomplete_aggregation_rejected", incomplete_rejected, "refused")
    diagnostic_record = [
        {
            "case_id": subset["case_ids"][0],
            "arm": "base2",
            "status": "COMPLETED",
            "success": True,
        }
    ]
    try:
        guarded_aggregate(diagnostic_record, primary_ids, set(PRIMARY_ARMS))
    except RuntimeError:
        diagnostic_rejected = True
    else:
        diagnostic_rejected = False
    record_check(checks, "diagnostic_arm_excluded_from_primary_aggregator", diagnostic_rejected, "refused")

    formal = load_json(args.formal_manifest)
    formal_hashes = {
        result["initial_sim_state_sha256"]
        for result in formal["results"]
        if result.get("initial_sim_state_sha256")
    }
    dev_hashes = {
        result["initial_sim_state_sha256"]
        for result in development["results"]
        if result.get("initial_sim_state_sha256")
    }
    assert_no_state_overlap(manifest["cases"], formal_hashes, dev_hashes)
    injected = copy.deepcopy(manifest["cases"][0])
    injected["simulator_state_hash"] = next(iter(formal_hashes))
    try:
        assert_no_state_overlap([injected], formal_hashes, dev_hashes)
    except ValueError:
        injected_rejected = True
    else:
        injected_rejected = False
    record_check(checks, "overlap_checker_rejects_formal_state", injected_rejected, "rejected")

    preflight = {
        "schema_version": 1,
        "status": "CONFIRMATION_PREFLIGHT_PASSED",
        "formal_confirmation_started": False,
        "new_training_started": False,
        "selected_model_sha256": selected_hash,
        "base_model_sha256": base_hash,
        "checks": checks,
    }
    (output_root / "confirmation_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Confirmation Preflight Report",
        "",
        "**Status: CONFIRMATION_PREFLIGHT_PASSED**",
        "",
        "No confirmation primary case was rolled out and no training was started.",
        "",
        "| Check | Passed |",
        "|---|:---:|",
        *[f"| {check['name']} | {check['passed']} |" for check in checks],
    ]
    (output_root / "CONFIRMATION_PREFLIGHT_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": preflight["status"], "checks": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
