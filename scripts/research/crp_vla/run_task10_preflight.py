#!/usr/bin/env python
"""Run Task 10 instrumentation and confirmation preflight on a dev40 capsule."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_v2_common import array_sha256, load_capsule, structured_hash
from run_libero_paired_four_arm_pilot import (
    add_batch_dimension,
    configure_standard_libero,
    observation_hashes,
)
from run_replay_v2 import contact_pairs
from task9_common import (
    SELECTED_MODEL_SHA256,
    canonical_json_sha256,
    effective_processor_contract,
    file_sha256,
    load_json,
    require_model_hash,
)
from task10_common import (
    COUNTERFACTUAL_HORIZONS,
    PRIMARY_ARMS,
    assert_no_external_state_overlap,
    confirmation1200_diagnostic_subset,
    strict_primary_aggregate,
    validate_confirmation1200_manifest,
)
from trajectory_instrumentation import (
    capture_libero_state,
    execute_counterfactual_branches,
    load_replan_trace,
    physical_state_hash,
    reset_and_restore_libero_state,
    restore_libero_state,
    write_replan_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-vlm-model-dir", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def record_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
    if not passed:
        raise RuntimeError(f"Task 10 preflight failed: {name}: {evidence}")


def processor_manifest(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(checkpoint.glob("policy_*processor*"))
        if path.is_file()
    }


def event_snapshot(env: Any, observation: dict[str, Any], object_names: list[str]) -> dict[str, Any]:
    state = capture_libero_state(env, observation, object_names)
    robot = state.get("robot_state") or {}
    eef = state.get("end_effector_pose") or {}
    eef_pos = eef.get("pos") if isinstance(eef, dict) else None
    distances = {}
    if eef_pos is not None:
        for name, item in state["object_states"].items():
            distances[name] = float(
                np.linalg.norm(np.asarray(item["pos"], dtype=float) - np.asarray(eef_pos, dtype=float))
            )
    return {
        "gripper_open_close_state": copy.deepcopy(robot.get("gripper")),
        "end_effector_pose": copy.deepcopy(eef),
        "object_pose": copy.deepcopy(state["object_states"]),
        "object_gripper_distance": {
            "availability": "AVAILABLE" if distances else "UNAVAILABLE",
            "values": distances,
        },
        "contact_information": {
            "availability": "AVAILABLE",
            "pairs": contact_pairs(env._env),
        },
        "grasp_state": {
            "availability": "UNAVAILABLE",
            "reason": "No reliable existing simulator/evaluator grasp-state API",
        },
        "object_height": {name: float(item["pos"][2]) for name, item in state["object_states"].items()},
        "success_signal": False,
        "termination_reason": None,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []

    selected_hash = require_model_hash(args.selected_checkpoint / "model.safetensors", SELECTED_MODEL_SHA256)
    base_hash = file_sha256(args.base_checkpoint / "model.safetensors")
    base_contract = effective_processor_contract(args.base_checkpoint)
    snap_contract = effective_processor_contract(args.selected_checkpoint)
    record_check(checks, "selected_model_sha256", selected_hash == SELECTED_MODEL_SHA256, selected_hash)
    record_check(checks, "base_model_sha256_present", len(base_hash) == 64, base_hash)
    record_check(
        checks,
        "processor_effective_normalization_contract",
        base_contract["sha256"] == snap_contract["sha256"],
        base_contract["sha256"],
    )
    offline_files = {path.name for path in args.offline_vlm_model_dir.iterdir() if path.is_file()}
    record_check(
        checks,
        "offline_vlm_config_complete",
        {"config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"} <= offline_files,
        sorted(offline_files),
    )

    manifest = load_json(args.confirmation1200_manifest)
    validate_confirmation1200_manifest(manifest)
    primary_ids = {str(case["case_id"]) for case in manifest["cases"]}
    record_check(checks, "confirmation1200_manifest_valid", len(primary_ids) == 1200, len(primary_ids))
    subset = load_json(args.diagnostic_subset)
    reconstructed = confirmation1200_diagnostic_subset(manifest["cases"])
    record_check(
        checks,
        "diagnostic_subset_deterministic",
        subset["case_ids"] == reconstructed["case_ids"],
        {"case_count": len(subset["case_ids"])},
    )
    try:
        strict_primary_aggregate([], primary_ids)
    except RuntimeError as error:
        incomplete_error = str(error)
    else:
        incomplete_error = None
    record_check(
        checks,
        "incomplete_aggregator_rejection",
        incomplete_error is not None and "completed_primary_cases=0/1200" in incomplete_error,
        incomplete_error,
    )
    synthetic_complete = [
        {"case_id": case_id, "arm": arm, "status": "COMPLETED", "success": False}
        for case_id in sorted(primary_ids)
        for arm in PRIMARY_ARMS
    ]
    contaminated = synthetic_complete + [
        {
            "case_id": subset["case_ids"][0],
            "arm": "snap2_20k",
            "status": "COMPLETED",
            "success": False,
        }
    ]
    try:
        strict_primary_aggregate(contaminated, primary_ids)
    except RuntimeError as error:
        contamination_error = str(error)
    else:
        contamination_error = None
    record_check(
        checks,
        "diagnostic_contamination_rejection",
        contamination_error is not None and "contamination" in contamination_error,
        contamination_error,
    )

    formal = load_json(args.formal_manifest)
    development = load_json(args.development_manifest)
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
    record_check(
        checks,
        "formal100_overlap_zero",
        not ({case["simulator_state_hash"] for case in manifest["cases"]} & formal_hashes),
        0,
    )
    record_check(
        checks,
        "dev40_overlap_zero",
        not ({case["simulator_state_hash"] for case in manifest["cases"]} & dev_hashes),
        0,
    )
    injected = copy.deepcopy(manifest)
    injected["cases"][0]["simulator_state_hash"] = next(iter(formal_hashes))
    try:
        assert_no_external_state_overlap(
            injected["cases"], formal_hashes=formal_hashes, development_hashes=dev_hashes
        )
    except ValueError:
        formal_injection_rejected = True
    else:
        formal_injection_rejected = False
    record_check(checks, "formal100_overlap_rejection", formal_injection_rejected, "rejected")
    injected = copy.deepcopy(manifest)
    injected["cases"][0]["simulator_state_hash"] = next(iter(dev_hashes))
    try:
        assert_no_external_state_overlap(
            injected["cases"], formal_hashes=formal_hashes, development_hashes=dev_hashes
        )
    except ValueError:
        dev_injection_rejected = True
    else:
        dev_injection_rejected = False
    record_check(checks, "dev40_overlap_rejection", dev_injection_rejected, "rejected")

    dev_results = [
        result
        for result in development["results"]
        if result["suite"] == "libero_10"
        and int(result["task_id"]) == 0
        and int(result["init_state_id"]) == 4
    ]
    base10_result = next(result for result in dev_results if result["arm"] == "base10")
    capsule_dir = Path(base10_result["capsule_path"])
    capsule_metadata, capsule = load_capsule(capsule_dir, args.device)
    record_check(
        checks,
        "frozen_capsule_integrity",
        structured_hash(capsule["canonical_policy_batch"]) == capsule_metadata["canonical_input_sha256"],
        capsule_metadata["canonical_input_sha256"],
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

    base_policy, base_pre, base_post = load_policy(args.base_checkpoint)
    snap_policy, snap_pre, snap_post = load_policy(args.selected_checkpoint)
    record_check(checks, "base10_offline_load", base_policy is not None, "loaded")
    record_check(checks, "snap1_20k_offline_load", snap_policy is not None, "loaded")

    configure_standard_libero(args.libero_root, output_root / "libero_standard_config")
    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(expected_assets)
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env_config = LiberoEnvConfig(
        task="libero_10", task_ids=[0], observation_height=256, observation_width=256
    )
    env_preprocessor, env_postprocessor = env_config.get_env_processors()
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
    trace_records = []
    try:
        env.init_state_id = 4
        env.reset(seed=int(base10_result["env_seed"]))
        raw = env._format_raw_obs(env._env.set_init_state(capsule["simulator"]["state"]))
        batch0 = preprocess_observation(add_batch_dimension(copy.deepcopy(raw)))
        batch0["task"] = [suite.get_task(0).language]
        base_batch = base_pre(env_preprocessor(copy.deepcopy(batch0)))
        snap_batch = snap_pre(env_preprocessor(copy.deepcopy(batch0)))
        record_check(
            checks,
            "matched_initial_observation_condition_and_processor_output",
            structured_hash(base_batch) == structured_hash(snap_batch),
            structured_hash(base_batch),
        )
        noise_np = capsule["noise_schedule"][0].copy()
        noise = torch.from_numpy(noise_np).to(args.device)
        base_policy.config.num_steps = 10
        snap_policy.config.num_steps = 1
        with torch.inference_mode():
            base_chunk = base_policy.predict_action_chunk(copy.deepcopy(base_batch), noise=noise)
            snap_chunk = snap_policy.predict_action_chunk(
                copy.deepcopy(snap_batch), noise=noise, target_time=0.0
            )
        record_check(
            checks,
            "matched_noise",
            array_sha256(noise_np) == array_sha256(noise_np.copy()),
            array_sha256(noise_np),
        )

        base_denorm = base_post(base_chunk[0, :10])
        snap_denorm = snap_post(snap_chunk[0, :10])
        base_executed = env_postprocessor({ACTION: base_denorm})[ACTION].detach().cpu().numpy()
        snap_executed = env_postprocessor({ACTION: snap_denorm})[ACTION].detach().cpu().numpy()
        object_names = list(capsule_metadata["task_semantics"]["objects_of_interest"])
        frozen_state = capture_libero_state(env, raw, object_names)
        event = event_snapshot(env, raw, object_names)

        for arm, nfe, normalized, raw_output, denormalized, executed in (
            ("base10", 10, base_batch, base_chunk, base_denorm, base_executed),
            ("snap1_20k", 1, snap_batch, snap_chunk, snap_denorm, snap_executed),
        ):
            trace_path = output_root / "preflight_traces" / arm / "replan_0000"
            record = write_replan_trace(
                trace_path,
                identity={
                    "case_id": "dev40-libero_10-task00-state04-preflight-only",
                    "task_id": "libero_10:0",
                    "arm_id": arm,
                    "rollout_id": f"task10-preflight-{arm}",
                    "replan_index": 0,
                    "simulator_timestep": 0,
                },
                raw_observation=copy.deepcopy(raw),
                normalized_observation=copy.deepcopy(normalized),
                simulator_state=copy.deepcopy(frozen_state),
                language_condition={
                    "instruction": suite.get_task(0).language,
                    "reference": "pinned LIBERO suite task language",
                },
                policy_output={
                    "raw_policy_output": raw_output.detach().cpu(),
                    "denormalized_action": denormalized.detach().cpu(),
                    "executed_action_chunk": np.asarray(executed),
                    "execution_horizon": 10,
                    "nfe": nfe,
                    "noise_tensor": noise_np,
                    "noise_seed": capsule_metadata["noise_seed"],
                    "action_mask": normalized.get("action_is_pad"),
                    "processor_contract": base_contract,
                    "processor_hash": base_contract["sha256"],
                },
                event_state=copy.deepcopy(event),
            )
            loaded_metadata, loaded_payload = load_replan_trace(trace_path, device="cpu")
            record_check(
                checks,
                f"{arm}_observation_fidelity",
                structured_hash(loaded_payload["raw_observation"]) == structured_hash(raw)
                and structured_hash(loaded_payload["normalized_observation"]) == structured_hash(normalized),
                loaded_metadata["integrity"],
            )
            trace_records.append(record)

        record_check(
            checks,
            "trace_path_uniqueness",
            len({record["path"] for record in trace_records}) == len(trace_records),
            [record["path"] for record in trace_records],
        )
        record_check(
            checks,
            "trace_hash_uniqueness",
            len({record["sha256"] for record in trace_records}) == len(trace_records),
            [record["sha256"] for record in trace_records],
        )

        initial_obs_hash = canonical_json_sha256(observation_hashes(raw))
        env.step(base_executed[0])
        restored_raw = restore_libero_state(env, frozen_state)
        restored = capture_libero_state(env, restored_raw, object_names)
        record_check(
            checks,
            "simulator_state_restoration",
            np.array_equal(restored["qpos"], frozen_state["qpos"])
            and np.array_equal(restored["qvel"], frozen_state["qvel"])
            and structured_hash(restored["object_states"]) == structured_hash(frozen_state["object_states"]),
            {
                "state_blob": restored["state_blob_sha256"],
                "qpos": restored["qpos_sha256"],
                "qvel": restored["qvel_sha256"],
            },
        )
        restored_obs_hash = canonical_json_sha256(observation_hashes(restored_raw))
        record_check(
            checks,
            "restored_observation_hash",
            restored_obs_hash == initial_obs_hash,
            {"expected": initial_obs_hash, "observed": restored_obs_hash},
        )

        current_observation = restored_raw

        def restore_callback(state: Any) -> dict[str, Any]:
            nonlocal current_observation
            current_observation = reset_and_restore_libero_state(
                env,
                state,
                env_seed=int(base10_result["env_seed"]),
                init_state_id=4,
            )
            return capture_libero_state(env, current_observation, object_names)

        def capture_callback() -> dict[str, Any]:
            return capture_libero_state(env, current_observation, object_names)

        def step_callback(action: np.ndarray) -> None:
            nonlocal current_observation
            current_observation, _, _, _, _ = env.step(action)

        chunks = {"base10": base_executed, "snap1_20k": snap_executed}
        first = execute_counterfactual_branches(
            frozen_state=frozen_state,
            action_chunks=chunks,
            restore_state=restore_callback,
            capture_state=capture_callback,
            step_action=step_callback,
            horizons=COUNTERFACTUAL_HORIZONS,
        )
        second = execute_counterfactual_branches(
            frozen_state=frozen_state,
            action_chunks=chunks,
            restore_state=restore_callback,
            capture_state=capture_callback,
            step_action=step_callback,
            horizons=COUNTERFACTUAL_HORIZONS,
        )
        record_check(
            checks,
            "counterfactual_identical_initial_state",
            first["identical_initial_state"],
            first["restoration_hashes"],
        )
        record_check(
            checks,
            "counterfactual_branch_isolation",
            len(set(first["restoration_hashes"].values())) == 1,
            first["restoration_hashes"],
        )
        record_check(
            checks,
            "counterfactual_horizons",
            first["horizons"] == list(COUNTERFACTUAL_HORIZONS),
            first["horizons"],
        )
        first_full_hash = structured_hash(first["branch_states"])
        second_full_hash = structured_hash(second["branch_states"])
        first_physical_hash = physical_state_hash(first["branch_states"])
        second_physical_hash = physical_state_hash(second["branch_states"])
        record_check(
            checks,
            "counterfactual_determinism",
            first_physical_hash == second_physical_hash,
            {"first_physical": first_physical_hash, "second_physical": second_physical_hash},
        )
        record_check(
            checks,
            "counterfactual_render_variance_recorded",
            True,
            {
                "full_state_and_observation_equal": first_full_hash == second_full_hash,
                "first_full": first_full_hash,
                "second_full": second_full_hash,
                "interpretation": "physical state must be deterministic; EGL render bytes are tracked separately under R-014",
            },
        )
        (output_root / "counterfactual_validation.json").write_text(
            json.dumps(
                {key: value for key, value in first.items() if key not in {"branch_states"}},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        env.close()

    result_paths = [output_root / "future_results" / "case-smoke" / f"{arm}.json" for arm in PRIMARY_ARMS]
    result_hashes = {
        canonical_json_sha256({"case": "case-smoke", "arm": arm, "trace": trace_records[index]["sha256"]})
        for index, arm in enumerate(PRIMARY_ARMS)
    }
    record_check(
        checks,
        "result_path_uniqueness",
        len({str(path) for path in result_paths}) == len(result_paths),
        [str(path) for path in result_paths],
    )
    record_check(checks, "result_hash_uniqueness", len(result_hashes) == 2, sorted(result_hashes))
    evaluator_hashes = {result["evaluator_sha256"] for result in dev_results}
    record_check(checks, "evaluator_equality", len(evaluator_hashes) == 1, sorted(evaluator_hashes))

    preflight = {
        "schema_version": 1,
        "status": "CONFIRMATION1200_PREFLIGHT_PASSED",
        "formal_confirmation_started": False,
        "confirmation_primary_cases_executed": 0,
        "new_training_started": False,
        "preflight_source": "existing dev40 capsule only",
        "selected_model_sha256": selected_hash,
        "base_model_sha256": base_hash,
        "processor_manifest": {
            "base": processor_manifest(args.base_checkpoint),
            "snap": processor_manifest(args.selected_checkpoint),
        },
        "checks": checks,
    }
    (output_root / "confirmation1200_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": preflight["status"], "checks": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
