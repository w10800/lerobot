#!/usr/bin/env python
"""Generate one policy-free Task14R state shard and official-state references."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import array_sha256, structured_hash
from run_libero_paired_four_arm_pilot import configure_standard_libero, git_repository_value
from task14r_common import (
    TASK14R_CAPACITY_PROBE_COUNT,
    TASK14R_NAME,
)
from trajectory_instrumentation import capture_libero_state, physical_state_hash

SETTLE_STEPS = 10
MAX_PLACEMENT_REJECTIONS = 5_000
MAX_ABS_QVEL = 50.0
MAX_OBJECT_SETTLE_DISPLACEMENT = 0.20
MIN_CONTACT_DISTANCE = -0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--used-state-registry", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_bytes_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def finite_nested(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_nested(child) for child in value.values())
    if isinstance(value, list | tuple):
        return all(finite_nested(child) for child in value)
    if isinstance(value, np.ndarray | np.generic | int | float):
        try:
            return bool(np.isfinite(np.asarray(value)).all())
        except TypeError:
            return True
    return True


def canonical_quaternion(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    for component in quat:
        if abs(float(component)) > 1e-12:
            if component < 0:
                quat *= -1
            break
    return quat


def _body_pose(sim: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(sim.model.body_name2id(body_name))
    return (
        np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy(),
        canonical_quaternion(sim.data.body_xquat[body_id]),
    )


def object_poses(inner: Any) -> dict[str, dict[str, Any]]:
    task_env = inner.env
    output: dict[str, dict[str, Any]] = {}
    for group, mapping in (
        ("movable", task_env.objects_dict),
        ("fixture", task_env.fixtures_dict),
    ):
        for name, obj in sorted(mapping.items()):
            pos, quat = _body_pose(inner.sim, obj.root_body)
            output[f"{group}:{name}"] = {
                "group": group,
                "name": str(name),
                "root_body": str(obj.root_body),
                "pos": pos,
                "quat": quat,
            }
    if not output:
        raise RuntimeError("No LIBERO object poses were available")
    return output


def region_positions(inner: Any) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in sorted(inner.env.parsed_problem["regions"]):
        try:
            site_id = int(inner.sim.model.site_name2id(name))
        except (KeyError, ValueError):
            continue
        output[str(name)] = np.asarray(inner.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    if not output:
        raise RuntimeError("No BDDL target-region sites were available")
    return output


def contact_summary(inner: Any) -> dict[str, float | int]:
    count = int(inner.sim.data.ncon)
    distances = [float(inner.sim.data.contact[index].dist) for index in range(count)]
    return {
        "count": count,
        "minimum_distance": min(distances) if distances else 0.0,
    }


def predicate_results(inner: Any) -> dict[str, bool]:
    conditions = inner.env.parsed_problem["initial_state"]
    results = inner.evaluate_conditions(conditions)
    if len(results) != len(conditions):
        raise RuntimeError("Initial-predicate evaluator returned an incomplete result")
    return {str(key): bool(value) for key, value in sorted(results.items())}


def state_features(
    inner: Any,
    raw_observation: dict[str, Any],
    before_poses: dict[str, dict[str, Any]],
    after_predicates: dict[str, bool],
) -> dict[str, float]:
    features: dict[str, float] = {}
    joint_pos = np.asarray(raw_observation["robot0_joint_pos"], dtype=np.float64)
    eef_pos = np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float64)
    eef_quat = canonical_quaternion(raw_observation["robot0_eef_quat"])
    for index, value in enumerate(joint_pos):
        features[f"robot_joint_pos/j{index}"] = float(value)
    for axis, value in zip("xyz", eef_pos, strict=True):
        features[f"eef_position/{axis}"] = float(value)
    for index, value in enumerate(eef_quat):
        features[f"eef_orientation/q{index}"] = float(value)

    after_poses = object_poses(inner)
    movable_names = []
    for key, pose in sorted(after_poses.items()):
        safe = key.replace(":", "_")
        if pose["group"] == "movable":
            movable_names.append(key)
        for axis, value in zip("xyz", pose["pos"], strict=True):
            features[f"object_position/{safe}/{axis}"] = float(value)
        for index, value in enumerate(pose["quat"]):
            features[f"object_orientation/{safe}/q{index}"] = float(value)
        displacement = float(np.linalg.norm(np.asarray(pose["pos"]) - np.asarray(before_poses[key]["pos"])))
        features[f"settling_displacement/{safe}"] = displacement

    for left, right in itertools.combinations(sorted(movable_names), 2):
        distance = float(
            np.linalg.norm(np.asarray(after_poses[left]["pos"]) - np.asarray(after_poses[right]["pos"]))
        )
        features[f"object_pair_distance/{left}--{right}".replace(":", "_")] = distance
    for object_name in sorted(movable_names):
        for region_name, region_pos in sorted(region_positions(inner).items()):
            distance = float(np.linalg.norm(np.asarray(after_poses[object_name]["pos"]) - region_pos))
            key = f"target_region_distance/{object_name}--{region_name}".replace(":", "_")
            features[key] = distance

    contacts = contact_summary(inner)
    features["contact/count"] = float(contacts["count"])
    features["contact/minimum_distance"] = float(contacts["minimum_distance"])
    for predicate, value in sorted(after_predicates.items()):
        features[f"task_initial_predicate/{predicate}"] = float(value)
    if not features or not np.isfinite(np.asarray(list(features.values()))).all():
        raise RuntimeError("Non-finite or empty state feature vector")
    return features


def reset_from_distribution(env: Any, seed: int) -> tuple[dict[str, Any], int]:
    from robosuite.utils.errors import RandomizationError

    env._ensure_env()
    inner = env._env
    if inner is None:
        raise RuntimeError("LIBERO simulator failed to initialize")
    inner.seed(int(seed))
    rejections = 0
    while True:
        try:
            raw = inner.env.reset()
            break
        except RandomizationError as error:
            rejections += 1
            if rejections >= MAX_PLACEMENT_REJECTIONS:
                raise RuntimeError("LIBERO placement rejection limit reached") from error
    for robot in inner.robots:
        robot.controller.use_delta = True
    return raw, rejections


def settle_current_state(
    env: Any,
    raw_observation: dict[str, Any],
    object_names: list[str],
) -> dict[str, Any]:
    from lerobot.envs.libero import get_libero_dummy_action

    inner = env._env
    if inner is None:
        raise RuntimeError("LIBERO simulator is unavailable")
    raw_state = np.asarray(inner.get_sim_state()).copy()
    before_poses = object_poses(inner)
    predicates_before = predicate_results(inner)
    success_before = bool(inner.check_success())
    contacts_before = contact_summary(inner)
    observation = raw_observation
    for _ in range(SETTLE_STEPS):
        observation, _, _, _ = inner.step(get_libero_dummy_action())
    predicates_after = predicate_results(inner)
    success_after = bool(inner.check_success())
    contacts_after = contact_summary(inner)
    formatted = env._format_raw_obs(observation)
    captured = capture_libero_state(env, formatted, object_names)
    features = state_features(inner, observation, before_poses, predicates_after)
    max_displacement = max(
        value for key, value in features.items() if key.startswith("settling_displacement/")
    )
    max_abs_qvel = float(np.max(np.abs(np.asarray(inner.sim.data.qvel)), initial=0.0))
    return {
        "raw_state": raw_state,
        "raw_state_sha256": array_sha256(raw_state),
        "raw_state_bytes_sha256": raw_bytes_sha256(raw_state),
        "settled_state_blob_sha256": captured["state_blob_sha256"],
        "settled_physical_state_sha256": physical_state_hash(captured),
        "settled_render_sha256": structured_hash(formatted),
        "finite": finite_nested(raw_state) and finite_nested(raw_observation) and finite_nested(captured),
        "predicates_before": predicates_before,
        "predicates_after": predicates_after,
        "success_before": success_before,
        "success_after": success_after,
        "contacts_before": contacts_before,
        "contacts_after": contacts_after,
        "max_abs_qvel": max_abs_qvel,
        "max_object_settle_displacement": float(max_displacement),
        "features": features,
    }


def restore_and_settle(
    env: Any,
    raw_state: np.ndarray,
    *,
    reset_seed: int,
    object_names: list[str],
) -> dict[str, Any]:
    reset_from_distribution(env, reset_seed)
    if env._env is None:
        raise RuntimeError("LIBERO simulator is unavailable")
    raw_observation = env._env.set_init_state(np.asarray(raw_state).copy())
    restored = np.asarray(env._env.get_sim_state())
    if array_sha256(restored) != array_sha256(np.asarray(raw_state)):
        raise RuntimeError("Exact raw-state restoration failed")
    return settle_current_state(env, raw_observation, object_names)


def compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if key not in {"raw_state", "features"}}


def write_payload(path: Path, raw_state: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_state=np.ascontiguousarray(raw_state))
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before Task14R state generation")
    design = json.loads(args.seed_manifest.read_text())
    registry = json.loads(args.used_state_registry.read_text())
    if design.get("status") != "TASK14R_STATE_SEEDS_FROZEN":
        raise RuntimeError("Task14R state seed manifest is not frozen")
    if registry.get("status") != "USED_STATE_REGISTRY_COMPLETE":
        raise RuntimeError("Used-state registry is not complete")
    task_key = f"{args.suite}:{args.task_id}"
    task_spec = next((row for row in design["tasks"] if row["task_key"] == task_key), None)
    if task_spec is None:
        raise ValueError(f"Unknown frozen Task14R task: {task_key}")
    candidates = [row for row in design["candidates"] if row["task_key"] == task_key]
    probes = [row for row in design["capacity_probes"] if row["task_key"] == task_key]
    if len(candidates) != 20 or (args.task_id == 0 and len(probes) != TASK14R_CAPACITY_PROBE_COUNT):
        raise RuntimeError("Frozen Task14R shard cardinality drift")
    shard_root = args.output_root / args.suite / f"task{args.task_id:02d}"
    if shard_root.exists():
        raise FileExistsError(shard_root)
    configure_standard_libero(args.libero_root, shard_root / "libero_standard_config")

    import libero.libero as libero_module
    from libero.libero import benchmark

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv, get_task_init_states

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    config = LiberoEnvConfig(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=256,
        observation_width=256,
    )
    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        camera_name=config.camera_name,
        obs_type="pixels_agent_pos",
        render_mode=config.render_mode,
        observation_width=config.observation_width,
        observation_height=config.observation_height,
        init_states=False,
        num_steps_wait=SETTLE_STEPS,
        camera_name_mapping=config.camera_name_mapping,
        control_freq=config.fps,
        control_mode=config.control_mode,
        hard_reset=True,
    )
    used_hashes = {str(row["identity_value"]) for row in registry["entries"]}
    official_expected = next(
        row["official_raw_state_sha256"]
        for row in design["official_reference"]
        if row["task_key"] == task_key
    )
    official_hashes = set(map(str, official_expected))
    seen_hashes: set[str] = set()
    started_ns = time.time_ns()
    generated_rows = []
    official_rows = []
    try:
        env._ensure_env()
        if env._env is None:
            raise RuntimeError("LIBERO simulator failed to initialize")
        object_names = sorted(env._env.env.objects_dict)
        if file_sha256(Path(env._task_bddl_file)) != task_spec["bddl_sha256"]:
            raise RuntimeError("Frozen BDDL hash mismatch")

        official_states = get_task_init_states(suite, args.task_id)
        if len(official_states) != 50:
            raise RuntimeError("Expected exactly 50 official reference states")
        observed_official_hashes = [array_sha256(np.asarray(state)) for state in official_states]
        if observed_official_hashes != list(official_expected):
            raise RuntimeError("Official reference state hashes drifted after seed freeze")
        for index, state in enumerate(official_states):
            trial = restore_and_settle(
                env,
                np.asarray(state),
                reset_seed=10_000_000 + index,
                object_names=object_names,
            )
            official_rows.append(
                {
                    "state_id": f"official-{args.suite}-task{args.task_id:02d}-state{index:02d}",
                    "task_key": task_key,
                    "official_state_index": index,
                    "raw_state_sha256": trial["raw_state_sha256"],
                    "settled_physical_state_sha256": trial["settled_physical_state_sha256"],
                    "features": trial["features"],
                }
            )

        for spec in [*candidates, *probes]:
            seed = int(spec["seed"]["numpy_seed"])
            raw_observation, placement_rejections = reset_from_distribution(env, seed)
            baseline = settle_current_state(env, raw_observation, object_names)
            raw_state = np.asarray(baseline["raw_state"]).copy()
            trial_a = restore_and_settle(env, raw_state, reset_seed=seed, object_names=object_names)
            trial_b = restore_and_settle(env, raw_state, reset_seed=seed, object_names=object_names)
            identity_hashes = {
                baseline["raw_state_sha256"],
                baseline["raw_state_bytes_sha256"],
                baseline["settled_state_blob_sha256"],
                baseline["settled_physical_state_sha256"],
            }
            duplicate = bool(identity_hashes & (used_hashes | official_hashes | seen_hashes))
            checks = {
                "simulator_loadable": True,
                "finite_observation": bool(baseline["finite"]),
                "legal_initial_predicates": all(baseline["predicates_before"].values())
                and all(baseline["predicates_after"].values()),
                "not_initially_successful": not baseline["success_before"] and not baseline["success_after"],
                "no_severe_penetration_or_explosion": min(
                    float(baseline["contacts_before"]["minimum_distance"]),
                    float(baseline["contacts_after"]["minimum_distance"]),
                )
                >= MIN_CONTACT_DISTANCE
                and float(baseline["max_abs_qvel"]) <= MAX_ABS_QVEL
                and float(baseline["max_object_settle_displacement"]) <= MAX_OBJECT_SETTLE_DISPLACEMENT,
                "reset_reproducible": len(
                    {
                        baseline["settled_physical_state_sha256"],
                        trial_a["settled_physical_state_sha256"],
                        trial_b["settled_physical_state_sha256"],
                    }
                )
                == 1,
                "zero_historical_overlap": not duplicate,
                "scene_task_match": True,
            }
            accepted = all(checks.values())
            payload = write_payload(
                shard_root / "payloads" / f"{spec['state_id']}.npz",
                raw_state,
            )
            row = {
                **spec,
                "task_key": task_key,
                "policy_query_count": 0,
                "outcomes_accessed": False,
                "placement_rejection_count": placement_rejections,
                "raw_state_sha256": baseline["raw_state_sha256"],
                "raw_state_bytes_sha256": baseline["raw_state_bytes_sha256"],
                "settled_state_blob_sha256": baseline["settled_state_blob_sha256"],
                "settled_physical_state_sha256": baseline["settled_physical_state_sha256"],
                "settled_render_sha256": baseline["settled_render_sha256"],
                "features": baseline["features"],
                "acceptance_checks": checks,
                "accepted": accepted,
                "payload": payload,
                "baseline": compact_trial(baseline),
                "reproducibility_trials": [compact_trial(trial_a), compact_trial(trial_b)],
            }
            generated_rows.append(row)
            seen_hashes.update(identity_hashes)
            print(
                f"task={task_key} state={spec['state_id']} accepted={int(accepted)}",
                flush=True,
            )
    finally:
        env.close()

    candidate_rows = [row for row in generated_rows if "candidate_index" in row]
    probe_rows = [row for row in generated_rows if "probe_index" in row]
    manifest = {
        "schema_version": "task14r.state_shard.v1",
        "task_name": TASK14R_NAME,
        "phase": "PHASE_B_POLICY_INDEPENDENT_STATE_GENERATION",
        "status": "TASK14R_STATE_SHARD_COMPLETE",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "libero_commit": git_repository_value(args.libero_root, "rev-parse", "HEAD"),
        "task_key": task_key,
        "suite": args.suite,
        "task_id": args.task_id,
        "policy_query_count": 0,
        "outcomes_accessed": False,
        "formal_outcome_rollout_count": 0,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "candidate_count": len(candidate_rows),
        "accepted_candidate_count": sum(bool(row["accepted"]) for row in candidate_rows),
        "capacity_probe_count": len(probe_rows),
        "accepted_capacity_probe_count": sum(bool(row["accepted"]) for row in probe_rows),
        "official_reference_count": len(official_rows),
        "candidates": candidate_rows,
        "capacity_probes": probe_rows,
        "official_references": official_rows,
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "task": task_key}, sort_keys=True))


if __name__ == "__main__":
    main()
