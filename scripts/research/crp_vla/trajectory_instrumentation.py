#!/usr/bin/env python
"""Lossless per-replan tracing and isolated counterfactual execution.

The archive format reuses replay-v2's hash-verified NPZ capsule primitives.
Arrays and tensors are stored in full; hashes are integrity metadata rather
than substitutes for observations or simulator state.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from replay_v2_common import array_sha256, load_capsule, structured_hash, write_capsule
from task10_common import COUNTERFACTUAL_HORIZONS, transition_divergence

TRACE_SCHEMA_VERSION = 1

REQUIRED_IDENTITY = {
    "case_id",
    "task_id",
    "arm_id",
    "rollout_id",
    "replan_index",
    "simulator_timestep",
}

TRACE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "crp-vla-task10-replan-trace-v1",
    "title": "CRP-VLA Task 10 per-replan trace archive metadata",
    "type": "object",
    "required": [
        "schema_version",
        "record_type",
        "identity",
        "integrity",
        "availability",
        "payload_skeleton",
        "array_manifest",
    ],
    "properties": {
        "schema_version": {"const": TRACE_SCHEMA_VERSION},
        "record_type": {"const": "CRP_VLA_REPLAN_TRACE"},
        "identity": {
            "type": "object",
            "required": sorted(REQUIRED_IDENTITY),
            "properties": {
                "case_id": {"type": "string"},
                "task_id": {"type": ["string", "integer"]},
                "arm_id": {"type": "string"},
                "rollout_id": {"type": "string"},
                "replan_index": {"type": "integer", "minimum": 0},
                "simulator_timestep": {"type": "integer", "minimum": 0},
            },
        },
        "integrity": {"type": "object"},
        "availability": {"type": "object"},
        "payload_skeleton": {"type": "object"},
        "array_manifest": {"type": "object", "minProperties": 1},
    },
}


class ReplanTraceRecorder:
    """Stream complete replan records without retaining a rollout in memory."""

    def __init__(
        self,
        root: Path,
        *,
        identity_base: dict[str, Any],
        language_condition: dict[str, Any],
        processor_contract: dict[str, Any],
        processor_hash: str,
        execution_horizon: int,
        nfe: int,
        noise_seed: int,
    ) -> None:
        self.root = root
        self.identity_base = identity_base
        self.language_condition = language_condition
        self.processor_contract = processor_contract
        self.processor_hash = processor_hash
        self.execution_horizon = int(execution_horizon)
        self.nfe = int(nfe)
        self.noise_seed = int(noise_seed)
        self.pending: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def start(
        self,
        *,
        replan_index: int,
        simulator_timestep: int,
        raw_observation: dict[str, Any],
        normalized_observation: dict[str, Any],
        simulator_state: dict[str, Any],
        raw_policy_output: Any,
        denormalized_action: Any,
        noise_tensor: Any,
        action_mask: Any,
        event_state: dict[str, Any],
    ) -> None:
        if self.pending is not None:
            raise RuntimeError("Previous replan trace must be finished before starting another")
        self.pending = {
            "identity": {
                **self.identity_base,
                "replan_index": int(replan_index),
                "simulator_timestep": int(simulator_timestep),
            },
            "raw_observation": copy.deepcopy(raw_observation),
            "normalized_observation": copy.deepcopy(normalized_observation),
            "simulator_state": copy.deepcopy(simulator_state),
            "raw_policy_output": copy.deepcopy(raw_policy_output),
            "denormalized_action": copy.deepcopy(denormalized_action),
            "noise_tensor": copy.deepcopy(noise_tensor),
            "action_mask": copy.deepcopy(action_mask),
            "event_state_at_replan": copy.deepcopy(event_state),
            "executed_actions": [],
        }

    def append_executed_action(self, action: np.ndarray) -> None:
        if self.pending is None:
            raise RuntimeError("Cannot append an action without an active replan trace")
        self.pending["executed_actions"].append(np.asarray(action).copy())

    def finish(self, event_state_after_execution: dict[str, Any]) -> dict[str, Any] | None:
        if self.pending is None:
            return None
        pending = self.pending
        replan_index = int(pending["identity"]["replan_index"])
        path = self.root / f"replan_{replan_index:04d}"
        record = write_replan_trace(
            path,
            identity=pending["identity"],
            raw_observation=pending["raw_observation"],
            normalized_observation=pending["normalized_observation"],
            simulator_state=pending["simulator_state"],
            language_condition=self.language_condition,
            policy_output={
                "raw_policy_output": pending["raw_policy_output"],
                "denormalized_action": pending["denormalized_action"],
                "executed_action_chunk": np.asarray(pending["executed_actions"]),
                "execution_horizon": self.execution_horizon,
                "nfe": self.nfe,
                "noise_tensor": pending["noise_tensor"],
                "noise_seed": self.noise_seed,
                "action_mask": pending["action_mask"],
                "processor_contract": self.processor_contract,
                "processor_hash": self.processor_hash,
            },
            event_state={
                "at_replan": pending["event_state_at_replan"],
                "after_execution": copy.deepcopy(event_state_after_execution),
                "contact_information": event_state_after_execution.get(
                    "contact_information", {"availability": "UNAVAILABLE"}
                ),
                "grasp_state": event_state_after_execution.get(
                    "grasp_state", {"availability": "UNAVAILABLE"}
                ),
                "object_gripper_distance": event_state_after_execution.get(
                    "object_gripper_distance", {"availability": "UNAVAILABLE"}
                ),
            },
        )
        self.records.append(record)
        self.pending = None
        return record


def _copy_array(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


def _robot_state(observation: Mapping[str, Any] | None) -> tuple[Any, Any, Any]:
    state = (observation or {}).get("robot_state", {})
    if not isinstance(state, Mapping):
        return None, None, None
    eef = state.get("eef")
    gripper = state.get("gripper")
    return copy.deepcopy(state), copy.deepcopy(eef), copy.deepcopy(gripper)


def capture_libero_state(
    env: Any,
    observation: Mapping[str, Any] | None,
    object_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Capture the minimum complete LIBERO branch state plus objective events."""
    inner = getattr(env, "_env", env)
    if inner is None or not hasattr(inner, "sim") or not hasattr(inner, "get_sim_state"):
        raise ValueError("LIBERO simulator is unavailable")
    sim = inner.sim
    serialized = _copy_array(inner.get_sim_state())
    qpos = _copy_array(sim.data.qpos)
    qvel = _copy_array(sim.data.qvel)
    names = tuple(object_names or ())
    object_states: dict[str, dict[str, np.ndarray]] = {}
    for index in range(1, int(sim.model.nbody)):
        body_name = sim.model.body_id2name(index)
        if not body_name:
            continue
        if names and not any(token.lower() in body_name.lower() for token in names):
            continue
        object_states[str(body_name)] = {
            "pos": _copy_array(sim.data.body_xpos[index]),
            "quat": _copy_array(sim.data.body_xquat[index]),
        }
    robot_state, end_effector_pose, gripper_state = _robot_state(observation)
    return {
        "state_blob": serialized,
        "qpos": qpos,
        "qvel": qvel,
        "canonical_observation": copy.deepcopy(observation),
        "object_states": object_states,
        "robot_state": robot_state,
        "end_effector_pose": end_effector_pose,
        "gripper_state": gripper_state,
        "state_blob_sha256": array_sha256(serialized),
        "qpos_sha256": array_sha256(qpos),
        "qvel_sha256": array_sha256(qvel),
    }


def objective_event_snapshot(
    simulator_state: Mapping[str, Any],
    *,
    contacts: list[dict[str, Any]] | None,
    success: bool,
    termination_reason: str | None,
) -> dict[str, Any]:
    """Record available simulator facts and mark unsupported facts unavailable."""
    eef = simulator_state.get("end_effector_pose") or {}
    eef_pos = eef.get("pos") if isinstance(eef, Mapping) else None
    distances: dict[str, float] = {}
    if eef_pos is not None:
        for name, item in (simulator_state.get("object_states") or {}).items():
            if "pos" in item:
                distances[str(name)] = float(
                    np.linalg.norm(
                        np.asarray(item["pos"], dtype=np.float64) - np.asarray(eef_pos, dtype=np.float64)
                    )
                )
    return {
        "gripper_open_close_state": copy.deepcopy(simulator_state.get("gripper_state")),
        "end_effector_pose": copy.deepcopy(simulator_state.get("end_effector_pose")),
        "object_pose": copy.deepcopy(simulator_state.get("object_states", {})),
        "object_gripper_distance": {
            "availability": "AVAILABLE" if distances else "UNAVAILABLE",
            "values": distances,
        },
        "contact_information": {
            "availability": "AVAILABLE" if contacts is not None else "UNAVAILABLE",
            "pairs": copy.deepcopy(contacts),
        },
        "grasp_state": {
            "availability": "UNAVAILABLE",
            "reason": "No reliable existing simulator/evaluator grasp-state API",
        },
        "object_height": {
            str(name): float(np.asarray(item["pos"])[2])
            for name, item in (simulator_state.get("object_states") or {}).items()
            if "pos" in item
        },
        "success_signal": bool(success),
        "termination_reason": termination_reason,
    }


def restore_libero_state(env: Any, frozen_state: Mapping[str, Any]) -> Any:
    """Restore an exact serialized state and fail if qpos/qvel do not match."""
    inner = getattr(env, "_env", env)
    if inner is None or not hasattr(inner, "set_init_state"):
        raise ValueError("LIBERO simulator restoration API is unavailable")
    rerendered_observation = inner.set_init_state(_copy_array(frozen_state["state_blob"]))
    qpos = _copy_array(inner.sim.data.qpos)
    qvel = _copy_array(inner.sim.data.qvel)
    if not np.array_equal(qpos, np.asarray(frozen_state["qpos"])):
        raise RuntimeError("Exact qpos restoration failed")
    if not np.array_equal(qvel, np.asarray(frozen_state["qvel"])):
        raise RuntimeError("Exact qvel restoration failed")
    restored_blob = _copy_array(inner.get_sim_state())
    if array_sha256(restored_blob) != frozen_state["state_blob_sha256"]:
        raise RuntimeError("Exact simulator state-blob restoration failed")
    return copy.deepcopy(frozen_state.get("canonical_observation", rerendered_observation))


def _validate_trace_payload(identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    missing_identity = sorted(REQUIRED_IDENTITY - set(identity))
    if missing_identity:
        raise ValueError(f"Trace identity missing fields: {missing_identity}")
    required_payload = {
        "raw_observation",
        "normalized_observation",
        "simulator_state",
        "language_condition",
        "policy_output",
        "event_state",
    }
    missing_payload = sorted(required_payload - set(payload))
    if missing_payload:
        raise ValueError(f"Trace payload missing fields: {missing_payload}")
    simulator = payload["simulator_state"]
    for key in ("state_blob", "qpos", "qvel", "object_states", "robot_state", "gripper_state"):
        if key not in simulator:
            raise ValueError(f"Trace simulator state missing {key}")
    policy = payload["policy_output"]
    for key in (
        "raw_policy_output",
        "denormalized_action",
        "executed_action_chunk",
        "execution_horizon",
        "nfe",
        "noise_tensor",
        "noise_seed",
        "action_mask",
        "processor_contract",
        "processor_hash",
    ):
        if key not in policy:
            raise ValueError(f"Trace policy output missing {key}")


def write_replan_trace(
    path: Path,
    *,
    identity: dict[str, Any],
    raw_observation: dict[str, Any],
    normalized_observation: dict[str, Any],
    simulator_state: dict[str, Any],
    language_condition: dict[str, Any],
    policy_output: dict[str, Any],
    event_state: dict[str, Any],
) -> dict[str, Any]:
    """Write one immutable, hash-verified replan archive."""
    payload = {
        "raw_observation": raw_observation,
        "normalized_observation": normalized_observation,
        "simulator_state": simulator_state,
        "language_condition": language_condition,
        "policy_output": policy_output,
        "event_state": event_state,
    }
    _validate_trace_payload(identity, payload)
    integrity = {
        "raw_observation_sha256": structured_hash(raw_observation),
        "normalized_observation_sha256": structured_hash(normalized_observation),
        "simulator_state_sha256": structured_hash(simulator_state),
        "policy_output_sha256": structured_hash(policy_output),
        "complete_payload_sha256": structured_hash(payload),
    }
    availability = {
        "contact_information": event_state.get("contact_information", {}).get("availability", "UNAVAILABLE"),
        "grasp_state": event_state.get("grasp_state", {}).get("availability", "UNAVAILABLE"),
        "object_gripper_distance": event_state.get("object_gripper_distance", {}).get(
            "availability", "UNAVAILABLE"
        ),
    }
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "record_type": "CRP_VLA_REPLAN_TRACE",
        "identity": identity,
        "integrity": integrity,
        "availability": availability,
    }
    record = write_capsule(path, metadata, payload)
    record["replan_index"] = int(identity["replan_index"])
    record["integrity"] = integrity
    return record


def load_replan_trace(path: Path, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, payload = load_capsule(path, device=device)
    if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("Replan trace schema mismatch")
    if metadata.get("record_type") != "CRP_VLA_REPLAN_TRACE":
        raise ValueError("Unexpected replan trace record type")
    _validate_trace_payload(metadata["identity"], payload)
    checks = {
        "raw_observation_sha256": structured_hash(payload["raw_observation"]),
        "normalized_observation_sha256": structured_hash(payload["normalized_observation"]),
        "simulator_state_sha256": structured_hash(payload["simulator_state"]),
        "policy_output_sha256": structured_hash(payload["policy_output"]),
        "complete_payload_sha256": structured_hash(payload),
    }
    if checks != metadata.get("integrity"):
        raise ValueError("Replan trace structured integrity mismatch")
    return metadata, payload


def execute_counterfactual_branches(
    *,
    frozen_state: Mapping[str, Any],
    action_chunks: Mapping[str, np.ndarray],
    restore_state: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    capture_state: Callable[[], Mapping[str, Any]],
    step_action: Callable[[np.ndarray], Any],
    horizons: tuple[int, ...] = COUNTERFACTUAL_HORIZONS,
) -> dict[str, Any]:
    """Run every action branch from an identical restored state.

    The caller owns the simulator adapter.  Restoration is performed before
    every branch; state fingerprints are compared before any action is stepped.
    """
    if len(action_chunks) < 2:
        raise ValueError("Counterfactual analysis requires at least two branches")
    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or horizons[0] < 1:
        raise ValueError("Counterfactual horizons must be positive")
    maximum = horizons[-1]
    for branch, chunk in action_chunks.items():
        array = np.asarray(chunk)
        if array.ndim != 2 or len(array) < maximum:
            raise ValueError(f"Branch {branch} lacks actions through horizon {maximum}")

    initial_hash = structured_hash(dict(frozen_state))
    transitions: dict[str, dict[str, Any]] = {}
    restoration_hashes: dict[str, str] = {}
    for branch in sorted(action_chunks):
        restored = dict(restore_state(frozen_state))
        restored_hash = structured_hash(restored)
        restoration_hashes[branch] = restored_hash
        if restored_hash != initial_hash:
            raise RuntimeError(f"Counterfactual branch {branch} did not restore the identical initial state")
        branch_states: dict[str, Any] = {}
        for index, action in enumerate(np.asarray(action_chunks[branch])[:maximum], start=1):
            step_action(np.asarray(action).copy())
            if index in horizons:
                branch_states[str(index)] = copy.deepcopy(dict(capture_state()))
        transitions[branch] = branch_states

    if len(set(restoration_hashes.values())) != 1:
        raise RuntimeError("Counterfactual branch restoration hashes differ")
    branch_names = sorted(action_chunks)
    reference = branch_names[0]
    comparisons: dict[str, Any] = {}
    for branch in branch_names[1:]:
        comparison_id = f"{reference}_vs_{branch}"
        comparisons[comparison_id] = {
            str(horizon): transition_divergence(
                transitions[reference][str(horizon)],
                transitions[branch][str(horizon)],
            )
            for horizon in horizons
        }
    return {
        "status": "COUNTERFACTUAL_BRANCHES_COMPLETED",
        "initial_state_sha256": initial_hash,
        "restoration_hashes": restoration_hashes,
        "identical_initial_state": len(set(restoration_hashes.values())) == 1,
        "horizons": list(horizons),
        "branch_states": transitions,
        "transition_divergence": comparisons,
        "combined_weighted_scalar_used": False,
    }
