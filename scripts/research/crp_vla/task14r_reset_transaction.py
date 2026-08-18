#!/usr/bin/env python
"""Complete-state reset transactions for the policy-free Task14R R0 gate.

The transaction deliberately never calls ``env.reset()`` after the canonical
case has been captured.  All repeated probes reuse one compiled MuJoCo model
and one renderer context, restore MuJoCo's complete integration state, and
restore the Python-side controller, gripper, buffer, wrapper, and observable
state explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .replay_v2_common import array_sha256, structured_hash
except ImportError:  # Direct script execution adds this directory to sys.path.
    from replay_v2_common import array_sha256, structured_hash

TASK14R_RESET_RECOVERY_NAME = "TASK14R_RESET_TRANSACTION_RECOVERY"
TASK14R_R0_STATUS_FROZEN = "TASK14R_R0_PROTOCOL_FROZEN"
TASK14R_R0_STATUS_PASSED = "TASK14R_R0_RESTORE_TRANSACTION_PASSED"
TASK14R_R0_STATUS_FAILED = "TASK14R_R0_RESTORE_TRANSACTION_FAILED"
TASK14R_R0_EVIDENCE_LABELS = (
    "ENGINEERING_ONLY",
    "RESET_DIAGNOSTIC_CONDITIONED",
    "NON_PRIMARY",
)
TASK14R_R0_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK14R_R0_TASKS_PER_SUITE = 10
TASK14R_R0_RESTORE_REPEATS = 3
TASK14R_R0_INITIAL_STATE_ID = 0
TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT = "20757544607413c6ab1a3458092ceec6bc1fc85d"
TASK14R_R0_SCHEMA_VERSION = "task14r.reset_transaction.protocol.v2"
TASK14R_R0_SEED_NAMESPACE = "CRP-VLA-TASK14R-R0-RESET-TRANSACTION-V1"
TASK14R_R0_LIBERO_COMMIT = "8460457bfca6e0ef2e856bc104e2c60b023ef2a7"
TASK14R_R0_RENDERER_BACKEND = "egl"
TASK14R_R0_RENDERER_OFFSAMPLES = 0
TASK14R_R0_PROBE_STEPS_PER_TRAJECTORY = 15
TASK14R_R0_EXPECTED_TASK_COUNT = 40
TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT = 120
TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT = 120
TASK14R_R0_EXPECTED_PROBE_STEP_COUNT = 1800
TASK14R_R0_SOURCE_FILES = (
    "scripts/research/crp_vla/run_task14r_r0.py",
    "scripts/research/crp_vla/task14r_reset_transaction.py",
    "scripts/research/crp_vla/launch_task14r_r0.sh",
)
TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY = (
    "R0 pass qualifies only the no-MSAA complete-state transaction.",
    "It does not authorize policy execution or establish equivalence to the standard LIBERO renderer.",
    "A separate policy-free renderer-shift audit is required before R1.",
)

# Fifteen fixed, policy-free OSC commands.  Every continuous action dimension
# is exercised independently in both directions at amplitude 0.05.  Gripper
# closed is -1.0 and open is 1.0 under the active Panda controller contract.
TASK14R_R0_PROBE_ACTIONS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, -0.05, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.05, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, -0.05, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, -0.05, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.05, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.0, -0.05, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -0.05, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
)

PROBE_TRACE_COMPARISON_FIELDS = (
    "step",
    "action",
    "integration_state_sha256",
    "python_state_sha256",
    "observation_sha256",
    "contact_state_sha256",
    "terminated",
    "truncated",
    "success_predicate",
)


class Task14RStageError(RuntimeError):
    """A fail-closed error whose transaction stage is safe to audit."""

    def __init__(self, failure_stage: str, message: str) -> None:
        super().__init__(message)
        self.failure_stage = str(failure_stage)


_ROBOT_REFERENCE_FIELDS = {"sim", "robot_model", "controller", "gripper", "controller_config"}
_CONTROLLER_REFERENCE_FIELDS = {"sim", "interpolator_pos", "interpolator_ori"}
_OBSERVABLE_STATE_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
    "_enabled",
    "_active",
    "_sampling_timestep",
    "_is_number",
    "_data_shape",
)
_CORE_WRAPPER_FIELDS = ("cur_time", "timestep", "done", "deterministic_reset", "_action_dim")
_CONTROLLER_DERIVED_FIELDS = (
    "ee_pos",
    "ee_ori_mat",
    "ee_pos_vel",
    "ee_ori_vel",
    "joint_pos",
    "joint_vel",
    "J_pos",
    "J_ori",
    "J_full",
    "mass_matrix",
)


def _plain_copy(value: Any, *, path: str) -> Any:
    """Copy only deterministic capsule values and reject opaque Python state."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return copy.deepcopy(value)
    if isinstance(value, Mapping):
        return {
            str(key): _plain_copy(child, path=f"{path}.{key}")
            for key, child in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain_copy(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    raise TypeError(f"Unsupported complete-state value at {path}: {type(value)!r}")


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:160]}


def plain_state_mismatches(expected: Any, actual: Any, *, path: str = "root") -> list[dict[str, Any]]:
    """Return exact, path-localized mismatches for fail-closed diagnostics."""
    rows: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, current: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = {str(key) for key in left}
            right_keys = {str(key) for key in right}
            if left_keys != right_keys:
                rows.append(
                    {
                        "path": current,
                        "reason": "mapping_keys",
                        "expected_only": sorted(left_keys - right_keys),
                        "actual_only": sorted(right_keys - left_keys),
                    }
                )
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], f"{current}.{key}")
            return
        if (
            isinstance(left, Sequence)
            and isinstance(right, Sequence)
            and not isinstance(left, str | bytes | bytearray)
            and not isinstance(right, str | bytes | bytearray)
            and not isinstance(left, np.ndarray)
            and not isinstance(right, np.ndarray)
        ):
            if len(left) != len(right):
                rows.append(
                    {
                        "path": current,
                        "reason": "sequence_length",
                        "expected": len(left),
                        "actual": len(right),
                    }
                )
            for index, (left_child, right_child) in enumerate(zip(left, right, strict=False)):
                visit(left_child, right_child, f"{current}[{index}]")
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            if not np.array_equal(left_array, right_array):
                row = {
                    "path": current,
                    "reason": "array_value",
                    "expected": _value_summary(left_array),
                    "actual": _value_summary(right_array),
                }
                if left_array.shape == right_array.shape and left_array.dtype.kind in "biufc":
                    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
                    row["changed_count"] = int(np.count_nonzero(difference))
                    row["maximum_absolute_difference"] = float(np.max(difference, initial=0.0))
                rows.append(row)
            return
        if type(left) is not type(right) or left != right:
            rows.append(
                {
                    "path": current,
                    "reason": "scalar_value",
                    "expected": _value_summary(left),
                    "actual": _value_summary(right),
                }
            )

    visit(expected, actual, path)
    return rows


def _snapshot_attributes(obj: Any, *, ignored: set[str], path: str) -> dict[str, Any]:
    values = {}
    for name, value in sorted(vars(obj).items()):
        if name in ignored:
            continue
        values[name] = _plain_copy(value, path=f"{path}.{name}")
    return values


def _restore_attributes(obj: Any, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        setattr(obj, str(name), _plain_copy(value, path=str(name)))


def _is_buffer(value: Any) -> bool:
    return value.__class__.__module__ == "robosuite.utils.buffers"


def task14r_r0_seed(suite: str, task_id: int) -> int:
    identity = f"{TASK14R_R0_SEED_NAMESPACE}|{suite}|{int(task_id)}"
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _validate_probe_actions(actions: Any) -> None:
    if not isinstance(actions, Sequence) or isinstance(actions, str | bytes | bytearray):
        raise ValueError("Task14R R0 probe_actions must be a sequence")
    if len(actions) != TASK14R_R0_PROBE_STEPS_PER_TRAJECTORY:
        raise ValueError("Task14R R0 probe count must be exactly 15")
    matrix = np.asarray(actions, dtype=np.float64)
    if matrix.shape != (TASK14R_R0_PROBE_STEPS_PER_TRAJECTORY, 7):
        raise ValueError("Task14R R0 action shape must be (15, 7)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Task14R R0 probe actions must be finite")
    if np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise ValueError("Task14R R0 probe actions must remain in [-1, 1]")
    for dimension in range(6):
        if not np.any(matrix[:, dimension] > 0.0):
            raise ValueError(f"Task14R R0 continuous dimension {dimension} lacks positive excitation")
        if not np.any(matrix[:, dimension] < 0.0):
            raise ValueError(f"Task14R R0 continuous dimension {dimension} lacks negative excitation")
        nonzero = np.abs(matrix[:, dimension][matrix[:, dimension] != 0.0])
        if not np.all(nonzero == 0.05):
            raise ValueError(f"Task14R R0 continuous dimension {dimension} amplitude drift")
    if set(matrix[:, 6].tolist()) != {-1.0, 1.0}:
        raise ValueError("Task14R R0 gripper must include both -1.0 and 1.0")


def build_task14r_r0_protocol(
    *, implementation_parent_commit: str, source_files_sha256: Mapping[str, str]
) -> dict[str, Any]:
    tasks = [
        {
            "suite": suite,
            "task_id": task_id,
            "init_state_id": TASK14R_R0_INITIAL_STATE_ID,
            "env_seed": task14r_r0_seed(suite, task_id),
            "case_id": f"r0-{suite}-task{task_id:02d}-state{TASK14R_R0_INITIAL_STATE_ID:02d}",
        }
        for suite in TASK14R_R0_SUITES
        for task_id in range(TASK14R_R0_TASKS_PER_SUITE)
    ]
    return {
        "schema_version": TASK14R_R0_SCHEMA_VERSION,
        "task_name": TASK14R_RESET_RECOVERY_NAME,
        "phase": "R0_COMPLETE_STATE_TRANSACTION_QUALIFICATION",
        "status": TASK14R_R0_STATUS_FROZEN,
        "evidence_labels": list(TASK14R_R0_EVIDENCE_LABELS),
        "implementation_parent_commit": str(implementation_parent_commit),
        "libero_commit": TASK14R_R0_LIBERO_COMMIT,
        "renderer_contract": {
            "backend": TASK14R_R0_RENDERER_BACKEND,
            "offsamples": TASK14R_R0_RENDERER_OFFSAMPLES,
            "pixel_gate": "EXACT",
        },
        "renderer_qualification_boundary": list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY),
        "source_files_sha256": dict(sorted(source_files_sha256.items())),
        "task_count": len(tasks),
        "restore_repeats_per_task": TASK14R_R0_RESTORE_REPEATS,
        "probe_steps_per_trajectory": TASK14R_R0_PROBE_STEPS_PER_TRAJECTORY,
        "expected_task_count": TASK14R_R0_EXPECTED_TASK_COUNT,
        "expected_restore_transaction_count": TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "expected_probe_trajectory_count": TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "expected_probe_step_count": TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
        "probe_actions": [list(row) for row in TASK14R_R0_PROBE_ACTIONS],
        "tasks": tasks,
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
        "permitted_terminal_statuses": [TASK14R_R0_STATUS_PASSED, TASK14R_R0_STATUS_FAILED],
    }


def validate_task14r_r0_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_implementation_parent_commit: str,
    expected_source_files_sha256: Mapping[str, str],
) -> None:
    if protocol.get("schema_version") != TASK14R_R0_SCHEMA_VERSION:
        raise ValueError("Task14R R0 schema_version drift")
    if protocol.get("status") != TASK14R_R0_STATUS_FROZEN:
        raise ValueError("Task14R R0 protocol is not frozen")
    if protocol.get("evidence_labels") != list(TASK14R_R0_EVIDENCE_LABELS):
        raise ValueError("Task14R R0 evidence labels drift")
    if (
        int(protocol.get("task_count", 0)) != TASK14R_R0_EXPECTED_TASK_COUNT
        or len(protocol.get("tasks", [])) != TASK14R_R0_EXPECTED_TASK_COUNT
    ):
        raise ValueError("Task14R R0 requires exactly 40 task definitions")
    if protocol.get("implementation_parent_commit") != expected_implementation_parent_commit:
        raise ValueError("Task14R R0 implementation_parent_commit drift")
    if protocol.get("source_files_sha256") != dict(sorted(expected_source_files_sha256.items())):
        raise ValueError("Task14R R0 source_files_sha256 drift")
    if set(expected_source_files_sha256) != set(TASK14R_R0_SOURCE_FILES):
        raise ValueError("Task14R R0 source hash inventory drift")
    for relative, digest in expected_source_files_sha256.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Task14R R0 invalid source SHA-256: {relative}")
    if protocol.get("permitted_terminal_statuses") != [
        TASK14R_R0_STATUS_PASSED,
        TASK14R_R0_STATUS_FAILED,
    ]:
        raise ValueError("Task14R R0 permitted_terminal_statuses drift")
    _validate_probe_actions(protocol.get("probe_actions"))
    expected = build_task14r_r0_protocol(
        implementation_parent_commit=expected_implementation_parent_commit,
        source_files_sha256=expected_source_files_sha256,
    )
    for field in (
        "schema_version",
        "task_name",
        "phase",
        "evidence_labels",
        "implementation_parent_commit",
        "task_count",
        "libero_commit",
        "renderer_contract",
        "renderer_qualification_boundary",
        "source_files_sha256",
        "restore_repeats_per_task",
        "probe_steps_per_trajectory",
        "expected_task_count",
        "expected_restore_transaction_count",
        "expected_probe_trajectory_count",
        "expected_probe_step_count",
        "probe_actions",
        "tasks",
        "policy_query_count",
        "formal_case_count",
        "formal_outcome_rollout_count",
        "training_or_parameter_updates",
        "automatic_next_phase",
        "permitted_terminal_statuses",
    ):
        if protocol.get(field) != expected[field]:
            raise ValueError(f"Task14R R0 protocol field drift: {field}")


def mujoco_model_fingerprint(sim: Any) -> dict[str, Any]:
    """Hash the compiled MJB plus every exposed numeric model array."""
    import mujoco

    model = sim.model._model
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    arrays = {}
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            arrays[name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
    payload = {"mjb_sha256": hashlib.sha256(binary.tobytes()).hexdigest(), "arrays": arrays}
    payload["complete_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def capture_mujoco_integration_state(sim: Any) -> dict[str, Any]:
    """Capture all MuJoCo fields needed to resume integration exactly."""
    import mujoco

    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(sim.model._model, specification), dtype=np.float64)
    mujoco.mj_getState(sim.model._model, sim.data._data, state, specification)
    return {"specification": specification, "state": state, "state_sha256": array_sha256(state)}


def configure_deterministic_renderer(env: Any) -> dict[str, Any]:
    """Disable offscreen MSAA and rebuild only the framebuffer context."""
    core = env._env.env
    sim = core.sim
    render_context = sim._render_context_offscreen
    if render_context is None:
        raise RuntimeError("Offscreen render context is unavailable")
    before = int(sim.model.vis.quality.offsamples)
    sim.model.vis.quality.offsamples = TASK14R_R0_RENDERER_OFFSAMPLES
    render_context.con.free()
    render_context._set_mujoco_context_and_buffers()
    sim.forward()
    after = int(sim.model.vis.quality.offsamples)
    if after != TASK14R_R0_RENDERER_OFFSAMPLES:
        raise RuntimeError("Failed to freeze MuJoCo offscreen antialiasing")
    return {
        "backend": TASK14R_R0_RENDERER_BACKEND,
        "offsamples_before": before,
        "offsamples_after": after,
        "pixel_gate": "EXACT",
        "simulator_or_model_rebuilt": False,
        "framebuffer_context_rebuilt": True,
    }


def restore_mujoco_integration_state(sim: Any, snapshot: Mapping[str, Any]) -> None:
    import mujoco

    specification = int(snapshot["specification"])
    state = np.asarray(snapshot["state"], dtype=np.float64)
    expected_size = mujoco.mj_stateSize(sim.model._model, specification)
    if state.shape != (expected_size,):
        raise ValueError("MuJoCo integration-state shape drift")
    mujoco.mj_setState(sim.model._model, sim.data._data, state, specification)
    mujoco.mj_forward(sim.model._model, sim.data._data)
    restored = capture_mujoco_integration_state(sim)
    if restored["specification"] != specification or not np.array_equal(restored["state"], state):
        raise RuntimeError("Exact MuJoCo integration-state restoration failed")


def _snapshot_controller(controller: Any) -> dict[str, Any]:
    interpolators = {}
    for name in ("interpolator_pos", "interpolator_ori"):
        value = getattr(controller, name, None)
        interpolators[name] = (
            None
            if value is None
            else {
                "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                "attributes": _snapshot_attributes(value, ignored=set(), path=f"controller.{name}"),
            }
        )
    return {
        "class": f"{controller.__class__.__module__}.{controller.__class__.__qualname__}",
        "attributes": _snapshot_attributes(
            controller,
            ignored=_CONTROLLER_REFERENCE_FIELDS,
            path="controller",
        ),
        "interpolators": interpolators,
    }


def _restore_controller(controller: Any, snapshot: Mapping[str, Any]) -> None:
    class_name = f"{controller.__class__.__module__}.{controller.__class__.__qualname__}"
    if class_name != snapshot["class"]:
        raise RuntimeError("Controller class drift")
    _restore_attributes(controller, snapshot["attributes"])
    for name, item in snapshot["interpolators"].items():
        current = getattr(controller, name, None)
        if item is None:
            if current is not None:
                raise RuntimeError(f"Controller interpolator drift: {name}")
            continue
        if current is None:
            raise RuntimeError(f"Controller interpolator disappeared: {name}")
        class_name = f"{current.__class__.__module__}.{current.__class__.__qualname__}"
        if class_name != item["class"]:
            raise RuntimeError(f"Controller interpolator class drift: {name}")
        _restore_attributes(current, item["attributes"])


def _snapshot_robot(robot: Any) -> dict[str, Any]:
    buffer_names = {name for name, value in vars(robot).items() if _is_buffer(value)}
    controller_config = {
        str(key): value for key, value in robot.controller_config.items() if str(key) != "sim"
    }
    return {
        "class": f"{robot.__class__.__module__}.{robot.__class__.__qualname__}",
        "attributes": _snapshot_attributes(
            robot,
            ignored=_ROBOT_REFERENCE_FIELDS | buffer_names,
            path="robot",
        ),
        "buffers": {
            name: {
                "class": f"{getattr(robot, name).__class__.__module__}."
                f"{getattr(robot, name).__class__.__qualname__}",
                "attributes": _snapshot_attributes(getattr(robot, name), ignored=set(), path=f"robot.{name}"),
            }
            for name in sorted(buffer_names)
        },
        "controller_config_without_sim": _plain_copy(
            controller_config, path="robot.controller_config_without_sim"
        ),
        "gripper": {
            "class": f"{robot.gripper.__class__.__module__}.{robot.gripper.__class__.__qualname__}",
            "current_action": _plain_copy(robot.gripper.current_action, path="gripper.current_action"),
        },
        "controller": _snapshot_controller(robot.controller),
    }


def _restore_robot(robot: Any, snapshot: Mapping[str, Any]) -> None:
    class_name = f"{robot.__class__.__module__}.{robot.__class__.__qualname__}"
    if class_name != snapshot["class"]:
        raise RuntimeError("Robot class drift")
    current_controller_config = {
        str(key): value for key, value in robot.controller_config.items() if str(key) != "sim"
    }
    if structured_hash(
        _plain_copy(current_controller_config, path="robot.controller_config_without_sim")
    ) != (structured_hash(snapshot["controller_config_without_sim"])):
        raise RuntimeError("Robot controller configuration drift")
    _restore_attributes(robot, snapshot["attributes"])
    for name, item in snapshot["buffers"].items():
        buffer = getattr(robot, name, None)
        if buffer is None or not _is_buffer(buffer):
            raise RuntimeError(f"Robot buffer disappeared: {name}")
        buffer_class = f"{buffer.__class__.__module__}.{buffer.__class__.__qualname__}"
        if buffer_class != item["class"]:
            raise RuntimeError(f"Robot buffer class drift: {name}")
        _restore_attributes(buffer, item["attributes"])
    gripper_class = f"{robot.gripper.__class__.__module__}.{robot.gripper.__class__.__qualname__}"
    if gripper_class != snapshot["gripper"]["class"]:
        raise RuntimeError("Gripper class drift")
    robot.gripper.current_action = _plain_copy(
        snapshot["gripper"]["current_action"], path="gripper.current_action"
    )
    _restore_controller(robot.controller, snapshot["controller"])
    expected_derived = {
        name: _plain_copy(snapshot["controller"]["attributes"][name], path=f"controller.{name}")
        for name in _CONTROLLER_DERIVED_FIELDS
    }
    robot.controller.update(force=True)
    for name, expected in expected_derived.items():
        actual = getattr(robot.controller, name)
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise RuntimeError(f"Controller derived-state reconstruction failed: {name}")
    # update(force=True) intentionally changes this cache flag.  Restore its
    # canonical value after verifying every derived field.
    robot.controller.new_update = bool(snapshot["controller"]["attributes"]["new_update"])
    if structured_hash(_snapshot_controller(robot.controller)) != structured_hash(snapshot["controller"]):
        raise RuntimeError("Exact controller-state restoration failed")


def capture_python_transaction_state(env: Any) -> dict[str, Any]:
    if getattr(env, "_env", None) is None or getattr(env._env, "env", None) is None:
        raise ValueError("LIBERO core environment is unavailable")
    core = env._env.env
    observable_states = {}
    for name, observable in sorted(core._observables.items()):
        observable_states[str(name)] = {
            field: _plain_copy(getattr(observable, field), path=f"observable.{name}.{field}")
            for field in _OBSERVABLE_STATE_FIELDS
        }
    initial_state = None
    if core.sim_state_initial is not None:
        initial_state = np.asarray(core.sim_state_initial.flatten()).copy()
    return {
        "outer": {"init_state_id": int(env.init_state_id)},
        "core": {
            field: _plain_copy(getattr(core, field), path=f"core.{field}") for field in _CORE_WRAPPER_FIELDS
        },
        "sim_state_initial": initial_state,
        "obs_cache": _plain_copy(core._obs_cache, path="core._obs_cache"),
        "observables": observable_states,
        "robots": [_snapshot_robot(robot) for robot in core.robots],
    }


def restore_python_transaction_state(env: Any, snapshot: Mapping[str, Any]) -> None:
    core = env._env.env
    env.init_state_id = int(snapshot["outer"]["init_state_id"])
    for field, value in snapshot["core"].items():
        setattr(core, field, _plain_copy(value, path=f"core.{field}"))
    if snapshot["sim_state_initial"] is None:
        core.sim_state_initial = None
    else:
        state_type = type(core.sim.get_state())
        core.sim_state_initial = state_type.from_flattened(
            np.asarray(snapshot["sim_state_initial"], dtype=np.float64), core.sim
        )
    core._obs_cache = _plain_copy(snapshot["obs_cache"], path="core._obs_cache")
    if set(core._observables) != set(snapshot["observables"]):
        raise RuntimeError("Observable inventory drift")
    for name, values in snapshot["observables"].items():
        observable = core._observables[name]
        for field, value in values.items():
            setattr(observable, field, _plain_copy(value, path=f"observable.{name}.{field}"))
    if len(core.robots) != len(snapshot["robots"]):
        raise RuntimeError("Robot inventory drift")
    for robot, robot_snapshot in zip(core.robots, snapshot["robots"], strict=True):
        _restore_robot(robot, robot_snapshot)


def synchronize_controller_state(env: Any) -> None:
    """Define a self-consistent capsule boundary after LIBERO settle steps."""
    core = env._env.env
    core.sim.forward()
    for robot in core.robots:
        robot.controller.update(force=True)
        # A normal completed policy step leaves the controller ready to update
        # when the next action goal is installed.
        robot.controller.new_update = True


def build_complete_state_capsule(env: Any, *, case: Mapping[str, Any]) -> dict[str, Any]:
    """Capture one canonical case and its observation-regeneration boundary."""
    core = env._env.env
    sim = core.sim
    try:
        synchronize_controller_state(env)
    except Exception as error:
        raise Task14RStageError("controller_reconstruction", str(error)) from error
    model = mujoco_model_fingerprint(sim)
    integration = capture_mujoco_integration_state(sim)
    entry_python = capture_python_transaction_state(env)
    try:
        raw_observation = core._get_observations(force_update=True)
        observation = env._format_raw_obs(raw_observation)
    except Exception as error:
        raise Task14RStageError("observation_regeneration", str(error)) from error
    ready_python = capture_python_transaction_state(env)
    ready_integration = capture_mujoco_integration_state(sim)
    if not np.array_equal(integration["state"], ready_integration["state"]):
        raise RuntimeError("Observation regeneration mutated MuJoCo integration state")
    return {
        "schema_version": "task14r.complete_state_capsule.v1",
        "case": dict(case),
        "model": model,
        "integration": integration,
        "entry_python": entry_python,
        "ready_python": ready_python,
        "canonical_observation": _plain_copy(observation, path="canonical_observation"),
        "canonical_observation_sha256": structured_hash(observation),
        "initial_success_predicate": bool(env._env.check_success()),
    }


def restore_complete_state_transaction(
    env: Any, capsule: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore a capsule without rebuilding the model or resetting the wrapper."""
    core = env._env.env
    before_model = mujoco_model_fingerprint(core.sim)
    if before_model["complete_sha256"] != capsule["model"]["complete_sha256"]:
        raise Task14RStageError("model_identity", "Compiled MuJoCo model changed before restoration")
    try:
        restore_mujoco_integration_state(core.sim, capsule["integration"])
    except Exception as error:
        raise Task14RStageError("complete_state_restore", str(error)) from error
    try:
        restore_python_transaction_state(env, capsule["entry_python"])
    except Exception as error:
        raise Task14RStageError("controller_reconstruction", str(error)) from error
    try:
        raw_observation = core._get_observations(force_update=True)
        observation = env._format_raw_obs(raw_observation)
    except Exception as error:
        raise Task14RStageError("observation_regeneration", str(error)) from error
    ready_python = capture_python_transaction_state(env)
    ready_integration = capture_mujoco_integration_state(core.sim)
    after_model = mujoco_model_fingerprint(core.sim)
    checks = {
        "model_identity": after_model["complete_sha256"] == capsule["model"]["complete_sha256"],
        "integration_identity": np.array_equal(
            ready_integration["state"], np.asarray(capsule["integration"]["state"])
        ),
        "python_ready_identity": structured_hash(ready_python) == structured_hash(capsule["ready_python"]),
        "observation_identity": structured_hash(observation) == str(capsule["canonical_observation_sha256"]),
        "initial_success_predicate_identity": bool(env._env.check_success())
        == bool(capsule["initial_success_predicate"]),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        failed = sorted(name for name, passed in checks.items() if name != "passed" and not passed)
        diagnostics = {
            "failed_checks": failed,
            "observation_mismatches": plain_state_mismatches(
                capsule["canonical_observation"], observation, path="observation"
            )[:20],
            "python_ready_mismatches": plain_state_mismatches(
                capsule["ready_python"], ready_python, path="python_ready"
            )[:20],
        }
        if "model_identity" in failed:
            stage = "model_identity"
        elif "observation_identity" in failed or "initial_success_predicate_identity" in failed:
            stage = "observation_regeneration"
        elif "python_ready_identity" in failed:
            stage = "controller_reconstruction"
        else:
            stage = "complete_state_restore"
        raise Task14RStageError(
            stage,
            f"Complete-state transaction failed: {json.dumps(diagnostics, sort_keys=True)}",
        )
    return observation, checks


def contact_state(core: Any) -> list[dict[str, Any]]:
    rows = []
    sim = core.sim
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1 = sim.model.geom_id2name(int(contact.geom1)) or f"geom:{int(contact.geom1)}"
        geom2 = sim.model.geom_id2name(int(contact.geom2)) or f"geom:{int(contact.geom2)}"
        rows.append(
            {
                "geoms": sorted((str(geom1), str(geom2))),
                "distance": float(contact.dist),
            }
        )
    return sorted(rows, key=lambda row: (row["geoms"], row["distance"]))


def capture_probe_step(
    env: Any,
    observation: Mapping[str, Any],
    *,
    repeat_index: int,
    step: int,
    action: Sequence[float],
    terminated: bool,
    truncated: bool,
    success: bool,
) -> dict[str, Any]:
    core = env._env.env
    integration = capture_mujoco_integration_state(core.sim)
    return {
        "repeat_index": int(repeat_index),
        "step": int(step),
        "action": [float(value) for value in action],
        "integration_state_sha256": integration["state_sha256"],
        "python_state_sha256": structured_hash(capture_python_transaction_state(env)),
        "observation_sha256": structured_hash(observation),
        "contact_state_sha256": structured_hash(contact_state(core)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_predicate": bool(success),
    }


def compare_probe_repeats(repeats: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if len(repeats) != TASK14R_R0_RESTORE_REPEATS:
        raise ValueError("Task14R R0 requires exactly three probe repeats")
    reference = repeats[0]
    mismatches = []
    for repeat_index, rows in enumerate(repeats[1:], start=1):
        if len(rows) != len(reference):
            mismatches.append(
                {
                    "repeat_index": repeat_index,
                    "step": min(len(reference), len(rows)),
                    "mismatched_fields": ["step_count"],
                }
            )
            continue
        for step, (left, right) in enumerate(zip(reference, rows, strict=True)):
            mismatched_fields = [
                field for field in PROBE_TRACE_COMPARISON_FIELDS if left.get(field) != right.get(field)
            ]
            if mismatched_fields:
                mismatches.append(
                    {
                        "repeat_index": repeat_index,
                        "step": step,
                        "mismatched_fields": mismatched_fields,
                    }
                )
                break
    first = mismatches[0] if mismatches else None
    return {
        "repeat_count": len(repeats),
        "step_count": len(reference),
        "mismatches": mismatches,
        "first_mismatch_repeat": None if first is None else first["repeat_index"],
        "first_mismatch_step": None if first is None else first["step"],
        "mismatched_fields": [] if first is None else first["mismatched_fields"],
        "passed": not mismatches,
    }


def task14r_r0_terminal_summary(task_audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(task_audits)
    passed = sum(bool(row.get("passed")) for row in task_audits)
    failed = task_count - passed
    restore_count = sum(int(row.get("restore_transaction_count", 0)) for row in task_audits)
    trajectory_count = sum(int(row.get("probe_trajectory_count", 0)) for row in task_audits)
    step_count = sum(int(row.get("probe_step_count", 0)) for row in task_audits)
    policy_query_count = sum(int(row.get("policy_query_count", 0)) for row in task_audits)
    formal_case_count = sum(int(row.get("formal_case_count", 0)) for row in task_audits)
    formal_outcome_count = sum(int(row.get("formal_outcome_rollout_count", 0)) for row in task_audits)
    training_updates = any(bool(row.get("training_or_parameter_updates")) for row in task_audits)
    automatic_next_phase = any(bool(row.get("automatic_next_phase")) for row in task_audits)
    gate = {
        "task_count": task_count == TASK14R_R0_EXPECTED_TASK_COUNT,
        "passed_task_count": passed == TASK14R_R0_EXPECTED_TASK_COUNT,
        "failed_task_count": failed == 0,
        "restore_transaction_count": restore_count == TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "probe_trajectory_count": trajectory_count == TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "probe_step_count": step_count == TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
        "policy_query_count": policy_query_count == 0,
        "formal_case_count": formal_case_count == 0,
        "formal_outcome_rollout_count": formal_outcome_count == 0,
        "training_or_parameter_updates": not training_updates,
        "automatic_next_phase": not automatic_next_phase,
    }
    status = TASK14R_R0_STATUS_PASSED if all(gate.values()) else TASK14R_R0_STATUS_FAILED
    return {
        "status": status,
        "task_count": task_count,
        "passed_task_count": passed,
        "failed_task_count": failed,
        "restore_transaction_count": restore_count,
        "probe_trajectory_count": trajectory_count,
        "probe_step_count": step_count,
        "policy_query_count": policy_query_count,
        "formal_case_count": formal_case_count,
        "formal_outcome_rollout_count": formal_outcome_count,
        "training_or_parameter_updates": training_updates,
        "automatic_next_phase": automatic_next_phase,
        "terminal_gate_checks": gate,
        "expected_task_count": TASK14R_R0_EXPECTED_TASK_COUNT,
        "expected_restore_transaction_count": TASK14R_R0_EXPECTED_RESTORE_TRANSACTION_COUNT,
        "expected_probe_trajectory_count": TASK14R_R0_EXPECTED_PROBE_TRAJECTORY_COUNT,
        "expected_probe_step_count": TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
        "renderer_qualification_boundary": list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY),
    }
