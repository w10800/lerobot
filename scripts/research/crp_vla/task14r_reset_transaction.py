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
TASK14R_R0_SEED_NAMESPACE = "CRP-VLA-TASK14R-R0-RESET-TRANSACTION-V1"
TASK14R_R0_LIBERO_COMMIT = "8460457bfca6e0ef2e856bc104e2c60b023ef2a7"
TASK14R_R0_RENDERER_BACKEND = "egl"
TASK14R_R0_RENDERER_OFFSAMPLES = 0

# Ten deliberately small, policy-free OSC commands.  The sequence exercises
# translation, rotation, and gripper state without selecting cases by outcome.
TASK14R_R0_PROBE_ACTIONS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
)

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


def build_task14r_r0_protocol(*, implementation_parent_commit: str) -> dict[str, Any]:
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
        "schema_version": "task14r.reset_transaction.protocol.v1",
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
        "task_count": len(tasks),
        "restore_repeats_per_task": TASK14R_R0_RESTORE_REPEATS,
        "probe_actions": [list(row) for row in TASK14R_R0_PROBE_ACTIONS],
        "tasks": tasks,
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
        "permitted_terminal_statuses": [TASK14R_R0_STATUS_PASSED, TASK14R_R0_STATUS_FAILED],
    }


def validate_task14r_r0_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != TASK14R_R0_STATUS_FROZEN:
        raise ValueError("Task14R R0 protocol is not frozen")
    if protocol.get("evidence_labels") != list(TASK14R_R0_EVIDENCE_LABELS):
        raise ValueError("Task14R R0 evidence labels drift")
    if int(protocol.get("task_count", 0)) != 40 or len(protocol.get("tasks", [])) != 40:
        raise ValueError("Task14R R0 requires exactly 40 task definitions")
    expected = build_task14r_r0_protocol(
        implementation_parent_commit=str(protocol.get("implementation_parent_commit", ""))
    )
    for field in (
        "task_name",
        "phase",
        "evidence_labels",
        "task_count",
        "libero_commit",
        "renderer_contract",
        "restore_repeats_per_task",
        "probe_actions",
        "tasks",
        "policy_query_count",
        "formal_case_count",
        "formal_outcome_rollout_count",
        "training_or_parameter_updates",
        "automatic_next_phase",
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
    synchronize_controller_state(env)
    model = mujoco_model_fingerprint(sim)
    integration = capture_mujoco_integration_state(sim)
    entry_python = capture_python_transaction_state(env)
    raw_observation = core._get_observations(force_update=True)
    observation = env._format_raw_obs(raw_observation)
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
        raise RuntimeError("Compiled MuJoCo model changed before restoration")
    restore_mujoco_integration_state(core.sim, capsule["integration"])
    restore_python_transaction_state(env, capsule["entry_python"])
    raw_observation = core._get_observations(force_update=True)
    observation = env._format_raw_obs(raw_observation)
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
        raise RuntimeError(f"Complete-state transaction failed: {json.dumps(diagnostics, sort_keys=True)}")
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
    step: int,
    action: Sequence[float],
    terminated: bool,
    truncated: bool,
    success: bool,
) -> dict[str, Any]:
    core = env._env.env
    integration = capture_mujoco_integration_state(core.sim)
    return {
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
            mismatches.append({"repeat_index": repeat_index, "reason": "step_count", "step": None})
            continue
        for step, (left, right) in enumerate(zip(reference, rows, strict=True)):
            if structured_hash(left) != structured_hash(right):
                mismatches.append({"repeat_index": repeat_index, "reason": "step_state", "step": step})
                break
    return {
        "repeat_count": len(repeats),
        "step_count": len(reference),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def task14r_r0_terminal_summary(task_audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(task_audits)
    passed = sum(bool(row.get("passed")) for row in task_audits)
    status = TASK14R_R0_STATUS_PASSED if task_count == 40 and passed == 40 else TASK14R_R0_STATUS_FAILED
    return {
        "status": status,
        "task_count": task_count,
        "passed_task_count": passed,
        "failed_task_count": task_count - passed,
        "restore_transaction_count": sum(int(row.get("restore_transaction_count", 0)) for row in task_audits),
        "probe_trajectory_count": sum(int(row.get("probe_trajectory_count", 0)) for row in task_audits),
        "policy_query_count": 0,
        "formal_case_count": 0,
        "formal_outcome_rollout_count": 0,
        "training_or_parameter_updates": False,
        "automatic_next_phase": False,
    }
