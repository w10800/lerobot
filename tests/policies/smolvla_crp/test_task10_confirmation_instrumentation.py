from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_DIR = Path(__file__).parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_module("task10_common")
TRACE = load_module("trajectory_instrumentation")
REPLAY = load_module("replay_v2_common")


def make_case(task: int, state: int) -> dict:
    old = state < 15
    case_id = f"task{task:02d}-state{state:02d}"
    return {
        "case_id": case_id,
        "task_id": f"suite:{task}",
        "task_name": "task",
        "initial_state_id": state,
        "initial_state_source": f"fixture[{state}]",
        "initial_state_hash": f"initial-{case_id}",
        "simulator_state_hash": f"sim-{case_id}",
        "qpos_qvel_hash": f"q-{case_id}",
        "observation_hash": f"obs-{case_id}",
        "processor_hash": "processor",
        "evaluator_hash": "evaluator",
        "runtime_metadata": {},
        "noise_schedule_hash": f"noise-{case_id}",
        "formal100_overlap_check": False,
        "dev40_overlap_check": False,
        "task9_600_membership": old,
    }


def make_manifest() -> dict:
    cases = [make_case(task, state) for task in range(40) for state in range(30)]
    return {
        "schema_version": 2,
        "status": "CONFIRMATION1200_MANIFEST_FROZEN",
        "selected_checkpoint_step": COMMON.SELECTED_STEP,
        "selected_model_sha256": COMMON.SELECTED_MODEL_SHA256,
        "noninferiority_margin": COMMON.NONINFERIORITY_MARGIN,
        "case_count": 1200,
        "task_count": 40,
        "cases_per_task": 30,
        "cases": cases,
    }


def test_confirmation1200_manifest_requires_exact_balanced_unique_design():
    manifest = make_manifest()
    COMMON.validate_confirmation1200_manifest(manifest)
    manifest["cases"][1]["simulator_state_hash"] = manifest["cases"][0]["simulator_state_hash"]
    with pytest.raises(ValueError, match="Duplicate Task 10 simulator_state_hash"):
        COMMON.validate_confirmation1200_manifest(manifest)


def test_task9_subset_must_be_unchanged():
    manifest = make_manifest()
    old = []
    for case in manifest["cases"]:
        if case["task9_600_membership"]:
            old.append({key: value for key, value in case.items() if key != "task9_600_membership"})
    COMMON.validate_task9_subset(manifest["cases"], old)
    manifest["cases"][0]["observation_hash"] = "changed"
    with pytest.raises(ValueError, match="subset case changed"):
        COMMON.validate_task9_subset(manifest["cases"], old)


def test_diagnostic_subset_is_outcome_blind_balanced_and_reconstructible():
    cases = make_manifest()["cases"]
    first = COMMON.confirmation1200_diagnostic_subset(cases)
    second = COMMON.confirmation1200_diagnostic_subset(list(reversed(cases)))
    assert first == second
    assert first["case_count"] == 200
    assert first["outcome_blind"] is True
    assert first["primary_aggregation_eligible"] is False


def test_primary_aggregator_refuses_incomplete_and_diagnostic_contamination():
    ids = {case["case_id"] for case in make_manifest()["cases"]}
    with pytest.raises(RuntimeError, match="completed_primary_cases=0/1200"):
        COMMON.strict_primary_aggregate([], ids)
    complete = [
        {"case_id": case_id, "arm": arm, "status": "COMPLETED", "success": False}
        for case_id in sorted(ids)
        for arm in COMMON.PRIMARY_ARMS
    ]
    complete.append({"case_id": sorted(ids)[0], "arm": "snap2_20k", "status": "COMPLETED", "success": True})
    with pytest.raises(RuntimeError, match="contamination"):
        COMMON.strict_primary_aggregate(complete, ids)


def trace_kwargs() -> dict:
    return {
        "identity": {
            "case_id": "case",
            "task_id": "suite:0",
            "arm_id": "base10",
            "rollout_id": "rollout",
            "replan_index": 0,
            "simulator_timestep": 0,
        },
        "raw_observation": {
            "rgb": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "robot_state": {"gripper": np.asarray([0.1, 0.2])},
        },
        "normalized_observation": {
            "image": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
            "state": torch.arange(8, dtype=torch.bfloat16).reshape(1, 8),
            "task": ["pick object"],
        },
        "simulator_state": {
            "state_blob": np.arange(6, dtype=np.float64),
            "qpos": np.arange(3, dtype=np.float64),
            "qvel": np.arange(3, dtype=np.float64),
            "object_states": {"object": {"pos": np.zeros(3), "quat": np.asarray([0, 0, 0, 1])}},
            "robot_state": {"eef": {"pos": np.zeros(3), "quat": np.asarray([0, 0, 0, 1])}},
            "gripper_state": np.zeros(2),
        },
        "language_condition": {"instruction": "pick object", "reference": "fixture"},
        "policy_output": {
            "raw_policy_output": torch.zeros(1, 50, 32),
            "denormalized_action": torch.zeros(10, 7),
            "executed_action_chunk": np.zeros((10, 7), dtype=np.float32),
            "execution_horizon": 10,
            "nfe": 10,
            "noise_tensor": np.zeros((1, 50, 32), dtype=np.float32),
            "noise_seed": 42,
            "action_mask": None,
            "processor_contract": {"version": 1},
            "processor_hash": "processor",
        },
        "event_state": {
            "contact_information": {"availability": "AVAILABLE", "pairs": []},
            "grasp_state": {"availability": "UNAVAILABLE"},
            "object_gripper_distance": {"availability": "AVAILABLE", "values": {}},
        },
    }


def test_replan_trace_round_trip_preserves_raw_and_normalized_observations(tmp_path):
    kwargs = trace_kwargs()
    record = TRACE.write_replan_trace(tmp_path / "replan", **kwargs)
    metadata, payload = TRACE.load_replan_trace(tmp_path / "replan")
    assert len(record["sha256"]) == 64
    assert metadata["identity"]["replan_index"] == 0
    np.testing.assert_array_equal(payload["raw_observation"]["rgb"], kwargs["raw_observation"]["rgb"])
    assert payload["normalized_observation"]["state"].dtype == torch.bfloat16
    assert REPLAY.structured_hash(payload["normalized_observation"]) == REPLAY.structured_hash(
        kwargs["normalized_observation"]
    )


class FakeModel:
    nbody = 2

    @staticmethod
    def body_id2name(index: int):
        return "target_object" if index == 1 else None


class FakeData:
    def __init__(self):
        self.qpos = np.asarray([1.0, 2.0])
        self.qvel = np.asarray([3.0, 4.0])
        self.body_xpos = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        self.body_xquat = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])


class FakeInner:
    def __init__(self):
        self.sim = type("Sim", (), {"model": FakeModel(), "data": FakeData()})()

    def get_sim_state(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    def set_init_state(self, state):
        state = np.asarray(state)
        self.sim.data.qpos[:] = state[:2]
        self.sim.data.qvel[:] = state[2:]
        return {}


def fake_observation() -> dict:
    return {
        "robot_state": {
            "eef": {"pos": np.zeros(3), "quat": np.asarray([0.0, 0.0, 0.0, 1.0])},
            "gripper": np.asarray([0.1, 0.2]),
        }
    }


def test_libero_state_save_perturb_restore_is_exact():
    inner = FakeInner()
    observation = fake_observation()
    state = TRACE.capture_libero_state(inner, observation, ["target"])
    inner.sim.data.qpos[:] = 9
    inner.sim.data.qvel[:] = 8
    restored_observation = TRACE.restore_libero_state(inner, state)
    restored = TRACE.capture_libero_state(inner, fake_observation(), ["target"])
    np.testing.assert_array_equal(restored["qpos"], state["qpos"])
    np.testing.assert_array_equal(restored["qvel"], state["qvel"])
    assert REPLAY.structured_hash(restored["object_states"]) == REPLAY.structured_hash(state["object_states"])
    assert REPLAY.structured_hash(restored_observation) == REPLAY.structured_hash(observation)


def transition_state(position: float) -> dict:
    return {
        "state_blob": np.asarray([position]),
        "qpos": np.asarray([position]),
        "qvel": np.asarray([0.0]),
        "object_states": {
            "object": {
                "pos": np.asarray([position, 0.0, 0.0]),
                "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            }
        },
        "robot_state": {},
        "end_effector_pose": {
            "pos": np.asarray([position, 0.0, 0.0]),
            "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
        "gripper_state": np.asarray([position]),
    }


def test_counterfactual_branches_restore_identical_state_and_support_all_horizons():
    current = transition_state(0.0)
    frozen = copy.deepcopy(current)

    def restore(state):
        nonlocal current
        current = copy.deepcopy(state)
        return copy.deepcopy(current)

    def capture():
        return copy.deepcopy(current)

    def step(action):
        nonlocal current
        current = transition_state(float(current["qpos"][0] + action[0]))

    result = TRACE.execute_counterfactual_branches(
        frozen_state=frozen,
        action_chunks={"a": np.ones((10, 1)), "b": np.full((10, 1), 2.0)},
        restore_state=restore,
        capture_state=capture,
        step_action=step,
    )
    assert result["identical_initial_state"] is True
    assert result["horizons"] == [1, 3, 5, 10]
    assert result["transition_divergence"]["a_vs_b"]["10"]["joint_state_l2"]["value"] == 10.0
    assert result["combined_weighted_scalar_used"] is False


def test_transition_metrics_use_quaternion_geodesic_and_no_weighted_scalar():
    left = transition_state(0.0)
    right = transition_state(1.0)
    right["end_effector_pose"]["quat"] = np.asarray([0.0, 0.0, 1.0, 0.0])
    metrics = COMMON.transition_divergence(left, right)
    assert metrics["joint_state_l2"]["value"] == 1.0
    assert metrics["end_effector_position_l2"]["value"] == 1.0
    assert metrics["end_effector_orientation_radians"]["value"] == pytest.approx(np.pi)
    assert metrics["combined_weighted_scalar"]["available"] is False
