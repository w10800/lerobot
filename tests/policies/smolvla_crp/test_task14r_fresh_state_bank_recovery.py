from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.research.crp_vla.task14r_common import (
    TASK14R_PHASE_A_ARMS,
    audit_generated_distribution,
    audit_phase_a_historical_pair_identity,
    clipping_record,
    compare_closed_loop_steps,
    compose_online_action_chunk,
    freeze_state_bank,
    phase_a_offline_preprocessor_overrides,
    reserve_phase_a_output_root,
    structured_seed,
    terminal_status,
)


def test_phase_a_preprocessor_is_bound_to_frozen_offline_tokenizer(tmp_path: Path) -> None:
    offline_vlm = tmp_path / "offline_smolvlm"
    offline_vlm.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (offline_vlm / name).write_text("{}")

    overrides = phase_a_offline_preprocessor_overrides(offline_vlm, "cuda")

    assert overrides == {
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": str(offline_vlm.resolve())},
    }


def test_phase_a_offline_tokenizer_binding_fails_closed(tmp_path: Path) -> None:
    offline_vlm = tmp_path / "offline_smolvlm"
    offline_vlm.mkdir()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        phase_a_offline_preprocessor_overrides(offline_vlm, "cuda")


def test_phase_a_output_root_is_reserved_before_child_configuration(tmp_path: Path) -> None:
    output_root = tmp_path / "phase_a_attempt"
    config_dir = reserve_phase_a_output_root(output_root)

    assert output_root.is_dir()
    assert config_dir == output_root / "libero_standard_config"
    assert not config_dir.exists()
    with pytest.raises(FileExistsError):
        reserve_phase_a_output_root(output_root)


def _historical_state(fixture_x: float) -> dict[str, object]:
    return {
        "state_blob": torch.zeros(3).numpy(),
        "qpos": torch.zeros(2).numpy(),
        "qvel": torch.zeros(2).numpy(),
        "state_blob_sha256": "same-state-blob",
        "qpos_sha256": "same-qpos",
        "qvel_sha256": "same-qvel",
        "object_states": {
            "fixture_main": {
                "pos": torch.tensor([fixture_x, 0.0, 0.9], dtype=torch.float64).numpy(),
                "quat": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).numpy(),
            }
        },
        "robot_state": {},
        "end_effector_pose": {},
        "gripper_state": {},
        "canonical_observation": {"pixels": "excluded-from-physical-hash"},
    }


def test_phase_a_historical_identity_detects_nonserialized_fixture_drift() -> None:
    base = _historical_state(0.0)
    pairs = [
        {
            "case_id": f"case-{index}",
            "base_simulator_state": base,
            "snap_simulator_state": _historical_state(0.1 if index == 4 else 0.0),
        }
        for index in range(15)
    ]

    audit = audit_phase_a_historical_pair_identity(pairs)

    assert audit["status"] == "TASK14R_HISTORICAL_PAIR_IDENTITY_FAILED"
    assert audit["passed_case_count"] == 14
    assert audit["failed_case_count"] == 1
    assert audit["failed_case_ids"] == ["case-4"]
    assert audit["rows"][4]["state_blob_equal"]
    assert not audit["rows"][4]["physical_state_equal"]


def test_all_six_arms_preserve_component_contract() -> None:
    base = torch.arange(70, dtype=torch.float64).reshape(10, 7)
    snap = -base - 1
    outputs = {arm: compose_online_action_chunk(base, snap, arm) for arm in TASK14R_PHASE_A_ARMS}
    assert torch.equal(outputs["base_repeat"], base)
    assert torch.equal(outputs["full_swap"], base)
    assert torch.equal(outputs["snap_repeat"], snap)
    assert torch.equal(outputs["noop"], snap)
    assert torch.equal(outputs["pos_swap"][:, :3], base[:, :3])
    assert torch.equal(outputs["pos_swap"][:, 3:], snap[:, 3:])
    assert torch.equal(outputs["rot_swap"][:, :3], snap[:, :3])
    assert torch.equal(outputs["rot_swap"][:, 3:6], base[:, 3:6])
    assert torch.equal(outputs["rot_swap"][:, 6:], snap[:, 6:])


def test_candidate_seed_is_stable_and_namespaced() -> None:
    first = structured_seed("libero_spatial", 0, 0)
    second = structured_seed("libero_spatial", 0, 0)
    probe = structured_seed("libero_spatial", 0, 0, namespace="capacity_probe")
    assert first == second
    assert first["numpy_seed"] == 451399460
    assert first != probe


def test_clipping_record_keeps_before_and_after() -> None:
    row = clipping_record([1.2, -1.3, 0, 0, 0, 0, 2.0])
    assert row["before"] == [1.2, -1.3, 0.0, 0.0, 0.0, 0.0, 2.0]
    assert row["after"] == [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert row["changed_count"] == 3


def _step(index: int) -> dict[str, object]:
    return {
        "composed_action_before_clipping": [float(index)] * 7,
        "physical_state_before_sha256": f"before-{index}",
        "physical_state_after_sha256": f"after-{index}",
        "observation_sha256": f"obs-{index}",
        "replan_index": index // 10,
        "chunk_index": index % 10,
        "termination_reason": "SUCCESS" if index == 1 else None,
    }


def test_closed_loop_identity_localizes_state_divergence() -> None:
    left = [_step(0), _step(1)]
    right = copy.deepcopy(left)
    exact = compare_closed_loop_steps(left, right)
    assert exact["closed_loop_identity"]
    right[1]["physical_state_after_sha256"] = "different"
    mismatch = compare_closed_loop_steps(left, right)
    assert not mismatch["closed_loop_identity"]
    assert mismatch["first_physical_state_mismatch_step"] == 1
    assert mismatch["first_action_mismatch_step"] is None


def _candidate(task_key: str, index: int, accepted: bool = True) -> dict[str, object]:
    checks = {
        "finite_observation": accepted,
        "legal_initial_predicates": True,
        "not_initially_successful": True,
        "no_severe_penetration_or_explosion": True,
        "reset_reproducible": True,
        "zero_historical_overlap": True,
        "scene_task_match": True,
    }
    return {
        "task_key": task_key,
        "candidate_index": index,
        "state_id": f"{task_key}-candidate-{index}",
        "raw_state_sha256": f"raw-{task_key}-{index:02d}",
        "settled_physical_state_sha256": f"settled-{task_key}-{index:02d}",
        "policy_query_count": 0,
        "outcomes_accessed": False,
        "acceptance_checks": checks,
        "accepted": all(checks.values()),
    }


def test_state_bank_hash_sort_is_fixed_and_reserve_is_not_adaptive() -> None:
    rows = [_candidate("suite:0", index) for index in range(20)]
    bank = freeze_state_bank(rows, ["suite:0"])
    assert bank["status"] == "FRESH_STATE_BANK_400_FROZEN"
    assert bank["formal_case_count"] == 10
    assert bank["reserve_case_count"] == 10
    assert not bank["adaptive_reserve_activation"]
    assert [row["candidate_index"] for row in bank["formal_states"]] == list(range(10))


def test_state_bank_refuses_outcome_conditioned_selection() -> None:
    rows = [_candidate("suite:0", index) for index in range(20)]
    rows[0]["base_success"] = True
    with pytest.raises(ValueError, match="Outcome-conditioned"):
        freeze_state_bank(rows, ["suite:0"])


def _distribution_rows(task_key: str, count: int, offset: float) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        value = float(index) / max(count - 1, 1) + offset
        rows.append(
            {
                "task_key": task_key,
                "state_id": f"{task_key}-{index}",
                "features": {
                    "robot_joint_pos/j0": value,
                    "eef_position/x": value,
                    "object_position/object/x": value,
                    "settling_displacement/max_object": value / 100,
                },
            }
        )
    return rows


def test_distribution_audit_passes_supported_generated_states() -> None:
    official = _distribution_rows("suite:0", 50, 0.0)
    formal = _distribution_rows("suite:0", 10, 0.0)
    audit = audit_generated_distribution(official, formal, ["suite:0"])
    assert audit["status"] == "GENERATED_STATE_DISTRIBUTION_AUDIT_PASSED"


def test_distribution_audit_requires_four_task_family_mismatches() -> None:
    tasks = [f"suite:{index}" for index in range(4)]
    official = [row for task in tasks for row in _distribution_rows(task, 50, 0.0)]
    formal = [row for task in tasks for row in _distribution_rows(task, 10, 10.0)]
    audit = audit_generated_distribution(official, formal, tasks)
    assert audit["status"] == "GENERATED_STATE_DISTRIBUTION_MISMATCH"


def test_terminal_status_requires_every_recovery_gate() -> None:
    ready = terminal_status(
        bank_status="FRESH_STATE_BANK_400_FROZEN",
        distribution_status="GENERATED_STATE_DISTRIBUTION_AUDIT_PASSED",
        capacity_probe_passed=True,
        zero_historical_overlap=True,
        all_formal_reproducible=True,
    )
    assert ready == "READY_FOR_TASK14_CAUSAL_RELAUNCH"
    assert (
        terminal_status(
            bank_status="FRESH_STATE_BANK_400_FROZEN",
            distribution_status="GENERATED_STATE_DISTRIBUTION_MISMATCH",
            capacity_probe_passed=True,
            zero_historical_overlap=True,
            all_formal_reproducible=True,
        )
        == "GENERATED_STATE_DISTRIBUTION_MISMATCH"
    )
