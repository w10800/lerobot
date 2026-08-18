from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.research.crp_vla.task14_common import (
    ACTION_DIM,
    ORIENTATION_INDICES,
    POSITION_INDICES,
    audit_formal_state_capacity,
    collect_used_initial_state_ids,
    compose_action_chunks,
    controller_clipping_audit,
    derive_seed,
)


@pytest.fixture
def chunks() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(2 * 10 * ACTION_DIM, dtype=torch.float64).reshape(2, 10, ACTION_DIM)
    snap = -base - 1
    return base, snap


def test_noop_identity(chunks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base, snap = chunks
    output = compose_action_chunks(base, snap, "noop")
    assert torch.equal(output, snap)
    assert output.data_ptr() != snap.data_ptr()


def test_full_swap_identity(chunks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base, snap = chunks
    output = compose_action_chunks(base, snap, "full_swap")
    assert torch.equal(output, base)
    assert output.data_ptr() != base.data_ptr()


def test_position_mask_integrity(chunks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base, snap = chunks
    output = compose_action_chunks(base, snap, "pos_swap")
    assert torch.equal(output[..., list(POSITION_INDICES)], base[..., list(POSITION_INDICES)])
    assert torch.equal(output[..., 3:], snap[..., 3:])


def test_rotation_mask_integrity(chunks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base, snap = chunks
    output = compose_action_chunks(base, snap, "rot_swap")
    assert torch.equal(output[..., list(ORIENTATION_INDICES)], base[..., list(ORIENTATION_INDICES)])
    assert torch.equal(output[..., :3], snap[..., :3])
    assert torch.equal(output[..., 6:], snap[..., 6:])


def test_chunk_alignment_and_shape_validation(chunks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base, snap = chunks
    output = compose_action_chunks(base, snap, "pos_swap")
    assert output.shape == (2, 10, ACTION_DIM)
    with pytest.raises(ValueError, match="identical shapes"):
        compose_action_chunks(base[:, :-1], snap, "pos_swap")
    with pytest.raises(ValueError, match="final dimension"):
        compose_action_chunks(base[..., :-1], snap[..., :-1], "pos_swap")


def test_clipping_audit_is_explicit_per_dimension() -> None:
    actions = np.zeros((2, ACTION_DIM), dtype=np.float64)
    actions[0, 0] = 1.25
    actions[1, 6] = -1.5
    audit = controller_clipping_audit(actions)
    assert audit["total_count"] == 2
    assert audit["per_dimension"][0]["count"] == 1
    assert audit["per_dimension"][0]["maximum_absolute_magnitude"] == 0.25
    assert audit["per_dimension"][6]["count"] == 1
    assert audit["per_dimension"][6]["maximum_absolute_magnitude"] == 0.5
    assert audit["clipped"][0, 0] == 1.0
    assert audit["clipped"][1, 6] == -1.0


def test_registered_seed_is_stable() -> None:
    seed = derive_seed()
    assert seed == {
        "string": "CRP-VLA-TASK14-POSITION-CAUSAL-V1",
        "sha256": "1c2b7fcb74858357c6f5f41e43e1c2b1072c7628bdae2519fe7b23564ac8e0e4",
        "integer": 2029856568870536023,
    }


def test_capacity_gate_refuses_partial_formal_design() -> None:
    rows = [{"suite": "suite", "task_id": 0, "init_state_id": index} for index in range(49)]
    used, summary = collect_used_initial_state_ids((("history", rows),))
    assert summary[0]["unique_initial_cases"] == 49
    audit = audit_formal_state_capacity({("suite", 0): 50}, used, required_per_task=10)
    assert audit["status"] == "INSUFFICIENT_UNTOUCHED_FORMAL_STATES"
    assert audit["available_untouched_case_count"] == 1
    assert audit["tasks"][0]["untouched_state_ids"] == [49]


def test_capacity_gate_accepts_exact_requirement() -> None:
    rows = [{"suite": "suite", "task_id": 0, "init_state_id": index} for index in range(40)]
    used, _ = collect_used_initial_state_ids((("history", rows),))
    audit = audit_formal_state_capacity({("suite", 0): 50}, used, required_per_task=10)
    assert audit["status"] == "READY_FOR_FORMAL_MANIFEST"
    assert audit["tasks"][0]["untouched_state_ids"] == list(range(40, 50))
