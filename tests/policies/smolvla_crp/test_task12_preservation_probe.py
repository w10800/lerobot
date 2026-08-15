from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_task12_probe import balanced_outcome_blind_split, require_localized_mechanism  # noqa: E402
from run_task12_preservation_probe import normalize_mean_std, select_denormalized_targets  # noqa: E402


def test_probe_split_is_balanced_deterministic_and_outcome_blind() -> None:
    shards = [
        {"shard_id": f"shard-{index:02d}", "case_ids": [f"case-{index}-{j}" for j in range(5)]}
        for index in range(4)
    ]
    cases = {case_id for shard in shards for case_id in shard["case_ids"]}
    train_a, heldout_a = balanced_outcome_blind_split(shards, cases, seed=17)
    train_b, heldout_b = balanced_outcome_blind_split(list(reversed(shards)), cases, seed=17)
    assert (train_a, heldout_a) == (train_b, heldout_b)
    assert len(train_a) == 16
    assert len(heldout_a) == 4
    assert not set(train_a) & set(heldout_a)
    for shard in shards:
        assert len(set(shard["case_ids"]) & set(heldout_a)) == 1


def test_probe_requires_positive_localized_physical_divergence() -> None:
    metrics = {
        name: {"available": True, "point": 0.2, "ci95": [0.1, 0.3]}
        for name in (
            "mean_joint_state_l2_h1",
            "mean_joint_state_l2_h10",
            "mean_end_effector_position_l2_h1",
            "mean_end_effector_position_l2_h10",
        )
    }
    branch = {
        "status": "TASK12_PHASE_B_ISOLATED_BRANCHES_COMPLETE",
        "harmful_minus_preserved": metrics,
    }
    assert set(require_localized_mechanism(branch)) == set(metrics)
    metrics["mean_joint_state_l2_h1"]["ci95"][0] = -0.1
    with pytest.raises(ValueError, match="mechanism evidence"):
        require_localized_mechanism(branch)


def test_mean_std_action_normalization() -> None:
    action = np.asarray([[2.0, 5.0], [4.0, 9.0]], dtype=np.float32)
    mean = np.asarray([2.0, 1.0], dtype=np.float32)
    std = np.asarray([2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(normalize_mean_std(action, mean, std, eps=0.0), [[0.0, 1.0], [1.0, 2.0]])


def test_original_snap_is_the_self_replay_target_for_both_origin_arms() -> None:
    base = np.asarray([[1.0]], dtype=np.float32)
    snap = np.asarray([[2.0]], dtype=np.float32)
    target, self_target, kind = select_denormalized_targets("snap1_20k", base, snap)
    assert kind == "preservation"
    np.testing.assert_array_equal(target, base)
    np.testing.assert_array_equal(self_target, snap)
    target, self_target, kind = select_denormalized_targets("base10", base, snap)
    assert kind == "anchor"
    np.testing.assert_array_equal(target, snap)
    np.testing.assert_array_equal(self_target, snap)
