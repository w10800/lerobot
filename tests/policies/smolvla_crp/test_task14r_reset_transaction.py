from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research.crp_vla.task14r_reset_transaction import (
    TASK14R_R0_PROBE_ACTIONS,
    TASK14R_R0_STATUS_FAILED,
    TASK14R_R0_STATUS_PASSED,
    _plain_copy,
    build_task14r_r0_protocol,
    compare_probe_repeats,
    configure_deterministic_renderer,
    task14r_r0_seed,
    task14r_r0_terminal_summary,
    validate_task14r_r0_protocol,
)


def test_r0_protocol_is_complete_policy_free_and_nonformal() -> None:
    protocol = build_task14r_r0_protocol(implementation_parent_commit="parent")
    validate_task14r_r0_protocol(protocol)

    assert protocol["task_count"] == 40
    assert len({row["case_id"] for row in protocol["tasks"]}) == 40
    assert {row["init_state_id"] for row in protocol["tasks"]} == {0}
    assert protocol["restore_repeats_per_task"] == 3
    assert len(protocol["probe_actions"]) == 10
    assert protocol["renderer_contract"] == {
        "backend": "egl",
        "offsamples": 0,
        "pixel_gate": "EXACT",
    }
    assert protocol["policy_query_count"] == 0
    assert protocol["formal_case_count"] == 0
    assert protocol["formal_outcome_rollout_count"] == 0
    assert not protocol["training_or_parameter_updates"]
    assert not protocol["automatic_next_phase"]


def test_r0_protocol_rejects_adaptive_or_reduced_scope() -> None:
    protocol = build_task14r_r0_protocol(implementation_parent_commit="parent")
    protocol["tasks"] = protocol["tasks"][:-1]
    protocol["task_count"] = 39
    with pytest.raises(ValueError, match="exactly 40"):
        validate_task14r_r0_protocol(protocol)

    protocol = build_task14r_r0_protocol(implementation_parent_commit="parent")
    protocol["probe_actions"][0][0] = 0.5
    with pytest.raises(ValueError, match="probe_actions"):
        validate_task14r_r0_protocol(protocol)


def test_r0_seed_and_probe_sequence_are_stable() -> None:
    assert task14r_r0_seed("libero_spatial", 0) == task14r_r0_seed("libero_spatial", 0)
    assert task14r_r0_seed("libero_spatial", 0) != task14r_r0_seed("libero_spatial", 1)
    assert TASK14R_R0_PROBE_ACTIONS[0] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def _probe_step(step: int) -> dict[str, object]:
    return {
        "step": step,
        "action": [0.0] * 6 + [-1.0],
        "integration_state_sha256": f"integration-{step}",
        "python_state_sha256": f"python-{step}",
        "observation_sha256": f"observation-{step}",
        "contact_state_sha256": f"contact-{step}",
        "terminated": False,
        "truncated": False,
        "success_predicate": False,
    }


def test_probe_repeat_comparison_is_exact_and_localizes_first_mismatch() -> None:
    reference = [_probe_step(index) for index in range(10)]
    exact = compare_probe_repeats([copy.deepcopy(reference) for _ in range(3)])
    assert exact["passed"]
    changed = [copy.deepcopy(reference) for _ in range(3)]
    changed[2][4]["python_state_sha256"] = "different"
    mismatch = compare_probe_repeats(changed)
    assert not mismatch["passed"]
    assert mismatch["mismatches"] == [{"repeat_index": 2, "reason": "step_state", "step": 4}]


def test_complete_state_values_reject_opaque_objects_and_copy_arrays() -> None:
    original = np.arange(4, dtype=np.float64)
    copied = _plain_copy({"array": original, "shape": (2, 2)}, path="root")
    original[0] = 100
    assert copied["array"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert copied["shape"] == [2, 2]
    with pytest.raises(TypeError, match="Unsupported complete-state value"):
        _plain_copy(object(), path="root.opaque")


def test_renderer_contract_disables_msaa_without_rebuilding_simulator() -> None:
    class Context:
        def __init__(self) -> None:
            self.con = SimpleNamespace(free=lambda: setattr(self, "freed", True))
            self.freed = False
            self.rebuilt = False

        def _set_mujoco_context_and_buffers(self) -> None:
            self.rebuilt = True

    context = Context()
    sim = SimpleNamespace(
        model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4))),
        _render_context_offscreen=context,
        forward=lambda: None,
    )
    env = SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(sim=sim)))

    contract = configure_deterministic_renderer(env)

    assert contract == {
        "backend": "egl",
        "offsamples_before": 4,
        "offsamples_after": 0,
        "pixel_gate": "EXACT",
        "simulator_or_model_rebuilt": False,
        "framebuffer_context_rebuilt": True,
    }
    assert context.freed
    assert context.rebuilt
    assert sim.model.vis.quality.offsamples == 0


def test_r0_terminal_summary_cannot_pass_partial_or_failed_tasks() -> None:
    passed_tasks = [
        {
            "passed": True,
            "restore_transaction_count": 3,
            "probe_trajectory_count": 3,
        }
        for _ in range(40)
    ]
    complete = task14r_r0_terminal_summary(passed_tasks)
    assert complete["status"] == TASK14R_R0_STATUS_PASSED
    assert complete["restore_transaction_count"] == 120
    assert complete["probe_trajectory_count"] == 120
    assert complete["formal_outcome_rollout_count"] == 0
    assert not complete["training_or_parameter_updates"]

    assert task14r_r0_terminal_summary(passed_tasks[:-1])["status"] == TASK14R_R0_STATUS_FAILED
    failed = copy.deepcopy(passed_tasks)
    failed[0]["passed"] = False
    assert task14r_r0_terminal_summary(failed)["status"] == TASK14R_R0_STATUS_FAILED


def test_r0_runner_has_one_canonical_reset_and_no_policy_or_training_entrypoint() -> None:
    runner = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla" / "run_task14r_r0.py"
    source = runner.read_text()
    tree = ast.parse(source)
    reset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
    ]
    assert len(reset_calls) == 1
    assert "SmolVLA" not in source
    assert "predict_action_chunk" not in source
    assert "base-checkpoint" not in source
    assert "snap-checkpoint" not in source
    assert "optimizer" not in source.lower()


def test_r0_launcher_is_fixed_fail_closed_and_policy_free() -> None:
    launcher = (
        Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla" / "launch_task14r_r0.sh"
    )
    source = launcher.read_text()
    assert "set -euo pipefail" in source
    assert "codex/task14r-reset-transaction-recovery" in source
    assert 'OUTPUT_ROOT="${OUTPUT_PARENT}/r0_attempt001"' in source
    assert "export MUJOCO_GL=egl" in source
    assert 'if [[ -e "${OUTPUT_ROOT}" ]]' in source
    assert "run_task14r_r0.py" in source
    assert "checkpoint" not in source.lower()
    assert "phase_b" not in source.lower()
    assert "task14c" not in source.lower()
