from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research.crp_vla.task14r_reset_transaction import (
    TASK14R_R0_EXPECTED_PROBE_STEP_COUNT,
    TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0_PROBE_ACTIONS,
    TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY,
    TASK14R_R0_SCHEMA_VERSION,
    TASK14R_R0_SOURCE_FILES,
    TASK14R_R0_STATUS_FAILED,
    TASK14R_R0_STATUS_PASSED,
    Task14RStageError,
    _plain_copy,
    build_task14r_r0_protocol,
    compare_probe_repeats,
    configure_deterministic_renderer,
    task14r_r0_seed,
    task14r_r0_terminal_summary,
    validate_task14r_r0_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts" / "research" / "crp_vla"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_task14r_r0 import (  # noqa: E402
    execute_task_with_audit,
    file_sha256,
    validate_pre_simulator_gate,
    write_probe_trace,
)


def _source_hashes(fill: str = "a") -> dict[str, str]:
    return dict.fromkeys(TASK14R_R0_SOURCE_FILES, fill * 64)


def _protocol(source_hashes: dict[str, str] | None = None) -> dict[str, object]:
    hashes = _source_hashes() if source_hashes is None else source_hashes
    return build_task14r_r0_protocol(
        implementation_parent_commit=TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=hashes,
    )


def _validate(protocol: dict[str, object], source_hashes: dict[str, str] | None = None) -> None:
    hashes = _source_hashes() if source_hashes is None else source_hashes
    validate_task14r_r0_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=hashes,
    )


def test_r0_protocol_is_complete_policy_free_and_nonformal() -> None:
    protocol = _protocol()
    _validate(protocol)

    assert protocol["schema_version"] == TASK14R_R0_SCHEMA_VERSION
    assert protocol["implementation_parent_commit"] == TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT
    assert protocol["task_count"] == 40
    assert len({row["case_id"] for row in protocol["tasks"]}) == 40
    assert {row["init_state_id"] for row in protocol["tasks"]} == {0}
    assert protocol["restore_repeats_per_task"] == 3
    assert protocol["probe_steps_per_trajectory"] == 15
    assert len(protocol["probe_actions"]) == 15
    assert protocol["expected_probe_step_count"] == 1800
    assert protocol["renderer_contract"] == {
        "backend": "egl",
        "offsamples": 0,
        "pixel_gate": "EXACT",
    }
    assert protocol["renderer_qualification_boundary"] == list(TASK14R_R0_RENDERER_QUALIFICATION_BOUNDARY)
    assert protocol["source_files_sha256"] == _source_hashes()
    assert protocol["policy_query_count"] == 0
    assert protocol["formal_case_count"] == 0
    assert protocol["formal_outcome_rollout_count"] == 0
    assert not protocol["training_or_parameter_updates"]
    assert not protocol["automatic_next_phase"]


def test_all_six_continuous_dimensions_have_fixed_positive_and_negative_excitation() -> None:
    actions = np.asarray(TASK14R_R0_PROBE_ACTIONS)
    assert actions.shape == (15, 7)
    for dimension in range(6):
        assert np.any(actions[:, dimension] == 0.05)
        assert np.any(actions[:, dimension] == -0.05)
        assert set(np.abs(actions[:, dimension][actions[:, dimension] != 0.0])) == {0.05}
    assert set(actions[:, 6]) == {-1.0, 1.0}
    assert actions[0].tolist() == [0.0] * 6 + [-1.0]
    assert actions[-2].tolist() == [0.0] * 6 + [1.0]
    assert actions[-1].tolist() == [0.0] * 6 + [-1.0]


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_protocol_rejects_missing_z_roll_or_pitch_excitation(dimension: int) -> None:
    protocol = _protocol()
    for action in protocol["probe_actions"]:
        action[dimension] = 0.0
    with pytest.raises(ValueError, match=f"dimension {dimension}"):
        _validate(protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda actions: actions.pop(), "count"),
        (lambda actions: actions.__setitem__(0, actions[0][:-1]), "shape"),
        (lambda actions: actions[0].__setitem__(0, float("nan")), "finite"),
        (lambda actions: actions[0].__setitem__(0, 1.01), r"\[-1, 1\]"),
        (lambda actions: [action.__setitem__(6, -1.0) for action in actions], "gripper"),
    ],
)
def test_protocol_rejects_probe_count_shape_finiteness_range_or_gripper_drift(
    mutation: object, message: str
) -> None:
    protocol = _protocol()
    mutation(protocol["probe_actions"])
    with pytest.raises(ValueError, match=message):
        _validate(protocol)


def test_protocol_rejects_scope_schema_parent_terminal_renderer_and_source_drift() -> None:
    mutations = (
        ("schema_version", "old", "schema_version"),
        ("implementation_parent_commit", "untrusted", "implementation_parent_commit"),
        ("permitted_terminal_statuses", [TASK14R_R0_STATUS_PASSED], "permitted_terminal_statuses"),
        ("renderer_contract", {"backend": "egl", "offsamples": 4}, "renderer_contract"),
    )
    for field, value, message in mutations:
        protocol = _protocol()
        protocol[field] = value
        with pytest.raises(ValueError, match=message):
            _validate(protocol)

    protocol = _protocol()
    protocol["tasks"] = protocol["tasks"][:-1]
    protocol["task_count"] = 39
    with pytest.raises(ValueError, match="exactly 40"):
        _validate(protocol)

    protocol = _protocol()
    protocol["source_files_sha256"][TASK14R_R0_SOURCE_FILES[0]] = "b" * 64
    with pytest.raises(ValueError, match="source_files_sha256"):
        _validate(protocol)


def test_r0_seed_and_probe_sequence_are_stable() -> None:
    assert task14r_r0_seed("libero_spatial", 0) == task14r_r0_seed("libero_spatial", 0)
    assert task14r_r0_seed("libero_spatial", 0) != task14r_r0_seed("libero_spatial", 1)
    assert TASK14R_R0_PROBE_ACTIONS[0] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def _probe_step(step: int, repeat_index: int = 0) -> dict[str, object]:
    return {
        "repeat_index": repeat_index,
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


def test_probe_repeat_comparison_is_exact_and_localizes_hash_fields() -> None:
    repeats = [[_probe_step(index, repeat) for index in range(15)] for repeat in range(3)]
    exact = compare_probe_repeats(copy.deepcopy(repeats))
    assert exact["passed"]
    assert exact["first_mismatch_repeat"] is None
    assert exact["mismatched_fields"] == []

    repeats[2][4]["python_state_sha256"] = "different-python"
    repeats[2][4]["contact_state_sha256"] = "different-contact"
    mismatch = compare_probe_repeats(repeats)
    assert not mismatch["passed"]
    assert mismatch["first_mismatch_repeat"] == 2
    assert mismatch["first_mismatch_step"] == 4
    assert mismatch["mismatched_fields"] == ["python_state_sha256", "contact_state_sha256"]


def test_three_probe_traces_have_manifests_hashes_and_required_rows(tmp_path: Path) -> None:
    manifests = []
    for repeat in range(3):
        rows = [_probe_step(step, repeat) for step in range(15)]
        path = tmp_path / "probe_traces" / f"repeat{repeat:02d}.steps.jsonl.gz"
        manifests.append(write_probe_trace(path, rows))
        assert file_sha256(path) == manifests[-1]["sha256"]
        with gzip.open(path, "rt") as stream:
            assert len(stream.readlines()) == 15
    assert [row["step_count"] for row in manifests] == [15, 15, 15]
    assert len({row["sha256"] for row in manifests}) == 3


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


def _passed_tasks() -> list[dict[str, object]]:
    return [
        {
            "passed": True,
            "restore_transaction_count": 3,
            "probe_trajectory_count": 3,
            "probe_step_count": 45,
            "policy_query_count": 0,
            "formal_case_count": 0,
            "formal_outcome_rollout_count": 0,
            "training_or_parameter_updates": False,
            "automatic_next_phase": False,
        }
        for _ in range(40)
    ]


def test_r0_terminal_summary_requires_every_count_and_zero_activity_gate() -> None:
    complete = task14r_r0_terminal_summary(_passed_tasks())
    assert complete["status"] == TASK14R_R0_STATUS_PASSED
    assert complete["restore_transaction_count"] == 120
    assert complete["probe_trajectory_count"] == 120
    assert complete["probe_step_count"] == TASK14R_R0_EXPECTED_PROBE_STEP_COUNT
    assert all(complete["terminal_gate_checks"].values())

    for field in ("restore_transaction_count", "probe_trajectory_count", "probe_step_count"):
        tasks = _passed_tasks()
        tasks[0][field] -= 1
        summary = task14r_r0_terminal_summary(tasks)
        assert summary["passed_task_count"] == 40
        assert summary["status"] == TASK14R_R0_STATUS_FAILED
        assert not summary["terminal_gate_checks"][field]

    for field, value in (
        ("policy_query_count", 1),
        ("formal_case_count", 1),
        ("formal_outcome_rollout_count", 1),
        ("training_or_parameter_updates", True),
        ("automatic_next_phase", True),
    ):
        tasks = _passed_tasks()
        tasks[0][field] = value
        assert task14r_r0_terminal_summary(tasks)["status"] == TASK14R_R0_STATUS_FAILED


def test_first_restore_failure_is_counted_and_writes_task_failure_audit(tmp_path: Path) -> None:
    case = {"case_id": "r0-first", "suite": "libero_spatial", "task_id": 0, "init_state_id": 0}
    env = SimpleNamespace(close=lambda: None)

    def fail_restore(_env: object, _case: dict[str, object], _root: Path) -> dict[str, object]:
        raise Task14RStageError("complete_state_restore", "injected restore failure")

    audit = execute_task_with_audit(
        case=case,
        task_root=tmp_path / "tasks" / "r0-first",
        environment_factory=lambda: env,
        task_runner=fail_restore,
    )
    assert not audit["passed"]
    assert audit["failure_stage"] == "complete_state_restore"
    assert audit["completed_restore_transaction_count"] == 0
    assert audit["error_type"] == "Task14RStageError"
    assert audit["error"] == "injected restore failure"
    assert "injected restore failure" in audit["traceback"]
    assert audit["partial_restore_checks"] == []
    assert audit["partial_model_checks"] == []
    assert audit["partial_probe_trace_manifests"] == []
    assert audit["policy_query_count"] == 0
    assert audit["formal_outcome_rollout_count"] == 0
    assert not audit["training_or_parameter_updates"]
    assert (tmp_path / "tasks" / "r0-first" / "TASK_AUDIT_FAILURE.json").is_file()
    summary = task14r_r0_terminal_summary([audit])
    assert summary["task_count"] == 1
    assert summary["failed_task_count"] == 1
    assert summary["status"] == TASK14R_R0_STATUS_FAILED


def test_source_hash_drift_stops_at_pre_simulator_gate(tmp_path: Path) -> None:
    for relative in TASK14R_R0_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen:{relative}\n")
    hashes = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in TASK14R_R0_SOURCE_FILES
    }
    protocol = _protocol(hashes)
    assert validate_pre_simulator_gate(protocol, tmp_path) == hashes
    (tmp_path / TASK14R_R0_SOURCE_FILES[0]).write_text("drift\n")
    with pytest.raises(ValueError, match="source_files_sha256"):
        validate_pre_simulator_gate(protocol, tmp_path)
    assert not (tmp_path / "outputs").exists()


def test_r0_runner_has_one_canonical_reset_and_no_policy_training_or_later_phase_entrypoint() -> None:
    runner = SCRIPT_DIRECTORY / "run_task14r_r0.py"
    source = runner.read_text()
    tree = ast.parse(source)
    reset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
    ]
    assert len(reset_calls) == 1
    for forbidden in (
        "SmolVLA",
        "predict_action_chunk",
        "base-checkpoint",
        "snap-checkpoint",
        "optimizer",
        "phase_b",
        "phase_c",
        "task14c",
    ):
        assert forbidden.lower() not in source.lower()


def test_r0_launcher_is_fixed_fail_closed_and_policy_free() -> None:
    launcher = SCRIPT_DIRECTORY / "launch_task14r_r0.sh"
    source = launcher.read_text()
    assert "set -euo pipefail" in source
    assert "codex/task14r-r0-audit-hardening" in source
    assert "TASK14R_R0_EXPECTED_COMMIT" in source
    assert 'OUTPUT_ROOT="${OUTPUT_PARENT}/r0_attempt001"' in source
    assert "export MUJOCO_GL=egl" in source
    assert 'if [[ -e "${OUTPUT_ROOT}" ]]' in source
    assert "run_task14r_r0.py" in source
    for forbidden in ("checkpoint", "phase_b", "phase_c", "task14c", "training"):
        assert forbidden not in source.lower()
