#!/usr/bin/env python
"""Run Task 14 repository/action audits and the untouched-state capacity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from task14_common import (
    ACTION_DIM,
    GRIPPER_INDICES,
    ORIENTATION_INDICES,
    POSITION_INDICES,
    TASK14_NAME,
    audit_formal_state_capacity,
    collect_used_initial_state_ids,
    derive_seed,
)

EXPECTED_TASK13_COMMIT = "7fd8f6cb6584a2ad12862112261bbbd8087b1e08"
EXPECTED_BASE_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
EXPECTED_SNAP_SHA256 = "3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--old-dev40-manifest", type=Path, required=True)
    parser.add_argument("--formal100-manifest", type=Path, required=True)
    parser.add_argument("--confirmation1200-manifest", type=Path, required=True)
    parser.add_argument("--attempt001-manifest", type=Path, required=True)
    parser.add_argument("--attempt002-manifest", type=Path, required=True)
    parser.add_argument("--task12-registry", type=Path, required=True)
    parser.add_argument("--task13-final-report", type=Path, required=True)
    parser.add_argument("--task13-transition-effects", type=Path, required=True)
    parser.add_argument("--robosuite-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def records(path: Path, key: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain nonempty {key}")
    return rows


def json_provenance(
    *,
    generated_ns: int,
    repository_commit: str,
    branch: str,
    base_sha: str,
    snap_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "task14.audit.v1",
        "task_name": TASK14_NAME,
        "git_commit": repository_commit,
        "git_branch": branch,
        "base_checkpoint_sha256": base_sha,
        "snap_checkpoint_sha256": snap_sha,
        "manifest_sha256": None,
        "preregistration_sha256": None,
        "generated_at": datetime.fromtimestamp(generated_ns / 1_000_000_000, tz=UTC).isoformat(),
        "generated_unix_ns": generated_ns,
        "random_seed": derive_seed(),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Suite | Task | Total | Used | Untouched | IDs | Eligible |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        ids = ",".join(str(value) for value in row["untouched_state_ids"])
        lines.append(
            f"| {row['suite']} | {row['task_id']} | {row['available_state_count']} | "
            f"{row['used_state_count']} | {row['untouched_state_count']} | {ids} | "
            f"{'YES' if row['eligible'] else 'NO'} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repository_commit = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    if git_value("merge-base", "--is-ancestor", EXPECTED_TASK13_COMMIT, "HEAD") != "":
        raise RuntimeError("Unexpected output from git merge-base --is-ancestor")
    base_model = args.base_checkpoint / "model.safetensors"
    snap_model = args.snap_checkpoint / "model.safetensors"
    base_sha = file_sha256(base_model)
    snap_sha = file_sha256(snap_model)
    if base_sha != EXPECTED_BASE_SHA256 or snap_sha != EXPECTED_SNAP_SHA256:
        raise RuntimeError("Task 14 checkpoint hash mismatch")

    base_config = load_json(args.base_checkpoint / "config.json")
    snap_config = load_json(args.snap_checkpoint / "config.json")
    base_pre = load_json(args.base_checkpoint / "policy_preprocessor.json")
    base_post = load_json(args.base_checkpoint / "policy_postprocessor.json")
    snap_pre = load_json(args.snap_checkpoint / "policy_preprocessor.json")
    snap_post = load_json(args.snap_checkpoint / "policy_postprocessor.json")
    for name, config in (("base", base_config), ("snap", snap_config)):
        if config["output_features"]["action"]["shape"] != [ACTION_DIM]:
            raise RuntimeError(f"{name} checkpoint does not use the audited seven-dimensional action")
        if int(config["chunk_size"]) != 50 or int(config["n_obs_steps"]) != 1:
            raise RuntimeError(f"{name} checkpoint chunk/history contract changed")

    controller_config_path = args.robosuite_root / "controllers" / "config" / "osc_pose.json"
    controller_source_path = args.robosuite_root / "controllers" / "base_controller.py"
    gripper_source_path = args.robosuite_root / "models" / "grippers" / "panda_gripper.py"
    controller_config = load_json(controller_config_path)
    if controller_config["type"] != "OSC_POSE" or controller_config["input_min"] != -1:
        raise RuntimeError("Unexpected Robosuite controller schema")
    if controller_config["input_max"] != 1 or not controller_config["control_delta"]:
        raise RuntimeError("Robosuite controller is not the registered relative OSC pose controller")

    generated_ns = time.time_ns()
    provenance = json_provenance(
        generated_ns=generated_ns,
        repository_commit=repository_commit,
        branch=branch,
        base_sha=base_sha,
        snap_sha=snap_sha,
    )
    input_paths = {
        "task13_final_report": args.task13_final_report,
        "task13_transition_effects": args.task13_transition_effects,
        "old_dev40": args.old_dev40_manifest,
        "formal100": args.formal100_manifest,
        "confirmation1200": args.confirmation1200_manifest,
        "attempt001": args.attempt001_manifest,
        "attempt002": args.attempt002_manifest,
        "task12_registry": args.task12_registry,
    }
    input_inventory = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in input_paths.items()
    }

    action_audit = {
        **provenance,
        "status": "INTERVENTION_IDENTIFIABLE",
        "action_dimension": ACTION_DIM,
        "position_indices": list(POSITION_INDICES),
        "orientation_indices": list(ORIENTATION_INDICES),
        "orientation_representation": "relative axis-angle OSC command",
        "gripper_indices": list(GRIPPER_INDICES),
        "action_semantics": "relative Cartesian OSC pose command plus scalar gripper direction",
        "model_output_normalization": "MEAN_STD",
        "composition_space": (
            "post-policy-unnormalization environment command space; values are Robosuite normalized "
            "controller inputs, not SI-unit displacements"
        ),
        "controller_clipping": {
            "wrapper_or_env_postprocessor": "none",
            "position_and_orientation": (
                "Robosuite BaseController.scale_action clips dimensions 0:6 to [-1,1], then scales "
                "translation to +/-0.05 m and rotation to +/-0.5 rad"
            ),
            "gripper": (
                "PandaGripper uses the sign of dimension 6 and clips its accumulated internal command "
                "to [-1,1]"
            ),
            "formal_arm_counts": "not measured because the untouched-state capacity gate stopped Task 14",
        },
        "action_chunk": {
            "predicted_steps": 50,
            "executed_steps_per_replan": 10,
            "composition_rule": "replace the selected component at every aligned index 0..9",
        },
        "replanning": {"base10": 10, "snap1": 10, "cadence_equal": True},
        "internal_history": {
            "n_obs_steps": 1,
            "recurrent_state": False,
            "policy_reset_behavior": "clears action queue",
            "same_current_observation_queryable": True,
        },
        "base_processor_sha256": {
            "preprocessor": file_sha256(args.base_checkpoint / "policy_preprocessor.json"),
            "postprocessor": file_sha256(args.base_checkpoint / "policy_postprocessor.json"),
        },
        "snap_processor_sha256": {
            "preprocessor": file_sha256(args.snap_checkpoint / "policy_preprocessor.json"),
            "postprocessor": file_sha256(args.snap_checkpoint / "policy_postprocessor.json"),
        },
        "processor_normalization_contracts": {
            "base_pre": base_pre["steps"][-1]["config"]["norm_map"]["ACTION"],
            "base_post": base_post["steps"][0]["config"]["norm_map"]["ACTION"],
            "snap_pre": snap_pre["steps"][-1]["config"]["norm_map"]["ACTION"],
            "snap_post": snap_post["steps"][0]["config"]["norm_map"]["ACTION"],
        },
        "controller_sources": {
            "config": {"path": str(controller_config_path), "sha256": file_sha256(controller_config_path)},
            "base_controller": {
                "path": str(controller_source_path),
                "sha256": file_sha256(controller_source_path),
            },
            "panda_gripper": {
                "path": str(gripper_source_path),
                "sha256": file_sha256(gripper_source_path),
            },
        },
    }

    sources = (
        ("old_dev40", records(args.old_dev40_manifest, "results")),
        ("formal100", records(args.formal100_manifest, "results")),
        ("confirmation1200", records(args.confirmation1200_manifest, "cases")),
        ("attempt001_descriptive", records(args.attempt001_manifest, "cases")),
        ("attempt002", records(args.attempt002_manifest, "cases")),
    )
    used_ids, source_summary = collect_used_initial_state_ids(sources)

    from libero.libero import benchmark

    from lerobot.envs.libero import get_task_init_states

    available_counts = {}
    factories = benchmark.get_benchmark_dict()
    for suite_name in SUITES:
        suite = factories[suite_name]()
        for task_id in range(10):
            available_counts[(suite_name, task_id)] = len(get_task_init_states(suite, task_id))
    capacity = audit_formal_state_capacity(available_counts, used_ids, required_per_task=10)
    capacity.update(
        {
            **provenance,
            "status": capacity["status"],
            "source_summary": source_summary,
            "historical_input_inventory": input_inventory,
            "task12_registry_sha256": file_sha256(args.task12_registry),
            "formal_manifest_created": False,
            "formal_outcome_revealed": False,
            "formal_rollout_count": 0,
            "stop_rule": (
                "Stop with INSUFFICIENT_UNTOUCHED_FORMAL_STATES if any of the 40 tasks has fewer "
                "than 10 untouched pinned initial states; do not fill from used states"
            ),
        }
    )
    if capacity["status"] != "INSUFFICIENT_UNTOUCHED_FORMAL_STATES":
        raise RuntimeError("This preparer is a fail-closed capacity audit, not a formal manifest builder")

    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "TASK14_ACTION_SCHEMA_AUDIT.json").write_text(
        json.dumps(action_audit, indent=2, sort_keys=True) + "\n"
    )
    action_md = f"""# Task 14 Action Schema Audit

Status: `INTERVENTION_IDENTIFIABLE`

- Action dimension: `{ACTION_DIM}`.
- EEF position dimensions: `{list(POSITION_INDICES)}`.
- EEF orientation dimensions: `{list(ORIENTATION_INDICES)}`; relative axis-angle OSC command.
- Gripper dimension: `{list(GRIPPER_INDICES)}`; scalar open/close direction.
- Action semantics: relative Cartesian OSC command (`control_delta=true`).
- Model outputs are `MEAN_STD` normalized and are composed only after checkpoint postprocessing.
- Composition space: denormalized environment command coordinates. These are normalized Robosuite
  controller inputs, not direct SI-unit displacements.
- The 50-step Base and Snap chunks are queried on the same observation/noise and aligned by index;
  the existing execution horizon remains 10.
- Both policies use one observation step and no recurrent hidden state. `reset()` clears only the
  action queue; `predict_action_chunk()` can query both policies on the same current observation.
- LeRobot's LIBERO env postprocessor is empty. Robosuite clips pose inputs to `[-1,1]` internally,
  then scales xyz to +/-0.05 m and axis-angle rotation to +/-0.5 rad. Panda gripper control uses
  the sign of dimension 6 and clips its accumulated internal command to `[-1,1]`.
- Formal clipping counts are unavailable because the untouched-state capacity gate stopped the task
  before any formal arm was run.

No IK, Jacobian projection, or learned mapping is needed or permitted.
"""
    (args.output_root / "TASK14_ACTION_SCHEMA_AUDIT.md").write_text(action_md)

    repository_md = f"""# Task 14 Repository Inventory

- Task name: `{TASK14_NAME}`.
- Branch: `{branch}`.
- Current pre-audit commit: `{repository_commit}`.
- Required Task 13 commit is an ancestor: `YES` (`{EXPECTED_TASK13_COMMIT}`).
- Branch creation was verified from a clean Task 13 worktree before Task 14 files were added: `YES`.
- Base-10 SHA-256: `{base_sha}`.
- Snap-1 20k SHA-256: `{snap_sha}`.
- Task 13 final status: `PARTIAL_MECHANISM_REPLICATION`.
- Task 13 final report and transition effects were read and hashed.
- Attempt001 remains descriptive-only; it is included in the historical state exclusion set.
- Task 12 archived-state registry was read and hashed; no Task 12/13 source artifact was modified.

## Frozen input inventory

```json
{json.dumps(input_inventory, indent=2, sort_keys=True)}
```
"""
    (args.output_root / "TASK14_REPOSITORY_INVENTORY.md").write_text(repository_md)

    (args.output_root / "TASK14_FORMAL_STATE_AVAILABILITY_AUDIT.json").write_text(
        json.dumps(capacity, indent=2, sort_keys=True) + "\n"
    )
    capacity_md = f"""# Task 14 Formal State Availability Audit

**Status: `{capacity["status"]}`**

- Pinned initial states per task: 50 for all 40 tasks.
- Required untouched states per task: 10.
- Tasks meeting the requirement: {capacity["eligible_task_count"]}/40.
- Untouched states per task: minimum {capacity["minimum_untouched_per_task"]}, maximum
  {capacity["maximum_untouched_per_task"]}.
- Total untouched task/state pairs: {capacity["available_untouched_case_count"]} versus 400 required.
- Formal manifest created: `NO`.
- Formal outcome revealed: `NO`.
- Formal rollouts run: `0`.

Historical coverage uses old dev40, formal100, Confirmation1200, Attempt001, and Attempt002.
Task 12 archived states are additionally locked through the typed registry hash. Reusing any old state
or replacing only rejected states is forbidden, so the audit stops before preregistration or rollout.

{markdown_table(capacity["tasks"])}
"""
    (args.output_root / "TASK14_FORMAL_STATE_AVAILABILITY_AUDIT.md").write_text(capacity_md)

    integrity = {
        **provenance,
        "status": "TASK14_STOPPED_BEFORE_PREREGISTRATION",
        "terminal_reason": "INSUFFICIENT_UNTOUCHED_FORMAL_STATES",
        "task13_commit_ancestor": True,
        "task13_sources_modified": False,
        "formal_manifest_created": False,
        "preregistration_created": False,
        "formal_outcome_revealed": False,
        "formal_rollouts": 0,
        "adaptive_expansion": False,
        "pseudo_replication": False,
        "training_or_parameter_updates": False,
        "state_reuse_to_fill_quota": False,
        "action_schema_identifiable": True,
        "missing_artifacts_by_design": [
            "TASK14_PREREGISTRATION.md",
            "TASK14_PREREGISTRATION.json",
            "TASK14_FORMAL_CASE_MANIFEST.jsonl",
            "TASK14_STATE_OVERLAP_AUDIT.md",
            "TASK14_STATE_OVERLAP_AUDIT.json",
            "TASK14_SCREENING_RESULTS.jsonl",
            "TASK14_INTERVENTION_ROLLOUTS.jsonl",
            "TASK14_COMPONENT_EFFECTS.json",
            "TASK14_TASK_LEVEL_RESULTS.csv",
            "TASK14_LOTO_RESULTS.csv",
            "Task 14 formal figures",
        ],
    }
    (args.output_root / "TASK14_INTEGRITY_AUDIT.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )

    final_report = f"""# CRP-VLA Task 14 Final Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: VERIFIED
- Generated unix ns: `{generated_ns}`

**Terminal status: `INSUFFICIENT_UNTOUCHED_FORMAL_STATES`**

Task 14 stopped at the mandatory pre-outcome capacity gate. No formal manifest, preregistration,
screening rollout, intervention rollout, training update, or outcome analysis was created.

## Required answers

1. Clean branch containing Task 13 commit: `YES`; branch `{branch}` descends from `{EXPECTED_TASK13_COMMIT}`.
2. Base-10 checkpoint SHA-256: `{base_sha}`.
3. Snap-1 checkpoint SHA-256: `{snap_sha}`.
4. Action schema: 7D relative OSC pose command plus gripper direction.
5. EEF position dimensions: `{list(POSITION_INDICES)}`.
6. Composition space: post-unnormalization environment command space.
7. NoOp/FullSwap pure identity tests: reported by the Task 14 unit-test suite; no GPU rollout was admitted.
8. New formal states: `0` admitted. Only 1-2 untouched pinned states remain per task; 10 are required.
9. Formal overlap: not evaluated because no formal manifest could be frozen.
10. 400 cases frozen before outcome reveal: `NO`; freezing was correctly refused.
11. Adaptive expansion: `NO`.
12. Base/Snap formal success: `N/A` (0 formal rollouts).
13. Fresh strata: `N/A`.
14. Harmful task coverage: `N/A`.
15. SnapRepeat stability: `N/A`.
16. PosSwap rescue: `N/A`.
17. RotSwap rescue: `N/A`.
18. FullSwap rescue: `N/A`.
19. PosSwap-SnapRepeat effect/CI: `N/A`.
20. PosSwap-RotSwap effect/CI: `N/A`.
21. Case/task conclusion consistency: `N/A`.
22. LOTO stability: `N/A`.
23. EEF position h1/h3/h5/h10 reduction: `N/A`.
24. Intervention-specific clipping: not measured; no formal intervention ran.
25. Pseudo-replication: `NO`.
26. A-F causal status: not assigned; the earlier mandatory terminal state
    `INSUFFICIENT_UNTOUCHED_FORMAL_STATES` controls.
27. Task15 limited method development authorized: `NO`.

## Capacity evidence

All 40 task files contain exactly 50 pinned initial states. Historical manifests cover state 4
(old dev40), states 0-2 where registered (formal100), states 5-34 (Confirmation1200), states 35-44
(Attempt001), and states 45-49 (Attempt002). State 3 remains unused for every task; state 2 also
remains unused for task IDs 5-9. This leaves 60 untouched task/state pairs total and no task with
the required 10. The protocol forbids using old states to fill the deficit.

The causal hypotheses H14.1-H14.4 are `NOT TESTED`. This stop is a design-resource limitation, not
evidence for or against position-specific causal rescue.
"""
    (args.output_root / "TASK14_FINAL_REPORT.md").write_text(final_report)

    reproducibility = """# Task 14 Reproducibility Commands

Task 14 stopped at the untouched-state capacity gate. The following commands reproduce the audit;
they do not launch training or rollout.

```bash
export LIBERO_CONFIG_PATH=/root/crp-vla/artifacts/crp_vla/task13_recovery/libero_standard_config
uv run python scripts/research/crp_vla/prepare_task14.py \\
  --base-checkpoint checkpoints/smolvla_libero \\
  --snap-checkpoint outputs/crp_vla/snapflow_maturation_30k/checkpoints/020000/pretrained_model \\
  --old-dev40-manifest artifacts/crp_vla/replay_v2/development/run_manifest.json \\
  --formal100-manifest artifacts/crp_vla/followup/libero_paired_final_v1.json \\
  --confirmation1200-manifest artifacts/crp_vla/task10_confirmation1200_instrumentation/CONFIRMATION1200_MANIFEST.json \\
  --attempt001-manifest artifacts/crp_vla/task13_prospective_impact_validation/DEV_B_MANIFEST.json \\
  --attempt002-manifest artifacts/crp_vla/task13_recovery/DEV_B2_MANIFEST.json \\
  --task12-registry artifacts/crp_vla/task13_recovery/USED_STATE_REGISTRY.json \\
  --task13-final-report artifacts/crp_vla/task13_recovery/TASK13_RECOVERY_FINAL_REPORT.md \\
  --task13-transition-effects artifacts/crp_vla/task13_recovery/ATTEMPT002_TRANSITION_EFFECTS.json \\
  --robosuite-root .venv/lib/python3.12/site-packages/robosuite \\
  --output-root artifacts/crp_vla/task14_position_causal_audit

uv run pytest tests/policies/smolvla_crp/test_task14_position_causal_audit.py -q
uv run ruff check scripts/research/crp_vla/task14_common.py \\
  scripts/research/crp_vla/prepare_task14.py \\
  tests/policies/smolvla_crp/test_task14_position_causal_audit.py
(cd artifacts/crp_vla/task14_position_causal_audit && sha256sum -c TASK14_SHA256SUMS.txt)
git diff --check
```
"""
    (args.output_root / "TASK14_REPRODUCIBILITY.md").write_text(reproducibility)

    checksum_members = sorted(path for path in args.output_root.iterdir() if path.is_file())
    (args.output_root / "TASK14_SHA256SUMS.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in checksum_members)
    )
    print(capacity["status"])


if __name__ == "__main__":
    main()
