#!/usr/bin/env python
"""Freeze policy-independent Task14R state seeds and distribution-audit rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from replay_v2_common import array_sha256
from run_libero_paired_four_arm_pilot import configure_standard_libero, git_repository_value
from task14r_common import (
    TASK14R_CANDIDATES_PER_TASK,
    TASK14R_CAPACITY_PROBE_COUNT,
    TASK14R_CAPACITY_REQUIRED_VALID,
    TASK14R_FORMAL_PER_TASK,
    TASK14R_NAME,
    TASK14R_STATE_SEED_STRING,
    structured_seed,
)

EXPECTED_TASK14_COMMIT = "54fee829ba0b180c53a85b7a9f56b24fbe5ae768"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--used-state-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--runtime-config-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.source_audit.exists():
        raise FileExistsError("Task14R state-bank freeze outputs already exist")
    if git_value("status", "--porcelain"):
        raise RuntimeError("Repository must be clean before freezing Task14R state seeds")
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_TASK14_COMMIT, "HEAD"], check=True)
    configure_standard_libero(args.libero_root, args.runtime_config_dir)

    import libero.libero as libero_module
    from libero.libero import benchmark, get_libero_path

    from lerobot.envs.libero import get_task_init_states

    assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    libero_module._assets_path_cache = str(assets)
    factories = benchmark.get_benchmark_dict()
    candidates = []
    capacity_probes = []
    official_reference = []
    tasks = []
    bddl_root = Path(get_libero_path("bddl_files"))
    for suite_name in SUITES:
        suite = factories[suite_name]()
        for task_id in range(10):
            task = suite.get_task(task_id)
            task_key = f"{suite_name}:{task_id}"
            bddl_path = bddl_root / task.problem_folder / task.bddl_file
            states = get_task_init_states(suite, task_id)
            if len(states) != 50:
                raise RuntimeError(f"Expected exactly 50 official states for {task_key}")
            official_hashes = [array_sha256(state) for state in states]
            if len(set(official_hashes)) != 50:
                raise RuntimeError(f"Official state payload duplication detected for {task_key}")
            tasks.append(
                {
                    "task_key": task_key,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_instruction": task.language,
                    "bddl_path": portable_path(bddl_path),
                    "bddl_sha256": file_sha256(bddl_path),
                    "official_state_count": 50,
                }
            )
            official_reference.append(
                {
                    "task_key": task_key,
                    "official_raw_state_sha256": official_hashes,
                }
            )
            for candidate_index in range(TASK14R_CANDIDATES_PER_TASK):
                candidates.append(
                    {
                        "state_id": f"task14r-{suite_name}-task{task_id:02d}-candidate{candidate_index:02d}",
                        "task_key": task_key,
                        "suite": suite_name,
                        "task_id": task_id,
                        "candidate_index": candidate_index,
                        "seed": structured_seed(suite_name, task_id, candidate_index),
                    }
                )
            if task_id == 0:
                for probe_index in range(TASK14R_CAPACITY_PROBE_COUNT):
                    capacity_probes.append(
                        {
                            "state_id": f"task14r-{suite_name}-task00-capacity-probe{probe_index:02d}",
                            "task_key": task_key,
                            "suite": suite_name,
                            "task_id": task_id,
                            "probe_index": probe_index,
                            "seed": structured_seed(
                                suite_name, task_id, probe_index, namespace="capacity_probe"
                            ),
                            "eligible_for_formal_bank": False,
                        }
                    )

    if len(tasks) != 40 or len(candidates) != 800 or len(capacity_probes) != 256:
        raise RuntimeError("Task14R frozen design cardinality drift")

    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    source_files = {
        "env_wrapper": args.libero_root / "libero" / "libero" / "envs" / "env_wrapper.py",
        "bddl_base_domain": args.libero_root / "libero" / "libero" / "envs" / "bddl_base_domain.py",
        "region_sampler": args.libero_root
        / "libero"
        / "libero"
        / "envs"
        / "regions"
        / "base_region_sampler.py",
        "lerobot_libero": Path("src/lerobot/envs/libero.py"),
    }
    payload = {
        "schema_version": "task14r.state_seed_manifest.v1",
        "task_name": TASK14R_NAME,
        "phase": "PHASE_B_POLICY_INDEPENDENT_STATE_GENERATION",
        "status": "TASK14R_STATE_SEEDS_FROZEN",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": False,
        "libero_commit": libero_commit,
        "generated_unix_ns": time.time_ns(),
        "seed_string": TASK14R_STATE_SEED_STRING,
        "outcomes_accessed": False,
        "policy_query_count": 0,
        "policy_or_checkpoint_inputs": [],
        "task_count": len(tasks),
        "candidates_per_task": TASK14R_CANDIDATES_PER_TASK,
        "candidate_count": len(candidates),
        "formal_states_per_task": TASK14R_FORMAL_PER_TASK,
        "formal_case_count": 40 * TASK14R_FORMAL_PER_TASK,
        "capacity_probe": {
            "tasks": [f"{suite}:0" for suite in SUITES],
            "candidate_states_per_probe_task": TASK14R_CAPACITY_PROBE_COUNT,
            "required_valid_states_per_probe_task": TASK14R_CAPACITY_REQUIRED_VALID,
            "total_state_count": len(capacity_probes),
            "formal_bank_eligible": False,
            "pass_rule": (
                "at least 51 unique, serializable, reproducible accepted states among the 64 "
                "frozen seeds for every representative probe task"
            ),
        },
        "selection_rule": (
            "For each task, accept only frozen policy-independent validity checks; sort accepted "
            "candidates by raw_state_sha256 then candidate_index; first 10 are formal and all "
            "remaining accepted states are reserve. Reserve is never activated adaptively."
        ),
        "acceptance_checks": {
            "simulator_loadable": "the task environment resets, restores, settles, and serializes",
            "finite_observation": "all numeric observation and simulator arrays are finite",
            "legal_initial_predicates": "every BDDL initial predicate remains true after 10 settle steps",
            "not_initially_successful": "the task success predicate is false before and after settling",
            "no_severe_penetration_or_explosion": (
                "minimum contact distance >= -0.01 m, maximum absolute qvel <= 50, and maximum "
                "object settling displacement <= 0.20 m"
            ),
            "reset_reproducible": (
                "two independent full resets plus raw-state restoration and 10 settle steps have "
                "identical physical-state hashes; render hashes are recorded but not required"
            ),
            "zero_historical_overlap": (
                "raw and settled hashes do not match any official fixed state, typed used-state "
                "registry identity, or earlier generated/probe payload"
            ),
            "scene_task_match": "suite/task/BDDL hash matches the frozen manifest",
        },
        "distribution_audit_rule": {
            "reference_states_per_task": 50,
            "formal_generated_states_per_task": 10,
            "feature_families": [
                "robot_joint_pos",
                "eef_position",
                "eef_orientation",
                "object_position",
                "object_orientation",
                "object_pair_distance",
                "target_region_distance",
                "contact",
                "settling_displacement",
            ],
            "expanded_support_margin": "max(family absolute margin, 10% of official range)",
            "task_family_mismatch": (
                "more than 2/10 formal states outside expanded support OR median standardized "
                "feature-mean shift greater than 1.0"
            ),
            "terminal_mismatch": "at least 4/40 mismatched tasks in any feature family",
        },
        "used_state_registry": {
            "path": portable_path(args.used_state_registry),
            "sha256": file_sha256(args.used_state_registry),
        },
        "source_files": {
            name: {"path": portable_path(path), "sha256": file_sha256(path)}
            for name, path in source_files.items()
        },
        "tasks": tasks,
        "official_reference": official_reference,
        "candidates": candidates,
        "capacity_probes": capacity_probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    source_audit = f"""# Task14R Initial-State Generation Source Audit

Status: `PARTIALLY_VERIFIED_GENERATION_PROTOCOL`

## VERIFIED from pinned code

- The pinned LIBERO commit is `{libero_commit}`.
- Each standard task exposes exactly 50 serialized `.pruned_init` states.
- `BDDLBaseDomain.seed()` sets the global NumPy RNG.
- BDDL object placement uses region-specific uniform samplers and can reject colliding placements;
  the core samplers allow up to 5,000 placement attempts before `RandomizationError`.
- The current Robosuite configuration receives `initialization_noise=None`, so robot joint reset noise
  has zero magnitude and qpos is the robot model default.
- The LeRobot evaluation wrapper restores a serialized state and then applies exactly 10 dummy-action
  settle steps.

## VERIFIED from an official maintainer statement

The LIBERO maintainers state that fixed init states were collected by initializing customized
environments repeatedly with different random seeds and recording simulator states:
https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34

## UNVERIFIED

- the original per-task seed list;
- whether the released files were captured before or after settle steps;
- the original pruning/rejection criteria beyond sampler-level collision rejection;
- whether any manual task-specific filtering was used;
- exact correspondence between the current pinned fork and the environment version used to create
  the released fixed states.

Therefore Task14R outputs must be called a **simulator-generated evaluation distribution**, never
new official LIBERO test states. Phase C decides only whether the generated distribution is
reasonably supported by the released 50-state reference under a frozen internal audit rule.
"""
    args.source_audit.write_text(source_audit)
    print(json.dumps({"status": payload["status"], "candidates": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
