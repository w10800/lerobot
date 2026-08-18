# Task14R R0 complete-state transaction qualification

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-08-18
- Verification Status: PARTIALLY_VERIFIED
- Version Label: TASK14R-RESET-TRANSACTION-R0-v1
- Implementation parent commit: `5b426ea50d9a20e56ed6f118bc08554b0b452b47`
- Frozen machine-readable protocol: `TASK14R_R0_PROTOCOL.json`

## Scope

R0 is a policy-free engineering gate for `TASK14R_RESET_TRANSACTION_RECOVERY`. It qualifies a
complete-state restore transaction before any replacement six-arm experiment is considered. It does not
repair or reuse Task13 Attempt002 pairs and is not Task14R Phase A attempt004.

R0 covers all 40 ordinary-LIBERO task definitions. Each task uses released state 0, one deterministic
policy-independent environment seed, one compiled MuJoCo model, three restores, and three repetitions of
the same frozen ten-action probe. No wrapper reset or hard model rebuild may occur after capsule capture.

## Complete-state capsule

The capsule contains or verifies:

- compiled MJB SHA-256 and every exposed numeric model-array hash;
- MuJoCo `mjSTATE_INTEGRATION` state;
- OSC controller and interpolator state, including reconstructed derived kinematics/dynamics;
- Panda gripper `current_action`;
- Robosuite robot history buffers;
- wrapper time, timestep, done, action-dimension, and initial simulator state;
- observation cache and every observable's sampling/value state;
- regenerated raw observation and initial task-success predicate.

Opaque state, schema drift, model mutation, a mismatched observation, an early probe termination, or a
probe success predicate stops the run.

## Renderer contract

- backend: EGL
- offscreen samples: `0`
- pixel comparison: exact
- framebuffer context: rebuilt once after the canonical reset
- simulator/model rebuild after capsule: forbidden

Default EGL MSAA changed three wrist-camera pixels by one gray level in a compatibility diagnostic despite
exact physics and controller state. OSMesa was unavailable on the target host. No tolerance and no cached
image substitution are permitted. A single-task three-repeat, ten-step compatibility smoke passed exactly
after freezing `offsamples=0`; full 40-task qualification remains unverified.

## Terminal gate

The only terminal statuses are:

- `TASK14R_R0_RESTORE_TRANSACTION_PASSED`
- `TASK14R_R0_RESTORE_TRANSACTION_FAILED`

A pass requires:

```text
task_count = 40
passed_task_count = 40
failed_task_count = 0
restore_transaction_count = 120
probe_trajectory_count = 120
policy_query_count = 0
formal_case_count = 0
formal_outcome_rollout_count = 0
training_or_parameter_updates = false
automatic_next_phase = false
```

## Frozen launcher

Do not run until the artifact-bearing commit is clean and the exact command has been reviewed. The launcher
creates only the parent output directory and requires both the attempt root and console log to be absent.

```bash
tmux new-session -d -s task14r_r0_001 'cd /root/crp-vla && bash scripts/research/crp_vla/launch_task14r_r0.sh'
```

R0 cannot automatically launch the six-arm engineering run, Phase B, Phase C, Task14C, a formal outcome,
or training.
