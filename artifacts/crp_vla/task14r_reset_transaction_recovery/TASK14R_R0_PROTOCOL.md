# Task14R R0 complete-state transaction qualification

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-08-18
- Verification Status: PARTIALLY_VERIFIED
- Version Label: TASK14R-RESET-TRANSACTION-R0-v2
- Implementation parent commit: `20757544607413c6ab1a3458092ceec6bc1fc85d`
- Frozen machine-readable protocol: `TASK14R_R0_PROTOCOL.json`

## Scope

R0 is a policy-free engineering gate for `TASK14R_RESET_TRANSACTION_RECOVERY`. It qualifies a
complete-state restore transaction before any replacement six-arm experiment is considered. It does not
repair or reuse Task13 Attempt002 pairs and is not Task14R Phase A attempt004.

R0 covers all 40 ordinary-LIBERO task definitions. Each task uses released state 0, one deterministic
policy-independent environment seed, one compiled MuJoCo model, three restores, and three repetitions of
the same frozen 15-action probe. No wrapper reset or hard model rebuild may occur after capsule capture.

## Frozen action probe

All six continuous OSC dimensions are excited independently in both directions at amplitude `0.05`;
all non-target continuous dimensions remain zero. Panda gripper closed/open commands are `-1.0/+1.0`.

| Step | Excitation | Action `[x,y,z,roll,pitch,yaw,gripper]` |
|---:|---|---|
| 1 | neutral, closed | `[0,0,0,0,0,0,-1]` |
| 2 | +x | `[+0.05,0,0,0,0,0,-1]` |
| 3 | -x | `[-0.05,0,0,0,0,0,-1]` |
| 4 | +y | `[0,+0.05,0,0,0,0,-1]` |
| 5 | -y | `[0,-0.05,0,0,0,0,-1]` |
| 6 | +z | `[0,0,+0.05,0,0,0,-1]` |
| 7 | -z | `[0,0,-0.05,0,0,0,-1]` |
| 8 | +roll | `[0,0,0,+0.05,0,0,-1]` |
| 9 | -roll | `[0,0,0,-0.05,0,0,-1]` |
| 10 | +pitch | `[0,0,0,0,+0.05,0,-1]` |
| 11 | -pitch | `[0,0,0,0,-0.05,0,-1]` |
| 12 | +yaw | `[0,0,0,0,0,+0.05,-1]` |
| 13 | -yaw | `[0,0,0,0,0,-0.05,-1]` |
| 14 | gripper open | `[0,0,0,0,0,0,+1]` |
| 15 | gripper close | `[0,0,0,0,0,0,-1]` |

Validation rejects a non-15-step probe, any non-seven-dimensional or non-finite action, any action outside
`[-1,1]`, a missing positive or negative excitation on dimensions 0–5, amplitude drift, or a gripper
sequence that does not contain both `-1.0` and `+1.0`.

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

## Task failure and trace evidence

Every repeat writes `probe_traces/repeatNN.steps.jsonl.gz`. Each step records repeat/step indices, the
action, integration/Python/observation/contact hashes, termination flags, and success predicate. The task
audit records every trace path, SHA-256, and step count. Exact-repeat failures identify the first mismatched
repeat and step and list the specific mismatched hash or state fields.

Any capsule, round-trip, restore, observation, controller, model, trajectory, or exact-comparison exception
produces `tasks/<case_id>/TASK_AUDIT_FAILURE.json`. It retains the failure stage, exception and traceback,
completed restore/trajectory/step counts, partial restore/model checks, and partial trace manifests. The
failed task is included in the aggregate count before R0 stops.

## Renderer contract

- backend: EGL
- offscreen samples: `0`
- pixel comparison: exact
- framebuffer context: rebuilt once after the canonical reset
- simulator/model rebuild after capsule: forbidden

Default EGL MSAA changed three wrist-camera pixels by one gray level in a compatibility diagnostic despite
exact physics and controller state. OSMesa was unavailable on the target host. No tolerance and no cached
image substitution are permitted. A single-task three-repeat, ten-step compatibility smoke passed exactly
after freezing `offsamples=0`; it used the retired v1 probe and does not qualify the v2 15-step protocol.
Full 40-task qualification remains unverified.

R0 pass qualifies only the no-MSAA complete-state transaction. It does not authorize policy execution or
establish equivalence to the standard LIBERO renderer. A separate policy-free renderer-shift audit is
required before R1.

Runtime metadata and terminal output record `MUJOCO_GL`, `CUDA_VISIBLE_DEVICES`, GPU name, NVIDIA driver,
MuJoCo and Robosuite versions, EGL vendor/version when queryable, and `offsamples`.

## Provenance gate

The v2 protocol records SHA-256 for the runner, transaction helper, and launcher. The runner recomputes
all three before creating the output root or importing/constructing a simulator and rejects any drift. It
also independently verifies the v2 schema, fixed implementation parent, permitted terminal statuses,
renderer contract, full task/action protocol, and source-file inventory; it never trusts a parent string
read from the protocol as its own expected value.

## Terminal gate

The only terminal statuses are:

- `TASK14R_R0_RESTORE_TRANSACTION_PASSED`
- `TASK14R_R0_RESTORE_TRANSACTION_FAILED`

A pass requires every gate below simultaneously:

```text
task_count = 40
passed_task_count = 40
failed_task_count = 0
restore_transaction_count = 120
probe_trajectory_count = 120
probe_step_count = 1800
policy_query_count = 0
formal_case_count = 0
formal_outcome_rollout_count = 0
training_or_parameter_updates = false
automatic_next_phase = false
```

## Frozen launcher

Do not run until the artifact-bearing commit is clean and the exact command has been reviewed. The launcher
requires the reviewed commit through `TASK14R_R0_EXPECTED_COMMIT`, creates only the parent output directory,
and requires both the attempt root and console log to be absent.

```bash
TASK14R_R0_EXPECTED_COMMIT=<REVIEWED_FINAL_COMMIT> tmux new-session -d -s task14r_r0_001 \
  'cd /root/crp-vla && export TASK14R_R0_EXPECTED_COMMIT=<REVIEWED_FINAL_COMMIT> && bash scripts/research/crp_vla/launch_task14r_r0.sh'
```

R0 cannot automatically launch the six-arm engineering run, Phase B, Phase C, Task14C, a formal outcome,
or training.
