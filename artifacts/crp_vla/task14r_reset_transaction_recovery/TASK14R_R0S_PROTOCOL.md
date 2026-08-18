# Task14R R0S policy-free renderer-shift audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-08-18
- Verification Status: IMPLEMENTED_NOT_EXECUTED
- Version Label: TASK14R-R0S-RENDERER-SHIFT-v1
- Implementation parent commit: `60fbf7370d8668316b34a9d0d1a334247f792cca`
- Frozen machine-readable protocol: `TASK14R_R0S_PROTOCOL.json`

## Purpose and boundary

R0S is the sole policy-free gate between the passed no-MSAA R0 complete-state transaction and any
future R1 proposal. It compares the standard LIBERO EGL renderer (`offsamples=4`) with the qualified
no-MSAA renderer (`offsamples=0`) while holding the compiled model, complete simulator state,
controller/wrapper state, task, seed, and fixed action probe constant.

R0S loads no Base or Snap checkpoint, performs no policy query, reveals no formal outcome, and makes no
training or parameter update. It does not reuse the outcome-conditioned Task13 Attempt002 pairs. It uses
the same 40 task/state/seed definitions and 15 fixed policy-free actions as R0.

The required R0 terminal artifact is:

```text
outputs/crp_vla/task14r_reset_transaction_recovery/
  r0_attempt001/TASK14R_R0_FINAL.json
SHA-256: ae443481ec991f6c46cb74a8bf4adf90817cd528d5f1ff4e44d04f6727df1a3e
status: TASK14R_R0_RESTORE_TRANSACTION_PASSED
```

Its repository commit, all 40/120/120/1800 counts, all zero-activity fields, and every terminal gate must
match before R0S creates an output root or imports a simulator.

## Frozen paired design

For each of the 40 ordinary-LIBERO task definitions, R0S performs one canonical wrapper reset on released
state 0. No wrapper reset or model rebuild is permitted after the complete-state capsule is captured.

Each task runs three fixed repeat pairs. Every pair contains:

1. `standard_msaa`, EGL `offsamples=4`;
2. `no_msaa`, EGL `offsamples=0`.

Before every trajectory, R0S restores the same MuJoCo integration and Python transaction state, changes
only `sim.model.vis.quality.offsamples`, and rebuilds only the framebuffer context. It then records one
initial observation plus all 15 post-action observations. The normalized model fingerprint is computed
with only `offsamples` temporarily normalized, so every other compiled-model field remains an exact gate.

The total frozen scope is:

```text
task_count = 40
restore_transaction_count = 240
probe_trajectory_count = 240
probe_step_count = 3600
renderer_pair_count = 120
paired_frame_count = 1920
camera_comparison_count = 3840
within_renderer_physics_frame_count = 2560
standard_repeat_camera_comparison_count = 2560
no_msaa_repeat_camera_comparison_count = 2560
```

## Exact physical and non-image gates

Every within-renderer and cross-renderer frame comparison requires exact equality of:

- action and step index;
- MuJoCo integration state;
- controller, robot-buffer, wrapper, observable-timing, and non-render observation-cache state;
- formatted robot state;
- contact state;
- termination, truncation, and task-success predicate.

Rendered camera values alone are excluded from the Python-state identity hash. The exclusion is name-bound
to camera/image/RGB observables; multidimensional robot state such as the EEF rotation matrix remains in
the exact gate.

The no-MSAA arm must also reproduce every camera byte exactly across its three repeats. Any physical,
non-image, model, no-MSAA repeat-pixel, early-termination, success-predicate, count, provenance, or source
failure stops at the first failing task.

## Pixel-shift classification

R0S does not choose an outcome-tuned pixel tolerance. It stores all 16 frames from both cameras for every
trajectory in lossless compressed NPZ payloads and records their SHA-256. For every paired camera frame it
reports exactness, changed channel values, changed pixels, maximum absolute difference, total absolute and
squared difference, mean absolute difference, and root-mean-squared difference.

The only complete terminal classifications are:

- `TASK14R_R0S_RENDERERS_EXACTLY_EQUIVALENT`: all technical gates pass and every standard repeat and
  standard-vs-no-MSAA camera comparison is byte-exact;
- `TASK14R_R0S_RENDERER_SHIFT_CHARACTERIZED`: all technical gates pass, physics and non-image state are
  exact, no-MSAA repeats are exact, but at least one standard repeat or cross-render camera differs;
- `TASK14R_R0S_RENDERER_AUDIT_FAILED`: any technical, count, provenance, or zero-activity gate fails.

`RENDERER_SHIFT_CHARACTERIZED` is not an equivalence claim. Even an exact result does not automatically
launch or authorize R1. A human must review the complete artifact and issue a separate authorization.

## Evidence artifacts

Every trajectory writes:

```text
probe_traces/<renderer>.repeatNN.steps.jsonl.gz
frame_payloads/<renderer>.repeatNN.frames.npz
pixel_comparisons/renderer_pixel_comparisons.jsonl.gz
```

The task audit records each path, SHA-256, step/frame count, restore checks, within-render comparisons,
cross-render comparisons, and per-camera aggregates. A failure writes `TASK_AUDIT_FAILURE.json` with the
failure stage, exception, traceback, partial counts, and any completed trace/payload manifests.

## Terminal zero-activity gates

All complete statuses require:

```text
policy_query_count = 0
formal_case_count = 0
formal_outcome_rollout_count = 0
training_or_parameter_updates = false
automatic_next_phase = false
```

R0S cannot enter R1, Phase A/B/C, Task14C, renderer-conditioned policy evaluation, formal outcomes,
training, checkpoint selection, PR creation, or merge.

## Frozen launcher

The launcher requires the reviewed artifact-bearing commit in `TASK14R_R0S_EXPECTED_COMMIT`, the exact
branch and parent, a clean worktree, the passed R0 artifact, pinned LIBERO, and absent attempt root and
console log. It creates only the output parent before invoking the runner.

```bash
tmux new-session -d -s task14r_r0s_001 \
  'cd /root/crp-vla && \
  export TASK14R_R0S_EXPECTED_COMMIT=<REVIEWED_FINAL_COMMIT> && \
  bash scripts/research/crp_vla/launch_task14r_r0s.sh'
```

This command is for later manual review. Freezing the protocol does not authorize executing it.
