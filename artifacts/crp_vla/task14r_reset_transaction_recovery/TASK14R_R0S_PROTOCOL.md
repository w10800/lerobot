# Task14R R0S policy-free renderer-shift audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-08-19
- Verification Status: IMPLEMENTED_NOT_EXECUTED
- Version Label: TASK14R-R0S-RENDERER-SHIFT-v2
- Implementation parent commit: `609118b7016a411c52a0ce5a6bdd8dc36ab74d79`
- Frozen machine-readable protocol: `TASK14R_R0S_PROTOCOL.json`
- Frozen digest sidecar: `TASK14R_R0S_PROTOCOL.sha256`
- Protocol SHA-256: `d45eceeffedf07edd0850c8111d489faa9c54124ae965a5568ab4f59a7f141de`
- Runner SHA-256: `a6bf4ddf208ca0951c8e346cdf58e8e3142ab1142dec3c1e1894b32e67c56b77`
- Renderer helper SHA-256: `8d9c21e763f29bbfcc8dbbf5efca3e88a7edc672f3bf518b9f0c350884056c9f`
- Frozen R0 transaction helper SHA-256: `40985f05d415a418f18d9eac6243d65a31f9a28f6a870cfa1ecde0585c79cecd`
- Launcher SHA-256: `28429c5bc419ef795384704b22a65c7c552e222aa61a668e8084c9083ecd5456`

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

Its repository commit, protocol SHA-256, three source-file SHA-256 values, LIBERO commit,
40/120/120/1800 counts, zero-activity fields, and the exact eleven-field all-true terminal-gate inventory
must match before R0S creates an output root or imports a simulator. The R0S protocol itself must match its
committed SHA-256 sidecar before the output root is created.

## Frozen paired design

For each of the 40 ordinary-LIBERO task definitions, R0S performs one canonical wrapper reset on released
state 0. No wrapper reset or model rebuild is permitted after the complete-state capsule is captured.

Each task runs three fixed repeat pairs containing `standard_msaa` (EGL `offsamples=4`) and `no_msaa`
(EGL `offsamples=0`). Pair order is frozen before execution and exactly balanced:

- even task indices: repeat 0 standard→noMSAA, repeat 1 noMSAA→standard, repeat 2 standard→noMSAA;
- odd task indices: repeat 0 noMSAA→standard, repeat 1 standard→noMSAA, repeat 2 noMSAA→standard.

Across 120 pairs, standard is first exactly 60 times and noMSAA is first exactly 60 times. No result may
change the remaining order.

Before every trajectory, R0S restores the same MuJoCo integration and Python transaction state, changes
only `sim.model.vis.quality.offsamples`, and rebuilds only the framebuffer context. It then records one
initial observation plus all 15 post-action observations. The model audit records the raw MJB and every
numeric model-array fingerprint, restores only the frozen `offsamples` whitelist field to its canonical
value, and requires the full MJB, full model fingerprint, array inventory, and every array SHA to recover
the canonical capsule exactly. Generic fingerprint normalization is not accepted as proof.

The total frozen scope is:

```text
task_count = 40
renderer_arm_count = 2
repeats_per_renderer = 3
restore_transaction_count = 240
probe_trajectory_count = 240
probe_step_count = 3600
renderer_pair_count = 120
paired_frame_count = 1920
camera_comparison_count = 3840
cross_renderer_camera_comparison_count = 3840
within_renderer_physics_frame_count = 2560
standard_repeat_camera_comparison_count = 2560
no_msaa_repeat_camera_comparison_count = 2560
pixel_comparison_count = 8960
standard_first_pair_count = 60
no_msaa_first_pair_count = 60
trace_manifest_count = 240
frame_record_manifest_count = 240
frame_payload_manifest_count = 240
pixel_comparison_manifest_count = 120
task_audit_manifest_count = 40
```

## Exact physical and non-image gates

Every within-renderer and cross-renderer frame comparison requires exact equality of:

- action and step index;
- MuJoCo integration state;
- controller, robot-buffer, wrapper, observable-timing, and non-render observation-cache state;
- formatted robot state;
- contact state;
- whitelisted-restored full model physics fingerprint;
- termination, truncation, task-success predicate, and independently recorded `info` success.

Rendered camera values alone are excluded from the Python-state identity hash. The only frozen exclusion
keys are `agentview_image` and `robot0_eye_in_hand_image`; every excluded value's path, shape, dtype, and
SHA-256 is still manifested on every frame. Unknown camera-named or multidimensional arrays remain in the
exact gate. Thus calibration arrays and EEF rotation matrices cannot be silently excluded.

The no-MSAA arm must also reproduce every camera byte exactly across its three repeats. Any physical,
non-image, model, no-MSAA repeat-pixel, early-termination, success-predicate, count, provenance, or source
failure stops at the first failing task.

## Pixel-shift classification

R0S does not choose an outcome-tuned pixel tolerance. It stores all 16 frames from both cameras for every
trajectory in lossless compressed NPZ payloads and records their SHA-256. All equality verdicts use raw
`uint8` and only `np.array_equal`; tolerance is null. Every comparison row is keyed by task, repeat,
frame/step, and camera and reports both image hashes, changed pixels, changed channel values, maximum and
mean absolute difference, per-channel changed counts, a difference bounding box, the first differing
pixel/channel, reference and actual pixel/channel values, and absolute/squared-difference totals. RMS is
reported in the frozen aggregate summaries.

The three classes are never conflated: standard-MSAA within-repeat variation, no-MSAA within-repeat
variation, and paired standard-vs-noMSAA difference have separate manifests and summaries. No-MSAA must
remain byte-exact. Standard-within and cross-render variation are reported as distinct terminal counts.

The only complete terminal classifications are:

- `TASK14R_R0S_RENDERERS_EXACTLY_EQUIVALENT`: all technical gates pass and every standard repeat and
  standard-vs-no-MSAA camera comparison is byte-exact;
- `TASK14R_R0S_RENDERER_SHIFT_CHARACTERIZED`: all technical gates pass, physics and non-image state are
  exact, no-MSAA repeats are exact, but at least one standard repeat or cross-render camera differs;
- `TASK14R_R0S_RENDERER_AUDIT_FAILED`: any technical, count, provenance, or zero-activity gate fails.

`RENDERER_SHIFT_CHARACTERIZED` is not an equivalence claim and does not authorize R1. R0S does not
establish policy-input or policy-output equivalence. A separately frozen policy-impact bridge is required
before any policy execution. Even an exact renderer result cannot automatically launch or authorize R1.

## Evidence artifacts

Each task writes six trajectory artifact triplets:

```text
probe_traces/<renderer>.repeatNN.steps.jsonl.gz
frame_records/<renderer>.repeatNN.frames.jsonl.gz
frame_payloads/<renderer>.repeatNN.frames.npz
```

It also writes three task-level comparison manifests:

```text
pixel_comparisons/standard_msaa_within_repeat.jsonl.gz
pixel_comparisons/no_msaa_within_repeat.jsonl.gz
pixel_comparisons/standard_msaa_vs_no_msaa.jsonl.gz
```

The task audit records each path, SHA-256, step/frame/comparison count, restore checks, within-render
comparisons, cross-render comparisons, and per-camera aggregates. Every file is reopened and rehashed
after the task and again before finalization. The 16 frame-state records are matched to the 15 action-step
records and lossless NPZ pixels; every comparison row is independently recomputed from its referenced NPZ
frames. Missing, corrupt, semantically inconsistent, or merely hash-consistent fabricated evidence fails.
A failure writes `TASK_AUDIT_FAILURE.json` with renderer arm, repeat, frame/step, camera, first mismatched
field, exception, traceback, executed and recorded partial counts, and completed or partial
trace/frame-record/frame-payload/comparison manifests.

## Terminal zero-activity gates

All complete statuses require:

```text
policy_query_count = 0
formal_case_count = 0
formal_outcome_rollout_count = 0
training_or_parameter_updates = false
automatic_next_phase = false
r1_authorized = false
policy_impact_bridge_required = true
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
