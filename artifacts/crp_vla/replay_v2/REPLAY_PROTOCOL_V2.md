# Replayable Paired Closed-Loop Protocol v2

## Status and boundary

This protocol is frozen for ordinary factual LIBERO diagnostic/development evaluation. It does not modify the original 100-case Formal LIBERO Gate FAIL, does not reuse its image hashes, is not a confirmation protocol, and does not authorize CRP training.

## Material passport

- Phase A design: `configs/crp_vla/replay_v2_smoke.json` (8 cases; two per suite; init state 3).
- Phase B design: `configs/crp_vla/replay_v2_development.json` (40 cases; all 40 tasks; init state 4).
- Phase A is infrastructure-only and is excluded from scientific statistics.
- Phase B may start only after all 48 Phase A rollouts pass the admission checks.
- The old formal cases and manifest remain immutable. Replay-v2 uses newly captured raw observations and canonical policy tensors.

## Episode capsule

Every case is stored as a directory containing `metadata.json`, `arrays.npz`, and `sha256_manifest.txt`. The compressed array payload contains the full MuJoCo simulator state, qpos, qvel, all initial RGB and robot-state arrays exposed to the policy, the canonical processed policy batch, and the entire pre-generated replan-noise schedule. Metadata records suite/task/instruction, BDDL hash and parsed task semantics, camera parameters, renderer/EGL fingerprint, processor/normalizer hashes, evaluator hash, execution horizon, maximum length, repository/environment/checkpoint revisions, and CUDA/PyTorch/MuJoCo provenance.

Loading a capsule verifies every member hash, array dtype, shape, and content hash. The first policy call is reconstructed from the stored canonical policy batch; it is not re-rendered. Subsequent inputs come from each arm's own trajectory.

## Same-process paired execution

Each capsule is executed in this fixed order in one worker process:

1. Base-10
2. Base-2
3. Base-1
4. Snap-10
5. Snap-2
6. Snap-1

Before each arm, the environment and policy state are reset, the exact capsule simulator state is restored, and the capsule noise-table prefix is used. Policy caches and action queues are never shared across arms. Each case is admitted only when all six records agree on capsule, simulator-state, canonical-input, noise-schedule, and evaluator hashes and all six complete. A pre-rollout infrastructure failure may be recorded for all six arms; a partial/mixed case is not admitted and Phase A/B fails. Model failures are never excluded after outcomes are observed.

## Instrumentation and storage

Every step records wall timestamp, observation hashes, processed-input hash, action-chunk reference, executed action, end-effector pose, gripper command/state, object-of-interest poses, contact pairs, registered collision evidence, success predicate, termination reason, NFE, latency, and accumulated first-contact/close/lift/receptacle events. Full predicted chunks and executed actions are stored in compressed numeric trace files.

Capsules and full step traces are large ignored artifacts. Git contains protocol/design/code, run and capsule/trace SHA manifests, compact event tables, per-case results, and summaries; it never contains checkpoints, optimizer states, datasets, videos, or the large trajectory payload.

## Failure classification

Allowed labels are `WRONG_TARGET`, `WRONG_RELATION`, `GRASP_FAILURE`, `GRIPPER_TIMING`, `TRAJECTORY_COLLISION`, `OVERSHOOT`, `OBJECT_DROPPED`, `RECEPTACLE_FAILURE`, and `UNKNOWN`. A non-UNKNOWN label requires its machine-readable event evidence. Ambiguous semantic relations, heuristics without a direct event, and unavailable task signals remain `UNKNOWN`; missing evidence is never imputed.

## Frozen analyses

Phase B reports overall and suite-stratified six-arm success, paired differences, two-sided exact McNemar tests, paired-case bootstrap intervals, task-cluster bootstrap intervals, 10→2/2→1/10→1 decompositions, and latency/success points. The initial common-plan executed-prefix MSE and gripper class/timing disagreement are linked descriptively to rollout failure. These associations are diagnostic, not causal and not checkpoint-selection criteria unless separately preregistered.
