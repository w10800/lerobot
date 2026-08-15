# CRP-VLA Task 9 Final Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED
- Version Label: crp_vla_task9_final_v1

**Final status: READY_FOR_DISJOINT_CONFIRMATION**

## VERIFIED

- Frozen selected checkpoint remains 20k with model SHA-256 `3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53`.
- 20k case-bootstrap selection frequency: `0.4916`; leave-one-case retention: `1.0`; leave-one-task retention: `1.0`.
- Development Base-10 vs 20k Snap-1 paired difference: `-0.025`; case CI `[-0.125, 0.05]`; task CI `[-0.1, 0.05]`; exact McNemar p `1.0`.
- Sample-size analysis recommends `1200` primary cases; the required default manifest remains frozen at `600` and was not automatically enlarged.
- The deterministic diagnostic subset contains `200` cases.
- Manifest overlap with formal100/dev40: `0` / `0`.
- Formal confirmation rollout has not started; no new training has started.

## INFERRED

- Under leave-one-task-out diagnostic prediction, combined disagreement AUROC/AUPRC are `0.26953125` / `0.14981340826929063` versus constant `0.0` / `0.11980862600057646`. This is predictive association, not causation.

## UNVERIFIED

- Later student-visited replans were not queryable because raw/normalized observation snapshots after the first replan were not retained. Failure-relative contact/gripper/lift/drop timing and multi-noise variance therefore remain unavailable.
- Condition-specific or grounding-specific degradation remains NOT SUPPORTED; the Formal ordinary-LIBERO gate remains FAIL; 20k remains selected for development and not confirmed.

## Final gate checks

- required_files_complete: `True`
- input_integrity: `True`
- selected_model_hash: `True`
- confirmation_manifest_frozen: `True`
- confirmation_case_count: `True`
- diagnostic_subset_count: `True`
- zero_overlap: `True`
- preflight: `True`
- formal_confirmation_not_started: `True`
- new_training_not_started: `True`
