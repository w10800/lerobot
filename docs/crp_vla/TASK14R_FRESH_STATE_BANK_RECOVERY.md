# TASK14R_FRESH_STATE_BANK_RECOVERY

Status: `IMPLEMENTATION_AND_PREFLIGHT_ONLY`

This recovery task is not Task15 and does not reveal a new formal Base/Snap outcome. It has only two
purposes: validate the component-intervention implementation on already revealed Attempt002 cases,
and freeze a fresh policy-independent simulator-state bank for a later separately authorized audit.

## Evidence boundary

Phase A uses exactly the 15 Attempt002 `harmful` cases and is permanently labeled:

- `ENGINEERING_ONLY`
- `OUTCOME_CONDITIONED`
- `NON_PRIMARY`

It runs `BaseRepeat`, `SnapRepeat`, `NoOp`, `PosSwap`, `RotSwap`, and `FullSwap`. At every replan,
both policies query the current observation actually visited by that intervention arm with the same
frozen Attempt002 noise tensor. Reading cached original-rollout actions is forbidden. Each simulator
step records both policy actions, the composed action, clipping before/after, observation and physical
state hashes, chunk index, replan index, and termination.

The engineering gates are exact stepwise closed-loop identity for `NoOp == SnapRepeat` and
`FullSwap == BaseRepeat`, including actions, physical state, action-queue alignment, and termination.
Failures remain failures and must be localized to policy noise, queue alignment, online observation,
reset completeness, action-chunk alignment, controller clipping, or simulator nondeterminism. A
successful `PosSwap` on this set is not paper evidence.

## Fresh-state generation

Phase B imports no policy or checkpoint. The frozen seed string is
`CRP-VLA-TASK14R-FRESH-STATE-BANK-V1`. For each of the 40 standard tasks it generates exactly 20
candidates from the pinned task BDDL and placement samplers. Acceptance is the conjunction of only:

- simulator load/serialization;
- finite observations and simulator arrays;
- all BDDL initial predicates before and after ten no-op settle steps;
- task success false before and after settling;
- minimum contact distance at least `-0.01 m`, maximum absolute qvel at most `50`, and maximum
  object settling displacement at most `0.20 m`;
- identical physical-state hashes across two full reset/restore/settle trials;
- no official, historical-registry, or generated payload overlap;
- frozen suite/task/BDDL identity.

Base/Snap success, action discrepancy, EEF discrepancy, harmful status, and any policy score are
forbidden selection fields. Accepted candidates are sorted by raw-state SHA-256 and candidate index;
the first 10 per task are formal (`400` total), and remaining accepted candidates are reserve. Reserve
states cannot be activated according to observed harmful counts.

A separate capacity probe uses 64 frozen seeds on task 0 of each suite and requires at least 51 unique,
serializable, physically reproducible accepted states. These probes are never formal-bank candidates.

## Source-audit boundary

The pinned implementation verifies that BDDL placement uses seeded NumPy sampling, region samplers,
collision-aware rejection, fixed robot qpos when initialization noise is disabled, MuJoCo state
serialization/restoration, and ten wrapper settle steps. An official maintainer describes the released
states as repeated seeded environment initialization followed by state recording:
<https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34>.

The original seed list, capture timing, pruning criteria, and any manual filtering of the released 50
states are not available. Generated states are therefore called a **simulator-generated evaluation
distribution**, never new official LIBERO test states.

## Distribution audit and terminal states

Phase C compares the selected 10 generated states with all 50 released states for each task using robot
joint position, EEF position/orientation, object position/orientation, pairwise object distance, target
region distance, contact state, BDDL initial predicates, and settling displacement.

For each feature, official support is expanded by the larger of the registered physical margin and 10%
of the official range. A task/family mismatches if more than 2 of 10 states are outside expanded support
or its median standardized feature-mean shift exceeds 1.0. A terminal mismatch requires at least 4 of
40 task mismatches in any family.

- `GENERATED_STATE_DISTRIBUTION_MISMATCH`: do not call the bank official or launch the causal audit.
- `READY_FOR_TASK14_CAUSAL_RELAUNCH`: Phase A gates passed, capacity passed, 400 states are frozen,
  reproducible and disjoint, and Phase C passed.
- all other cases: `TASK14R_RECOVERY_INCOMPLETE` or the more specific engineering/capacity failure.

Even `READY_FOR_TASK14_CAUSAL_RELAUNCH` does not execute formal outcomes. A separate task named
`TASK14C_PROSPECTIVE_POSITION_CAUSAL_AUDIT` must freeze and authorize those rollouts. No training,
checkpoint selection, Task15 method work, or adaptive reduction from 400 to 60 is permitted here.
