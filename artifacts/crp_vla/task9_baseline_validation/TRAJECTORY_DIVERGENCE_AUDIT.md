# Trajectory Divergence Audit

## VERIFIED

All 40 cases provide a strictly matched initial replan across Base-10 and 20k Snap-10/2/1. The outcome-blind 90th-percentile D_1_10 threshold is `0.380696`.

## INFERRED

Leave-one-task-out diagnostic prediction was computed from initial-replan disagreement only. Any discrimination is predictive association under this protocol, not a causal mechanism.

## UNVERIFIED

Later student-visited replans cannot be queried because Task 7/8 traces retain hashes and action chunks but not raw/normalized observation snapshots after the initial replan. Therefore pre-contact, pre-gripper-close, lift/drop timing and multi-noise variance are unavailable and were not inferred from RGB.

The statement “disagreement causes grasp failure” is not supported.
