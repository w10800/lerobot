# Task14R Initial-State Generation Source Audit

Status: `PARTIALLY_VERIFIED_GENERATION_PROTOCOL`

## VERIFIED from pinned code

- The pinned LIBERO commit is `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`.
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
