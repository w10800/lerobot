# Disjoint Confirmation Protocol

## Frozen prospective design

- Primary endpoint: `SR(20k Snap-1) - SR(Base-10)`.
- Non-inferiority margin: `-0.03`; neither formal100 nor dev40 may redefine it.
- Primary arms: Base-10 and frozen 20k Snap-1.
- Diagnostic arms: Base-2, Base-1, 20k Snap-10, 20k Snap-2 on the frozen diagnostic subset only.
- Every paired arm shares the exact initial simulator state, raw and normalized observation, condition, proprioception, processor contract, noise tensor/schedule, action normalization, execution horizon, evaluator, and runtime.
- CRP/CAG/KD/new-method training remains HOLD. No confirmation rollout is authorized by Task 9.

## Failure/invalid rerun rule

Only infrastructure failures documented without outcome access may be rerun. The same case, arm, state, noise schedule, evaluator, and output namespace must be retained; both failed attempt metadata and replacement attempt metadata remain immutable. Scientific failures are never rerun.
