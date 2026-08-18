# Task 14 Action Schema Audit

Status: `INTERVENTION_IDENTIFIABLE`

- Action dimension: `7`.
- EEF position dimensions: `[0, 1, 2]`.
- EEF orientation dimensions: `[3, 4, 5]`; relative axis-angle OSC command.
- Gripper dimension: `[6]`; scalar open/close direction.
- Action semantics: relative Cartesian OSC command (`control_delta=true`).
- Model outputs are `MEAN_STD` normalized and are composed only after checkpoint postprocessing.
- Composition space: denormalized environment command coordinates. These are normalized Robosuite
  controller inputs, not direct SI-unit displacements.
- The 50-step Base and Snap chunks are queried on the same observation/noise and aligned by index;
  the existing execution horizon remains 10.
- Both policies use one observation step and no recurrent hidden state. `reset()` clears only the
  action queue; `predict_action_chunk()` can query both policies on the same current observation.
- LeRobot's LIBERO env postprocessor is empty. Robosuite clips pose inputs to `[-1,1]` internally,
  then scales xyz to +/-0.05 m and axis-angle rotation to +/-0.5 rad. Panda gripper control uses
  the sign of dimension 6 and clips its accumulated internal command to `[-1,1]`.
- Formal clipping counts are unavailable because the untouched-state capacity gate stopped the task
  before any formal arm was run.

No IK, Jacobian projection, or learned mapping is needed or permitted.
