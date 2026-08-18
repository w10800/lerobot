# CRP-VLA project charter

Status: active research specification  
Evidence status: mixed; hypotheses and proposed thresholds are `UNVERIFIED` until local experiments run  
Effective date: 2026-08-13

## Research question

When the same flow-matching VLA is compressed from a 10-NFE action generator to a 1-NFE generator, does it lose additional conditional response to language, target objects, or spatial relations beyond what ordinary action MSE and standard LIBERO success reveal?

## Falsifiable claims

1. Compression creates a measurable conditional-response gap between the 10-step teacher and the 1-step student.
2. The gap is not fully explained by factual action error or ordinary closed-loop success.
3. Standard one-step distillation does not explicitly preserve finite-difference response to a controlled condition intervention.
4. CRP improves response preservation while retaining 1-NFE latency and acceptable factual-task capability.

Claims 1–4 are hypotheses, not findings. A negative result is a valid project outcome.

## Fixed experimental unit

For a factual condition `c` and counterfactual condition `c_tilde`, both branches use identical observation, state, preprocessing, initial action noise, checkpoint, NFE, execution horizon, and normalization. Only the condition changes.

Teacher and student responses are:

```text
delta_T = A_teacher(z, observation, c) - A_teacher(z, observation, c_tilde)
delta_S = A_student(z, observation, c) - A_student(z, observation, c_tilde)
```

Primary mechanism metrics:

- response norm ratio: `E[||delta_S||] / (E[||delta_T||] + eps)`
- response cosine: mean cosine similarity between `delta_S` and `delta_T`
- normalized response error: `E[||delta_S-delta_T||] / (E[||delta_T||] + eps)`

All three must also be reported per action-horizon position and, where possible, by translation, rotation, and gripper dimensions.

## In scope

- LeRobot SmolVLA and `lerobot/smolvla_libero`
- 10/5/2/1-NFE deterministic diagnostics
- SnapFlow reproduction in SmolVLA
- semantic-changing and paraphrase condition pairs
- standard LIBERO capability evaluation
- LIBERO-CF external evaluation
- CRP response matching with factual anchoring after the pre-training gate

## Out of scope for the first study

- claims that CRP generally solves VLA language grounding
- training on LIBERO-CF evaluation conditions
- openpi/pi0.5 or gesture/GesVLA integration before SmolVLA evidence is complete
- full VLM retraining in the first CRP implementation
- treating offline response metrics as substitutes for closed-loop evaluation

## Required comparisons

- SmolVLA 10-step teacher
- naive 1-step SmolVLA
- SnapFlow 1-step
- SnapFlow plus factual pointwise KD
- SnapFlow plus condition dropout
- SnapFlow plus CAG
- SnapFlow plus CRP
- SnapFlow plus CRP plus CAG

The two decisive rebuttal controls are random pair shuffling and factual pointwise KD. If either explains the full gain, the CRP mechanism claim must be weakened or rejected.

## Internal go/no-go thresholds

The provisional continuation rule requires all of the following:

- standard success difference between teacher and one-step student no more than about 3 percentage points;
- conditional-response retention falls by about 10–15% or a consistent response-direction/faithful-success loss is observed;
- effect appears in at least two intervention classes and two task groups;
- direction is consistent across multiple action-noise seeds;
- effect is not fully explained by factual action MSE;
- effect persists after SnapFlow, not only naive step reduction.

These thresholds are project screening rules, not accepted field standards. Any change requires a dated decision-log entry made before inspecting the corresponding final comparison.
