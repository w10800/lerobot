# Mature SnapFlow Checkpoint Selection Protocol

## Frozen status and exclusions

This rule is frozen before the D-013 training run. It governs exactly one continuous SnapFlow trajectory from the pinned Base checkpoint and the registered checkpoints at 1k, 3k, 5k, 10k, 20k, and 30k. The old 1k checkpoint remains a pilot, the original Formal LIBERO Gate remains FAIL, and CRP/CAG/KD/method training remains HOLD.

The selection process must not read or use LIBERO-CF, condition-response metrics, CRP metrics, the disjoint confirmation set, or any result from a new method. Snap-10, Snap-2, latency, and training-drift analyses are mandatory reports but are not selection criteria.

## Immutable training identity

- Base revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
- Base model SHA-256: `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`
- Dataset: `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- Seed: `1000`; batch size: `4`; precision: `bf16`
- Learning rate: `2.5e-5`; gradient clip: `1.0`
- SnapFlow alpha: `0.5`; shortcut weight: `0.1`
- Scheduler warmup/decay: `500/30000`
- Registered checkpoints: `1000, 3000, 5000, 10000, 20000, 30000`
- Configuration: `configs/crp_vla/snapflow_maturation_30k.yaml`

The run must be one uninterrupted optimizer/data-order trajectory. A machine interruption may resume only from its latest exact optimizer/scheduler/RNG checkpoint with unchanged topology and configuration; it is not a new candidate run. No checkpoint-specific restart is allowed.

## Frozen ordinary factual fixture

Before training, materialize one deterministic ordinary factual fixture with exactly one record per each of the 40 standard LIBERO tasks. For every task choose the highest indexed episode in the pinned dataset and its middle valid action frame; retain the raw sample, task identity, action chunk, processor/normalization hashes, and source frame hashes. Use action-noise seeds `0, 1, 2` for inference. This is a fixed in-distribution factual diagnostic fixture and is not claimed as an unseen generalization set because the maturation run trains on the full pinned dataset.

For every registered checkpoint report on the identical fixture:

- deterministic total SnapFlow validation loss and its FM/shortcut accounting under fixed noise/time seeds;
- raw 7-D action MSE for 10/2/1 NFE over the first 10 executed actions and the full 50-action chunk;
- finite-output and repeatability checks;
- median warmed 10/2/1 policy latency.

## Eligibility gates

A checkpoint is eligible only when all conditions hold:

1. model, optimizer, scheduler, RNG, config, dataset revision, and data-order hashes are complete;
2. training and factual evaluation contain no NaN/Inf, OOM, data exception, loss-accounting failure, or provenance mismatch;
3. its mean fixed-fixture factual total validation loss is no higher than the registered 1k checkpoint from the same continuous run;
4. all 40 replay-v2 development cases complete for Snap-10/Snap-2/Snap-1 with the frozen capsules, evaluator, and noise schedule.

If no checkpoint after 1k is eligible, select 1k and conclude that maturation was not established. The existing pilot checkpoint is reported for reproducibility but is not substituted for the new run's 1k checkpoint.

## Selection rule

Among eligible registered checkpoints, maximize the ordinary factual replay-v2 development Snap-1 success count across the fixed 40 cases. Break an exact success-count tie by selecting the earlier training step. Do not use confidence-interval width, Snap-2/Snap-10 outcomes, latency, failure category, offline response metrics, or any subgroup to break a tie.

The selected checkpoint is a development-selected mature-baseline candidate only. It cannot change the original Formal Gate and cannot be called confirmed until a separately approved, single-use confirmation protocol is executed.
