# Formal-training readiness gate

Status meanings: `[x]` verified on the stated host; `[ ]` incomplete; `[~]` prepared but requires the target GPU/Linux host.

## Scope and governance

- [x] Research question, falsifiable claims, and exclusions are written.
- [x] Operating contract and append-only logs exist.
- [x] Go/no-go thresholds are identified as provisional internal thresholds.
- [x] LIBERO-CF is designated evaluation-only.

## Version control and provenance

- [x] LeRobot base commit is pinned in Git history.
- [x] CRP work occurs on a dedicated branch.
- [x] External code repositories are pinned by submodule commit.
- [x] Model checkpoint revision and SHA-256 are recorded.
- [x] Dataset revision and relevant normalization metadata are recorded.
- [x] Target CUDA/PyTorch/GPU fingerprint is captured.

## Code gates

- [x] PR1 deterministic 10/5/2/1-NFE baseline passes with a real checkpoint (synthetic processed batch, engineering smoke only).
- [x] PR2 velocity API is numerically equivalent to the legacy path (exact on the real checkpoint smoke input).
- [x] PR3 zero-init target-time path is numerically equivalent at initialization and receives gradients.
- [x] PR4 SnapFlow target/loss tests pass, including stop-gradient and alpha endpoints.
- [x] PR5 matched-noise diagnostic schema and metric tests pass.
- [x] One LIBERO-CF task, one initial state, and two conditions complete in closed loop (engineering smoke only).

## Data and leakage gates

- [x] Training/evaluation separation policy is documented.
- [x] CF-Train pair schema and deterministic generation rule are frozen.
- [x] Automated train/test task-template/object/condition leakage audit passes for the frozen pilot catalog.
- [x] Teacher response filtering thresholds are declared before cache generation.
- [x] Response cache manifest includes checkpoint hash, teacher NFE, noise seed, normalization version, episode/frame, and prompts.

## Experiment gates

- [x] Baseline action hashes and latency are recorded for 10/5/2/1 NFE.
- [x] Four-arm compression diagnostics cover 40 semantic pairs, two balanced intervention types, and three matched-noise seeds.
- [x] Factual action error is analyzed alongside response metrics.
- [x] A 1k-step SnapFlow engineering smoke still exhibits a response gap; this is preliminary evidence, not a method result.
- [x] The registered ordinary-LIBERO gate completed and its FAIL is immutable: Snap-1 minus Base-10 was `-5%` with paired 95% interval `[-13%, +2%]`. This excludes the 1k pilot as a mature baseline and continues to block CRP; D-013 narrowly authorizes one preregistered baseline-maturation run to test whether a later checkpoint can qualify.
- [x] Replay-v2 Phase A completed 8/8 capsules and 48/48 same-process six-arm rollouts with zero infrastructure exceptions before the D-013 maturation authorization became executable.
- [x] A dated go/no-go decision is appended to `logs/crp_vla/DECISIONS.md`.

Formal training is forbidden while any mandatory item is unchecked or the latest applicable decision is HOLD/NO-GO. D-013 is a contract amendment for one SnapFlow baseline-maturation run only; the original gate FAIL and the CRP/CAG/KD/method-training HOLD remain controlling outside that exact run. Items marked `[~]` can only be completed on the Linux/CUDA simulator host and must not be treated as locally verified.
