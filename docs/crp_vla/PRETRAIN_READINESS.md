# Formal-training readiness gate

Status meanings: `[x]` verified locally; `[ ]` incomplete; `[~]` prepared but requires the target GPU/Linux host.

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
- [~] Target CUDA/PyTorch/GPU fingerprint is captured.

## Code gates

- [x] PR1 deterministic 10/5/2/1-NFE baseline passes with a real checkpoint (synthetic processed batch, engineering smoke only).
- [x] PR2 velocity API is numerically equivalent to the legacy path (exact on the real checkpoint smoke input).
- [x] PR3 zero-init target-time path is numerically equivalent at initialization and receives gradients.
- [x] PR4 SnapFlow target/loss tests pass, including stop-gradient and alpha endpoints.
- [x] PR5 matched-noise diagnostic schema and metric tests pass.
- [~] One LIBERO-CF task, one initial state, and two conditions complete in closed loop.

## Data and leakage gates

- [x] Training/evaluation separation policy is documented.
- [ ] CF-Train pair schema and deterministic generation rule are frozen.
- [ ] Automated train/test task-template/object/condition leakage audit passes.
- [ ] Teacher response filtering thresholds are declared before cache generation.
- [ ] Response cache manifest includes checkpoint hash, teacher NFE, noise seed, normalization version, episode/frame, and prompts.

## Experiment gates

- [ ] Baseline action hashes and latency are recorded for 10/5/2/1 NFE.
- [ ] Compression response plots cover at least two task groups, two intervention types, and multiple noise seeds.
- [ ] Factual action error is analyzed alongside response metrics.
- [ ] SnapFlow 1-step still exhibits the registered response gap.
- [ ] A dated go/no-go decision is appended to `logs/crp_vla/DECISIONS.md`.

Formal training is forbidden while any mandatory item is unchecked. Items marked `[~]` can only be completed on the Linux/CUDA simulator host and must not be treated as locally verified.
