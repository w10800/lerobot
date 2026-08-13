# CRP-VLA code experiment plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: crp_vla_code_plan_v1

## Experiment overview

- Title: Conditional response loss under one-step SmolVLA distillation
- Objective: isolate and measure compression-induced conditional-response loss, then test CRP as a targeted mitigation
- Hypothesis: one-step compression can preserve ordinary success while reducing matched-noise conditional finite-difference response
- Type: training, offline analysis, and closed-loop simulation

## Independent, dependent, and controlled variables

- Independent: inference NFE, distillation objective, intervention type, teacher NFE, CRP/anchor weights
- Primary dependent: normalized conditional-response error
- Secondary dependent: response norm ratio, response cosine, faithful/biased touch and success, ordinary LIBERO success, latency
- Controls: model initialization, observation/state, action noise, preprocessing, task initial state, normalization, execution horizon, checkpoint/data revisions
- Main confounds: teacher counterfactual failure, invalid interventions, train/test leakage, seed instability, action normalization drift, execution-horizon changes

## Pre-training implementation sequence

1. Baseline: expose deterministic 10/5/2/1-NFE sampling and record action hashes/latency.
2. Velocity API: make local velocity prediction independently testable with no numerical change.
3. Target time and SnapFlow: add zero-init target-time conditioning and two-step shortcut targets with detached target branches.
4. Diagnostics: compute response deltas and aggregate norm-ratio/cosine/normalized-error metrics with strict matched-noise validation.
5. Minimal integration smoke tests: one checkpoint sample and one LIBERO-CF task/initial-state/two-condition closed-loop run on the target Linux/CUDA host.

Formal SnapFlow or CRP training starts only after step 5 and the readiness gate pass.

## Planned formal training setup

- Framework: Python 3.12+, PyTorch, LeRobot
- Target environment: Linux, CUDA-capable single A100 80GB, bfloat16
- Base model: `lerobot/smolvla_libero`, pinned revision and file hash
- Student initialization: same checkpoint as the frozen 10-NFE teacher
- SnapFlow initial hyperparameters: alpha 0.5, shortcut weight 0.1, AdamW, learning rate 2.5e-5, gradient clip 1.0, warmup 500, batch size 4
- CRP search: beta in {0.03, 0.1, 0.3}; gamma in {0.03, 0.1}
- Smoke checkpoints: 1k, 3k, 5k, 10k before considering 30k

Paper-reported hyperparameters are starting points, not locally verified optima.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| environment fingerprint | JSON | contains Git, OS, Python, Torch/CUDA, device, submodule commits |
| baseline manifest | JSON | immutable model/data/config/seed identifiers |
| action samples | NPZ + JSON | repeated run with same seed matches within configured tolerance |
| counterfactual records | JSONL or Parquet | schema-valid and matched-noise invariant passes |
| response summary | JSON + plots | global and horizon-wise metrics reproducible from records |
| LIBERO-CF rollouts | benchmark format | one task/state/two conditions before training gate |

## Analysis plan

- Primary comparison: SnapFlow 1-step versus frozen SmolVLA 10-step on normalized response error.
- Report uncertainty over task, intervention, and noise seed; do not pool away task-level failures.
- Closed-loop final results require multiple seeds and sufficient rollouts; early smoke tests are engineering validation only.
- Inspect factual MSE and ordinary success jointly with response metrics.
- Pre-register exclusions: invalid scene-condition pairs, teacher response below the declared threshold, corrupt rollouts, and exact technical failures only.

## Stop conditions

- Stop method training if the registered compression gap does not pass the go/no-go rule.
- Stop and fix infrastructure if matched-noise or deterministic replay tests fail.
- Stop and revisit the claim if random-pair or pointwise-KD controls explain the same benefit.
- Never convert a failed run into an exclusion after inspecting its method label without a logged, method-blind technical reason.
