# CRP-VLA risk register

| ID | Risk | Severity | Mitigation | Current status |
|---|---|---:|---|---|
| R-001 | Teacher is condition-blind or wrong on counterfactual prompts | high | validity checks, multi-noise stability, factual rollout confidence, response threshold | open |
| R-002 | LIBERO-CF test leakage into CF-Train | critical | task/template/object/condition manifests and automated overlap audit | open |
| R-003 | Different noise or preprocessing creates a false response gap | critical | invariant-enforcing paired API and unit tests | in progress |
| R-004 | Refactor changes legacy SmolVLA numerics | high | equivalence tests before target-time/SnapFlow changes | in progress |
| R-005 | Offline response metrics do not predict closed-loop behavior | high | always pair offline metrics with standard LIBERO and LIBERO-CF rollouts | open |
| R-006 | Pointwise KD or random pair control explains the gain | high | mandatory rebuttal baselines; weaken/reject mechanism claim if matched | open |
| R-007 | macOS arm64 host cannot run Linux-only LIBERO/CUDA stack | high | local unit/static tests; final smoke gates on A100 Linux host | confirmed |
| R-008 | Upstream LeRobot API drift | medium | pinned base commit and explicit upgrade log | mitigated |
| R-009 | SnapFlow official implementation is unavailable | medium | reproduce only from the paper; record formula-level assumptions and tests | confirmed as of 2026-08-13 |
| R-010 | Large artifacts enter Git | medium | ignore rules plus manifests/hashes only | mitigated |
| R-011 | SnapFlow paper states three action-expert forwards per step but equations require FM, two detached target calls, and a distinct one-step student call | medium | equation-faithful four-call implementation; benchmark memory/throughput before formal training; revisit if official code appears | open |
