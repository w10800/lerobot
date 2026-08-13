# CRP-VLA decision log

## D-001 — primary codebase and base revision

- Date: 2026-08-13
- Decision: use Hugging Face LeRobot at base commit `a3a8653a5b569e8e70cd6fe4b0973c4b2b983120` as the primary repository.
- Reason: official SmolVLA implementation, checkpoint/config integration, and current LIBERO support.
- Consequence: upstream changes require an explicit upgrade decision and re-running numerical-equivalence tests.

## D-002 — external repository management

- Date: 2026-08-13
- Decision: keep LIBERO-CF and implementation references as pinned Git submodules.
- Reason: immutable provenance without copying or silently diverging external code.

## D-003 — training boundary

- Date: 2026-08-13
- Decision: stop the current milestone before formal SnapFlow/CRP training. Implement and validate PR1–PR5 first.
- Reason: the compression-induced response gap is the prerequisite evidence for CRP.

## D-004 — model scope

- Date: 2026-08-13
- Decision: use SmolVLA first; defer openpi/pi0.5 and gesture/GesVLA extensions.
- Reason: isolate the core method and fit single-A100 development constraints.

## D-005 — evidence language

- Date: 2026-08-13
- Decision: all proposed effects and thresholds remain `UNVERIFIED` until reproduced locally. Paper claims are attributed as external evidence.
- Reason: prevent plans or reported literature results from being mistaken for project results.

## D-006 — target-host interpreter and memory boundary

- Date: 2026-08-13
- Decision: use an explicit uv-managed Python 3.12 environment on the A100 host and treat 40GB as a hard measured memory boundary.
- Reason: the project tooling targets Python 3.12, while automatic resolution selected unclassified Python 3.14; the assigned A100 is the 40GB PCIe variant.
- Consequence: benchmark the registered configuration unchanged and stop for a logged decision if it exceeds memory, rather than silently reducing the workload.
