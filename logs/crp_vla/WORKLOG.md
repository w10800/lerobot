# CRP-VLA work log

Append entries; do not rewrite history to make failed attempts disappear.

## 2026-08-13 — workspace bootstrap

- Read and normalized the supplied CRP-VLA project brief.
- Cloned LeRobot and pinned base commit `a3a8653a5b569e8e70cd6fe4b0973c4b2b983120`.
- Created branch `research/crp-vla-pretrain`.
- Added pinned external repositories under `third_party/`.
- Added the operating contract, project charter, experiment plan, readiness gate, reproducibility contract, and research logs.
- Verified that the local host is macOS arm64 with system Python 3.9.6; current LeRobot requires Python 3.12+ and the LIBERO extra is Linux-only.
- Status: workspace governance verified; GPU/checkpoint/simulator validation not yet run.

## 2026-08-13 — literature, dependencies, and PR1–PR5 implementation

- Downloaded 13 project papers to the ignored local literature store and generated `literature/manifest.tsv` with SHA-256 and byte counts.
- Installed the locked Python 3.12 SmolVLA, test, and development environments with uv.
- Downloaded `lerobot/smolvla_libero` at resolved revision `31d453f7edd78c839a8bbc39744a292686daf0de`; model SHA-256 is `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
- Pinned `lerobot/libero` for CRP-VLA to dataset revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` because the original checkpoint train config had a null dataset revision.
- Added deterministic multi-NFE baseline CLI, cached/full velocity APIs, zero-init target-time conditioning, equation-level SnapFlow loss, matched-pair invariants, response metrics, and counterfactual diagnostic CLIs.
- Local verification: `10 passed` for CRP-specific tests; `14 passed, 4 skipped` for shared flow-matching and SmolVLA processor compatibility; Ruff check passed after formatting.
- Remaining mandatory verification: real checkpoint action equivalence and 10/5/2/1 hashes on Linux/CUDA, real multi-task/multi-seed compression diagnostics, and closed-loop LIBERO/LIBERO-CF smoke tests.
- Additional local real-checkpoint smoke checks passed on a synthetic processed batch: exact repeatability at 10/5/2/1 NFE; exact legacy/refactored velocity equivalence; exact zero-init target-time equivalence; finite SnapFlow forward with detached shortcut target. See `local_smoke_summary.json`.
