# CRP-VLA evidence ledger

| Claim or artifact | Status | Evidence | Notes |
|---|---|---|---|
| LeRobot base revision | VERIFIED | Git commit `a3a8653a5b569e8e70cd6fe4b0973c4b2b983120` | local checkout |
| LeRobot requires Python 3.12+ | VERIFIED | `pyproject.toml` | local source |
| LeRobot LIBERO dependency is Linux-only | VERIFIED | `pyproject.toml` environment marker | local source |
| Local host is suitable for CUDA/LIBERO closed-loop validation | REFUTED | macOS arm64; no `nvidia-smi` | use A100 Linux host |
| SnapFlow official code is available | UNVERIFIED/NOT FOUND | arXiv page has no official code link as of 2026-08-13 | re-check before major reimplementation updates |
| One-step compression causes a conditional-response gap | UNVERIFIED | planned PR1–PR5 experiments | core go/no-go hypothesis |
| CRP restores conditional response | UNVERIFIED | future formal training | do not state as result |
| Official SmolVLA LIBERO checkpoint revision/hash | VERIFIED | `docs/crp_vla/checkpoint_manifest.json` and local SHA-256 | weights downloaded locally |
| CRP unit primitives behave as specified | VERIFIED | 10 focused tests on local CPU | formula/gradient/invariant level only |
| Existing flow-matching and SmolVLA processor tests remain compatible | VERIFIED | 14 passed, 4 skipped | local CPU; not full upstream suite |
