# CRP-VLA workspace

CRP-VLA stands for **Conditional Response-Preserving One-Step Distillation for Flow-Matching VLAs**. The working paper title is:

> *What Does One-Step Distillation Forget? Preserving Conditional Transport Geometry in Flow-Matching VLAs*

The project tests whether compressing the same flow-matching VLA from ten denoising steps to one causes an additional loss of sensitivity to language or other conditions, even when ordinary task success remains similar. The proposed Conditional Response-Preserving (CRP) objective is not treated as effective until the compression gap and the method benefit are measured under the registered controls.

## Start here

- Project boundary and claims: `docs/crp_vla/PROJECT_CHARTER.md`
- Workspace layout: `docs/crp_vla/WORKSPACE.md`
- Experiment plan: `docs/crp_vla/EXPERIMENT_PLAN.md`
- Training gate: `docs/crp_vla/PRETRAIN_READINESS.md`
- Reproducibility contract: `docs/crp_vla/REPRODUCIBILITY.md`
- Literature: `literature/README.md`
- Work, decisions, and risks: `logs/crp_vla/`

The upstream LeRobot base is pinned by Git history. External benchmark and reference repositories are pinned as submodules in `third_party/`.

## Current boundary

The current milestone ends before formal training. It covers PR1–PR5 from the project brief: deterministic multi-NFE baseline support, a velocity API, zero-init target-time conditioning, SnapFlow objective primitives, and matched-noise counterfactual diagnostics. GPU checkpoint and closed-loop simulator validation remain mandatory before any result is claimed.
