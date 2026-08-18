# Task 14 Reproducibility Commands

Task 14 stopped at the untouched-state capacity gate. The following commands reproduce the audit;
they do not launch training or rollout.

```bash
export LIBERO_CONFIG_PATH=/root/crp-vla/artifacts/crp_vla/task13_recovery/libero_standard_config
uv run python scripts/research/crp_vla/prepare_task14.py \
  --base-checkpoint checkpoints/smolvla_libero \
  --snap-checkpoint outputs/crp_vla/snapflow_maturation_30k/checkpoints/020000/pretrained_model \
  --old-dev40-manifest artifacts/crp_vla/replay_v2/development/run_manifest.json \
  --formal100-manifest artifacts/crp_vla/followup/libero_paired_final_v1.json \
  --confirmation1200-manifest artifacts/crp_vla/task10_confirmation1200_instrumentation/CONFIRMATION1200_MANIFEST.json \
  --attempt001-manifest artifacts/crp_vla/task13_prospective_impact_validation/DEV_B_MANIFEST.json \
  --attempt002-manifest artifacts/crp_vla/task13_recovery/DEV_B2_MANIFEST.json \
  --task12-registry artifacts/crp_vla/task13_recovery/USED_STATE_REGISTRY.json \
  --task13-final-report artifacts/crp_vla/task13_recovery/TASK13_RECOVERY_FINAL_REPORT.md \
  --task13-transition-effects artifacts/crp_vla/task13_recovery/ATTEMPT002_TRANSITION_EFFECTS.json \
  --robosuite-root .venv/lib/python3.12/site-packages/robosuite \
  --output-root artifacts/crp_vla/task14_position_causal_audit

uv run pytest tests/policies/smolvla_crp/test_task14_position_causal_audit.py -q
uv run ruff check scripts/research/crp_vla/task14_common.py \
  scripts/research/crp_vla/prepare_task14.py \
  tests/policies/smolvla_crp/test_task14_position_causal_audit.py
(cd artifacts/crp_vla/task14_position_causal_audit && sha256sum -c TASK14_SHA256SUMS.txt)
git diff --check
```
