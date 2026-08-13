#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [ "$(uname -s)" != "Linux" ]; then
    printf 'BLOCKED: the full gate requires Linux for the LeRobot LIBERO extra.\n' >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'BLOCKED: nvidia-smi is unavailable; run this gate on the target CUDA host.\n' >&2
    exit 2
fi
: "${CRP_BATCH_FILE:?Set CRP_BATCH_FILE to a real, preprocessed and versioned LIBERO batch fixture}"
: "${CRP_CHECKPOINT_DIR:?Set CRP_CHECKPOINT_DIR to the pinned smolvla_libero checkpoint directory}"
: "${CRP_CHECKPOINT_REVISION:?Set CRP_CHECKPOINT_REVISION to the immutable Hugging Face commit}"

uv sync --locked --extra smolvla --extra libero --extra test --extra dev
uv run ruff check \
    src/lerobot/policies/smolvla/configuration_smolvla.py \
    src/lerobot/policies/smolvla/modeling_smolvla.py \
    src/lerobot/policies/smolvla_crp \
    scripts/research/crp_vla \
    tests/policies/smolvla_crp
uv run pytest \
    tests/policies/smolvla_crp \
    tests/policies/common/test_flow_matching.py \
    tests/processor/test_smolvla_processor.py -q
uv run python scripts/research/crp_vla/capture_environment.py \
    --output artifacts/crp_vla/pretrain_gate/environment.json
uv run python scripts/research/crp_vla/verify_baseline.py \
    --checkpoint "$CRP_CHECKPOINT_DIR" \
    --revision "$CRP_CHECKPOINT_REVISION" \
    --batch-file "$CRP_BATCH_FILE" \
    --output artifacts/crp_vla/pretrain_gate/baseline.json \
    --steps 10 5 2 1 \
    --noise-seed 0 \
    --repeats 2 \
    --device cuda
uv run python scripts/research/crp_vla/verify_target_time_equivalence.py \
    --checkpoint "$CRP_CHECKPOINT_DIR" \
    --revision "$CRP_CHECKPOINT_REVISION" \
    --batch-file "$CRP_BATCH_FILE" \
    --output artifacts/crp_vla/pretrain_gate/target_time_equivalence.json \
    --noise-seed 0 \
    --device cuda \
    --atol 0
uv run python scripts/research/crp_vla/verify_velocity_equivalence.py \
    --checkpoint "$CRP_CHECKPOINT_DIR" \
    --revision "$CRP_CHECKPOINT_REVISION" \
    --batch-file "$CRP_BATCH_FILE" \
    --output artifacts/crp_vla/pretrain_gate/velocity_equivalence.json \
    --noise-seed 0 \
    --device cuda \
    --atol 0

printf 'Engineering gate complete. Scientific go/no-go still requires the registered multi-task, multi-seed response sweep.\n'
