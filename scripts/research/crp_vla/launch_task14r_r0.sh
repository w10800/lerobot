#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="/root/crp-vla"
PROTOCOL_PATH="${REPOSITORY_ROOT}/artifacts/crp_vla/task14r_reset_transaction_recovery/TASK14R_R0_PROTOCOL.json"
LIBERO_ROOT="${REPOSITORY_ROOT}/third_party/libero-cf"
OUTPUT_PARENT="${REPOSITORY_ROOT}/outputs/crp_vla/task14r_reset_transaction_recovery"
OUTPUT_ROOT="${OUTPUT_PARENT}/r0_attempt001"
CONSOLE_LOG="${OUTPUT_PARENT}/r0_attempt001.console.log"
: "${TASK14R_R0_EXPECTED_COMMIT:?Set TASK14R_R0_EXPECTED_COMMIT to the reviewed artifact-bearing commit}"

cd "${REPOSITORY_ROOT}"

if [[ "$(git branch --show-current)" != "codex/task14r-r0-audit-hardening" ]]; then
    echo "Task14R R0 must run from codex/task14r-r0-audit-hardening" >&2
    exit 1
fi
if [[ "$(git rev-parse HEAD)" != "${TASK14R_R0_EXPECTED_COMMIT}" ]]; then
    echo "Task14R R0 commit does not match TASK14R_R0_EXPECTED_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Task14R R0 requires a clean repository" >&2
    exit 1
fi
if [[ ! -f "${PROTOCOL_PATH}" ]]; then
    echo "Frozen Task14R R0 protocol is missing" >&2
    exit 1
fi
if [[ ! -d "${LIBERO_ROOT}" ]]; then
    echo "Pinned LIBERO checkout is missing" >&2
    exit 1
fi

mkdir -p "${OUTPUT_PARENT}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Task14R R0 output root already exists: ${OUTPUT_ROOT}" >&2
    exit 1
fi
if [[ -e "${CONSOLE_LOG}" ]]; then
    echo "Task14R R0 console log already exists: ${CONSOLE_LOG}" >&2
    exit 1
fi

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec uv run python scripts/research/crp_vla/run_task14r_r0.py \
    --protocol "${PROTOCOL_PATH}" \
    --libero-root "${LIBERO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    >"${CONSOLE_LOG}" 2>&1
