#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="/root/crp-vla"
PROTOCOL_PATH="${REPOSITORY_ROOT}/artifacts/crp_vla/task14r_reset_transaction_recovery/TASK14R_R0S_PROTOCOL.json"
PROTOCOL_SHA256_PATH="${REPOSITORY_ROOT}/artifacts/crp_vla/task14r_reset_transaction_recovery/TASK14R_R0S_PROTOCOL.sha256"
LIBERO_ROOT="${REPOSITORY_ROOT}/third_party/libero-cf"
OUTPUT_PARENT="${REPOSITORY_ROOT}/outputs/crp_vla/task14r_reset_transaction_recovery"
OUTPUT_ROOT="${OUTPUT_PARENT}/r0s_attempt001"
CONSOLE_LOG="${OUTPUT_PARENT}/r0s_attempt001.console.log"
: "${TASK14R_R0S_EXPECTED_COMMIT:?Set TASK14R_R0S_EXPECTED_COMMIT to the reviewed artifact-bearing commit}"

cd "${REPOSITORY_ROOT}"

if [[ "$(git branch --show-current)" != "codex/task14r-r0s-renderer-shift-audit" ]]; then
    echo "Task14R R0S must run from codex/task14r-r0s-renderer-shift-audit" >&2
    exit 1
fi
if [[ "$(git rev-parse HEAD)" != "${TASK14R_R0S_EXPECTED_COMMIT}" ]]; then
    echo "Task14R R0S commit does not match TASK14R_R0S_EXPECTED_COMMIT" >&2
    exit 1
fi
if [[ "$(git rev-parse HEAD^)" != "609118b7016a411c52a0ce5a6bdd8dc36ab74d79" ]]; then
    echo "Task14R R0S parent commit drift" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Task14R R0S requires a clean repository" >&2
    exit 1
fi
if [[ ! -f "${PROTOCOL_PATH}" ]]; then
    echo "Frozen Task14R R0S protocol is missing" >&2
    exit 1
fi
if [[ ! -f "${PROTOCOL_SHA256_PATH}" ]]; then
    echo "Frozen Task14R R0S protocol SHA-256 sidecar is missing" >&2
    exit 1
fi
if [[ ! -d "${LIBERO_ROOT}" ]]; then
    echo "Pinned LIBERO checkout is missing" >&2
    exit 1
fi

mkdir -p "${OUTPUT_PARENT}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Task14R R0S output root already exists: ${OUTPUT_ROOT}" >&2
    exit 1
fi
if [[ -e "${CONSOLE_LOG}" ]]; then
    echo "Task14R R0S console log already exists: ${CONSOLE_LOG}" >&2
    exit 1
fi

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec uv run python scripts/research/crp_vla/run_task14r_r0s.py \
    --protocol "${PROTOCOL_PATH}" \
    --libero-root "${LIBERO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    >"${CONSOLE_LOG}" 2>&1
