# Reproducibility contract

Every run must archive a config snapshot and a manifest with:

- UTC timestamp and run ID
- primary repository commit and dirty-tree diff hash
- all submodule commits
- exact command and working directory
- checkpoint repository, revision, resolved commit, and SHA-256 for downloaded artifacts
- dataset repository/revision and normalization metadata hash
- OS, architecture, Python, package-lock hash, Torch, CUDA/cuDNN, GPU model, and driver
- global, data-loader, environment, and action-noise seeds
- NFE, teacher NFE, execution horizon, action horizon, and normalization version
- intervention pair ID and generation rule
- output schema version

## Determinism levels

- Bitwise: exact action bytes and hash match on the same hardware/software stack.
- Numeric: tensors match within declared `atol`/`rtol` when kernels are not bitwise deterministic.
- Statistical: stochastic closed-loop metrics are reported with seeds, rollout counts, and uncertainty.

Do not claim cross-device bitwise determinism. Record which level each verification achieved.

## Artifact policy

Raw checkpoints, datasets, caches, videos, and run directories stay outside Git. Their manifests and hashes are versioned. Small synthetic fixtures may be committed when licenses and privacy permit.

## Result status vocabulary

- `PLANNED`: specified but not run
- `SMOKE`: engineering path executed; not scientific evidence
- `ANALYZED`: outputs inspected but not independently reproduced
- `VERIFIED`: rerun meets the stated deterministic or stochastic tolerance
- `BLOCKED`: required environment/input unavailable or a gate failed
