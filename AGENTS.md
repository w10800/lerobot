This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

## CRP-VLA Operating Contract

This checkout is also the CRP-VLA research workspace. Before CRP-VLA work, read in order:

1. `CRP_VLA.md`
2. `docs/crp_vla/PROJECT_CHARTER.md`
3. `docs/crp_vla/EXPERIMENT_PLAN.md`
4. `docs/crp_vla/PRETRAIN_READINESS.md`
5. `logs/crp_vla/DECISIONS.md`, `logs/crp_vla/RISKS.md`, and the latest `logs/crp_vla/WORKLOG.md` entry

The following rules are mandatory:

- Keep the core claim narrow: measure conditional-response loss caused by compressing the same flow-matching VLA from multi-step to one-step generation. Do not relabel the project as a general language-grounding method.
- Separate `VERIFIED`, `INFERRED`, and `UNVERIFIED` statements. Never report an expected metric, a paper result, a smoke test, or a planned run as a local experimental result.
- Do not start full SnapFlow or CRP training until every mandatory item in `docs/crp_vla/PRETRAIN_READINESS.md` is checked and the go/no-go decision is logged.
- LIBERO-CF is evaluation-only. Counterfactual training pairs must be constructed from standard LIBERO training tasks with task/template/object leakage checks.
- Factual and counterfactual branches must share the exact observation, robot state, initial action noise, normalization version, and execution settings; only the condition may change.
- Preserve immutable provenance for every run: repository commit, dirty status, checkpoint revision/hash, dataset revision, environment fingerprint, seeds, NFE, execution horizon, and config snapshot.
- External repositories live under `third_party/` as pinned Git submodules. Do not edit them unless the change is intentional, documented, and versioned.
- Keep CRP additions modular. Prefer `src/lerobot/policies/smolvla_crp/`, `src/lerobot/envs/libero_cf/`, and `scripts/research/crp_vla/` over expanding `modeling_smolvla.py` indiscriminately.
- A behavior-preserving refactor requires numerical-equivalence tests before method code is layered on top. Target branches must be stop-gradient, and zero-initialized target-time conditioning must preserve legacy outputs at initialization.
- Record material decisions in `logs/crp_vla/DECISIONS.md`, new risks or blockers in `logs/crp_vla/RISKS.md`, and completed work/verification commands in `logs/crp_vla/WORKLOG.md` in the same change.
- Never commit checkpoints, datasets, response caches, rollouts, videos, credentials, or machine-specific environments. Store only manifests, hashes, configs, and small deterministic fixtures.
- Use one focused commit per logical PR-sized change. Run relevant unit tests and `git diff --check` before committing.
