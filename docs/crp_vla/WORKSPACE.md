# Workspace layout

```text
VLA/                              # LeRobot checkout and CRP-VLA primary Git repository
├── AGENTS.md                     # mandatory operating contract
├── CRP_VLA.md                    # project entry point
├── configs/crp_vla/              # reviewed experiment configurations
├── docs/crp_vla/                 # charter, plans, gates, reproducibility notes
├── literature/                   # versioned bibliography plus downloaded PDFs
│   ├── papers/
│   └── notes/
├── logs/crp_vla/                 # append-only work, decision, risk, and evidence logs
├── scripts/research/crp_vla/     # preparation, verification, evaluation, and plotting CLIs
├── src/lerobot/policies/smolvla_crp/
├── src/lerobot/envs/libero_cf/
├── tests/policies/smolvla_crp/
├── third_party/libero-cf/        # pinned evaluation-only submodule
└── third_party/reference/        # pinned read-only implementation references
```

Large or generated data must remain outside Git:

- `checkpoints/`
- `data/crp_vla/`
- `artifacts/crp_vla/`
- `outputs/crp_vla/`
- `wandb/`

Each generated directory must contain or be accompanied by a small versioned manifest that records origin, revision/hash, generation command, and schema version.

## Branch and commit policy

- Base upstream commit: recorded in `logs/crp_vla/DECISIONS.md` and Git history.
- Working branch: `research/crp-vla-pretrain`.
- Logical implementation sequence: baseline → velocity API → target-time/SnapFlow → diagnostics → adapter/cache/CRP after the gate.
- Do not combine numerical refactors with loss changes in one commit.
- External repositories are submodules and must stay pinned until an explicit upgrade decision is logged.
