# Task 13 Recovery Audit Correction

**Status: VERIFIED REPORT-LOGIC CORRECTION**

The initial Attempt001 post-hoc report incorrectly included `task13_attempt001_initial` as a comparison dataset while auditing Attempt001 itself. This produced exactly 1,200 self-matches (400 cases x three initial-state identity fields), not overlap with Task12 or another prior support. The corrected audit explicitly excludes the self-dataset and finds `0` prior/Task12 overlap records.

This correction does not restore prospective validity to Attempt001, does not change any Attempt002 case, outcome, query, branch, H1/H2/H3 statistic, or the final `PARTIAL_MECHANISM_REPLICATION` decision.

- Superseded post-hoc report SHA-256: `8feafa74cd89408e7c9f44baa506c9f9041932981a463dfe4bc5005856c0088a`.
- Corrected post-hoc report SHA-256: `5d6c3dec5b3e3a0b8d80e077895614e5e4f2fb235c2e433dff326077985d9e2b`.
- Correction code commit: `a7250edaf3600e3519ad90edf8262cb3d2be8625`.
- Corrected unix ns: `1786965921173736280`.
