# CRP-VLA Task 13 Recovery Final Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: VERIFIED
- Generated unix ns: `1786965696090573415`

**Primary status: PARTIAL_MECHANISM_REPLICATION**

## Required answers

1. Attempt001 lost prospective validity because the mandatory Task12 archived-state overlap audit checked zero states before outcome reveal.
2. Authoritative Task12 raw state records read: `7772`.
3. Unique Task12 archived state payload hashes: `7587`.
4. Attempt001 versus prior/Task12 datasets overlap records: `0`.
5. Attempt001 remains primary-invalid even when post-hoc overlap is zero: `YES`.
6. Dev-B2 used untouched fixed states 45–49: `YES`.
7. old dev40 overlap: `0`.
8. formal100 overlap: `0`.
9. Confirmation1200 overlap: `0`.
10. Task12 archived-state overlap: `0`.
11. Attempt001 overlap: `0`.
12. Attempt002 cases/rollouts: `200 / 400`.
13. Base-10: `168/200 = 84.0%`; Snap-1: `165/200 = 82.5%`; paired difference `-1.5 pp`.
14. harmful/preserved: `15 / 153`; student-only/both-fail: `12 / 20`.
15. H1 prospective replication: `True`.
16. H2 prospective replication: `False`.
17. H3 prospective replication: `False`.
18. Dq/Dp/DR at h1/h3/h5/h10 are fully reported in `ATTEMPT002_TRANSITION_EFFECTS.json`; family gates: `{'joint': {'case_positive_horizons': 0, 'task_positive_horizons': 2, 'all_horizon_loto_positive': True, 'passes': False}, 'eef_position': {'case_positive_horizons': 4, 'task_positive_horizons': 3, 'all_horizon_loto_positive': True, 'passes': True}, 'eef_orientation': {'case_positive_horizons': 3, 'task_positive_horizons': 1, 'all_horizon_loto_positive': True, 'passes': False}}`.
19. Base-visited raw-action point effect was weaker: `True`; full transition control is reported component-wise.
20. Cross-task robustness uses task-cluster bootstrap and 40 leave-one-task-out exclusions; family results are not allowed to rely on one task.
21. Pseudo-replication: `NO`.
22. Adaptive Attempt002 expansion: `NO`.
23. Metric/horizon/hypothesis changes based on Attempt001 outcomes: `NO`.
24. New model training: `NO`.
25. Task14 method development authorized by current evidence: `NO`.

## Interpretation

Attempt002 reproduced the closed-loop success ordering seen descriptively in Attempt001, but the gap shrank from -5.0 pp to -1.5 pp. The scientific mechanism decision is determined only by the frozen H1/H2/H3 analyses above, not by this success-rate direction alone. Attempt001 remains secondary/descriptive and is not pooled into the primary gate.
