# CRP-VLA Task 14 Final Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: VERIFIED
- Generated unix ns: `1787038850772208850`

**Terminal status: `INSUFFICIENT_UNTOUCHED_FORMAL_STATES`**

Task 14 stopped at the mandatory pre-outcome capacity gate. No formal manifest, preregistration,
screening rollout, intervention rollout, training update, or outcome analysis was created.

## Required answers

1. Clean branch containing Task 13 commit: `YES`; branch `task14-position-causal-audit` descends from `7fd8f6cb6584a2ad12862112261bbbd8087b1e08`.
2. Base-10 checkpoint SHA-256: `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
3. Snap-1 checkpoint SHA-256: `3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53`.
4. Action schema: 7D relative OSC pose command plus gripper direction.
5. EEF position dimensions: `[0, 1, 2]`.
6. Composition space: post-unnormalization environment command space.
7. NoOp/FullSwap pure identity tests: reported by the Task 14 unit-test suite; no GPU rollout was admitted.
8. New formal states: `0` admitted. Only 1-2 untouched pinned states remain per task; 10 are required.
9. Formal overlap: not evaluated because no formal manifest could be frozen.
10. 400 cases frozen before outcome reveal: `NO`; freezing was correctly refused.
11. Adaptive expansion: `NO`.
12. Base/Snap formal success: `N/A` (0 formal rollouts).
13. Fresh strata: `N/A`.
14. Harmful task coverage: `N/A`.
15. SnapRepeat stability: `N/A`.
16. PosSwap rescue: `N/A`.
17. RotSwap rescue: `N/A`.
18. FullSwap rescue: `N/A`.
19. PosSwap-SnapRepeat effect/CI: `N/A`.
20. PosSwap-RotSwap effect/CI: `N/A`.
21. Case/task conclusion consistency: `N/A`.
22. LOTO stability: `N/A`.
23. EEF position h1/h3/h5/h10 reduction: `N/A`.
24. Intervention-specific clipping: not measured; no formal intervention ran.
25. Pseudo-replication: `NO`.
26. A-F causal status: not assigned; the earlier mandatory terminal state
    `INSUFFICIENT_UNTOUCHED_FORMAL_STATES` controls.
27. Task15 limited method development authorized: `NO`.

## Capacity evidence

All 40 task files contain exactly 50 pinned initial states. Historical manifests cover state 4
(old dev40), states 0-2 where registered (formal100), states 5-34 (Confirmation1200), states 35-44
(Attempt001), and states 45-49 (Attempt002). State 3 remains unused for every task; state 2 also
remains unused for task IDs 5-9. This leaves 60 untouched task/state pairs total and no task with
the required 10. The protocol forbids using old states to fill the deficit.

The causal hypotheses H14.1-H14.4 are `NOT TESTED`. This stop is a design-resource limitation, not
evidence for or against position-specific causal rescue.
