# CRP-VLA decision log

## D-001 — primary codebase and base revision

- Date: 2026-08-13
- Decision: use Hugging Face LeRobot at base commit `a3a8653a5b569e8e70cd6fe4b0973c4b2b983120` as the primary repository.
- Reason: official SmolVLA implementation, checkpoint/config integration, and current LIBERO support.
- Consequence: upstream changes require an explicit upgrade decision and re-running numerical-equivalence tests.

## D-002 — external repository management

- Date: 2026-08-13
- Decision: keep LIBERO-CF and implementation references as pinned Git submodules.
- Reason: immutable provenance without copying or silently diverging external code.

## D-003 — training boundary

- Date: 2026-08-13
- Decision: stop the current milestone before formal SnapFlow/CRP training. Implement and validate PR1–PR5 first.
- Reason: the compression-induced response gap is the prerequisite evidence for CRP.

## D-004 — model scope

- Date: 2026-08-13
- Decision: use SmolVLA first; defer openpi/pi0.5 and gesture/GesVLA extensions.
- Reason: isolate the core method and fit single-A100 development constraints.

## D-005 — evidence language

- Date: 2026-08-13
- Decision: all proposed effects and thresholds remain `UNVERIFIED` until reproduced locally. Paper claims are attributed as external evidence.
- Reason: prevent plans or reported literature results from being mistaken for project results.

## D-006 — target-host interpreter and memory boundary

- Date: 2026-08-13
- Decision: use an explicit uv-managed Python 3.12 environment on the A100 host and treat 40GB as a hard measured memory boundary.
- Reason: the project tooling targets Python 3.12, while automatic resolution selected unclassified Python 3.14; the assigned A100 is the 40GB PCIe variant.
- Consequence: benchmark the registered configuration unchanged and stop for a logged decision if it exceeds memory, rather than silently reducing the workload.

## D-007 — teacher-response filter for diagnostic pairs

- Date: 2026-08-13
- Decision: before inspecting the response sweep, require the 10-NFE teacher's normalized action-chunk response L2 norm to be at least `0.05` for a pair to enter aggregate compression-gap analysis.
- Reason: pairs to which the teacher is effectively condition-blind cannot identify response retention; `0.05` is a provisional engineering floor in normalized action space, not a field standard.
- Consequence: report both input and retained counts, never tune this threshold after seeing method labels, and keep below-threshold records as validity failures rather than deleting them.

## D-008 — formal-training gate remains NO-GO after the A100 smoke

- Date: 2026-08-13
- Decision: keep formal SnapFlow and CRP training blocked. The 1k-step SnapFlow run is admitted only as an engineering smoke and diagnostic checkpoint.
- Verified evidence: the A100 training path completed 1,000 steps without NaN, OOM, or data errors; all six registered offline pairs passed the pre-declared teacher-response threshold; the conditional-response gap remained after the 1k smoke, with mixed task-group behavior.
- Missing decisive evidence: no registered ordinary-LIBERO rollout set yet establishes that the 10-NFE teacher and 1-NFE student differ in standard success by no more than about three percentage points. The offline effect also has only two task groups, two intervention classes, and three action-noise seeds, and the SnapFlow smoke improved one group while degrading the other.
- Consequence: do not launch the 3k/5k/10k/30k sequence or CRP sweep. First run a registered, adequately replicated ordinary-LIBERO teacher/student comparison and log its immutable rollout manifest; then revisit the continuation rule without changing its thresholds after seeing results.

## D-009 — replace project-level NO-GO wording with CRP Training HOLD / Baseline Evaluation GO

- Date: 2026-08-13
- Decision: engineering implementation and the SnapFlow reproduction path are GO; baseline attribution/evaluation is GO; formal CRP and 3k+ SnapFlow training remain HOLD.
- Verified evidence: loss accounting closes with zero residual; stable latency establishes a 3.49× 10→1 median speedup; 40 semantic pairs × 3 matched-noise seeds complete the four-arm attribution; and a 3-case ordinary-LIBERO paired pilot completes with immutable state/input/noise/action provenance.
- Scientific interpretation: Base 10→1 has a substantial offline conditional-response gap. The 1k SnapFlow checkpoint does not enlarge the within-checkpoint reduction gap and slightly reduces it on aggregate. Snap-1 nevertheless shows small final response-direction/amplitude shifts relative to Base-1 because training drift and reduction interact. The earlier two-pair action-fit/response-decoupling signal is only partially reproduced and is not a general causal result.
- Remaining gate: the ordinary-LIBERO pilot has only three paired cases and confidence intervals too wide to establish the predeclared approximately 3-point non-inferiority condition.
- Consequence: continue replicated baseline evaluation and mechanism diagnostics; do not start 3k/5k/30k SnapFlow or CRP training until a new decision explicitly admits the ordinary-LIBERO gate.

## D-010 — freeze the 100-case ordinary-LIBERO paired gate

- Date: 2026-08-13
- Decision: expand the successful 3-case engineering pilot to exactly 100 task/init-state cases before inspecting additional rollout outcomes. Cover all 40 tasks in `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` with init states 0/1, plus init state 2 for task IDs 0–4 in every suite.
- Primary comparison: Snap-1 minus Base-10 paired success difference, with a non-inferiority margin of `-0.03`. Report the 95% paired case bootstrap interval and both discordant directions; Base-1 and Snap-10 remain mandatory attribution arms.
- Admission rule: the formal gate can pass only if the lower 95% interval bound is at least `-0.03`, provenance/invariant checks pass, and no task/suite-level catastrophic regression is hidden by the aggregate. A point estimate alone cannot pass.
- Durability: write a partial manifest after every arm so a simulator or provenance failure cannot erase completed rollout evidence. Partial manifests are not final results and are never silently resumed.

## D-011 — ordinary-LIBERO non-inferiority gate fails; formal training remains HOLD

- Date: 2026-08-14
- Decision: mark the D-010 gate FAIL and keep 3k/5k/30k SnapFlow plus CRP training blocked.
- Verified result: all 100 cases and 400 rollouts completed at clean remote commit `bc0a9b3d79cf3ce77a17eaf76785be901e36981e`. Base-10/Base-1/Snap-10/Snap-1 success was `86/83/84/81%`. Snap-1 minus Base-10 was `-5%` with case-level paired bootstrap 95% interval `[-13%, +2%]`; negative/positive discordance counts were 10/5.
- Gate evaluation: the lower interval bound is below the frozen `-3%` margin, so non-inferiority is not established. No suite crossed the `<=-10%` catastrophic threshold, but spatial and object each had a `-8%` Snap-1 point difference and task-level negative clusters exist.
- Interpretation: the 1k checkpoint is not an admissible mature baseline for the project's intended “ordinary success preserved” phenomenon. Snap-10 also differs from Base-10, so training drift and reduction effects must remain separate.
- Consequence: do not reinterpret the failed gate as authorization to train. A future mature-baseline training proposal requires an explicit contract amendment and preregistration before execution, followed by the same frozen ordinary-LIBERO gate; CRP remains HOLD regardless.

## D-012 — PROPOSED mature-baseline operating contract; no training authorization

- Date: 2026-08-14
- Status: **PROPOSED**, not APPROVED and not GO.
- Proposed decision: permanently archive the 1k checkpoint, convert the current 100 cases to development/diagnostic use, preregister a disjoint 1,880-case confirmation set, and evaluate one checkpoint selected from a single continuous 3k/5k/10k/20k/30k SnapFlow trajectory using ordinary factual development criteria only.
- Diagnostic basis: the NFE=2 closed-loop extension is unverified because exact cross-process canonical inputs could not be recovered; offline six-arm response error is partially restored at 2 NFE, while matched-noise distortion exceeds condition-change distortion; all 15 formal discordant behavior categories remain UNKNOWN.
- Scientific wording: current evidence supports generic one-step action/transport distortion more strongly than grounding degradation or condition-specific distortion.
- Consequence: D-011 remains controlling. Do not start the proposed maturation run, any CRP/CAG/KD training, or hyperparameter search until a separate human-reviewed decision changes this proposal to APPROVED. CRP remains HOLD until a mature checkpoint passes the new confirmation gate.

## D-013 — APPROVE one preregistered SnapFlow baseline-maturation run; CRP remains HOLD

- Date: 2026-08-14
- Status: **APPROVED / GO for the exact baseline run below only**.
- Authorization: after replay-v2 Phase A completed 8/8 capsules and 48/48 rollouts with zero infrastructure exceptions, the user explicitly approved tasks 6–8 and the single mature-baseline run specified in task 8.
- Scope: one continuous 30,000-step SnapFlow run from Base revision `31d453f7edd78c839a8bbc39744a292686daf0de`, seed 1000, fixed data order, bf16, batch 4, learning rate `2.5e-5`, alpha `0.5`, shortcut weight `0.1`, gradient clip `1.0`, and the full 40-task pinned LIBERO dataset. Save exact steps 1k/3k/5k/10k/20k/30k. No model, objective, or loss change is permitted.
- Selection boundary: freeze `artifacts/crp_vla/maturation/CHECKPOINT_SELECTION_PROTOCOL.md` before execution. Selection may use only stability, the frozen ordinary factual validation fixture, and replay-v2 development Snap-1 success. LIBERO-CF, condition-response/CRP metrics, and any confirmation set are forbidden.
- Unchanged evidence: D-011's original Formal LIBERO Gate remains FAIL; the existing 1k checkpoint remains a pilot. This decision does not retrospectively pass it.
- Consequence: D-011/D-012 are overridden only for this baseline-maturation run. CRP, CAG, factual KD, response-preserving losses, hyperparameter sweeps, confirmation evaluation, and all new-method training remain HOLD. Stop after selecting and documenting one checkpoint.

## D-014 — SELECT the 20k mature SnapFlow development candidate; formal gate unchanged

- Date: 2026-08-15
- Status: **SELECTED for development / not confirmed**.
- Decision: select the uninterrupted maturation trajectory's 20,000-step checkpoint under the D-013 preregistered rule. It passed stability and factual eligibility, then uniquely maximized eligible replay-v2 development Snap-1 success at `32/40`.
- Verified evidence: all six registered checkpoints completed the same frozen 40-case Snap-10/Snap-2/Snap-1 replay, all 720 rollout records were `COMPLETED`, and all 720 trace-manifest paths and content hashes were unique and recomputed successfully. The 20k factual loss is `0.0734668181`; its Snap-10/Snap-2/Snap-1 successes are `30/32/32` of 40, and its model SHA-256 is `3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53`.
- Evidence boundary: this is a development-selected baseline candidate, not a confirmation result and not a replacement for D-011's failed Formal LIBERO Gate. The frozen development cases were used for selection and cannot serve as confirmation evidence.
- Consequence: the D-013 baseline-maturation authorization is complete and stops here. Any disjoint confirmation evaluation requires separate authorization. CRP, CAG, factual KD, response-preserving losses, hyperparameter sweeps, LIBERO-CF training, and all new-method training remain HOLD.

## D-015 — Task 9 preflight is ready; 20k selection and training HOLD remain unchanged

- Date: 2026-08-15
- Status: **READY_FOR_DISJOINT_CONFIRMATION / no rollout authorization**.
- Decision: retain the D-014 20k checkpoint exactly as selected. Task 9 completed the preregistered robustness, closed-loop divergence, prospective sample-size, disjoint-manifest, blinding, and preflight audits without training or running a confirmation primary case.
- Verified evidence: 20k was selected in `49.16%` of 10,000 case bootstraps and `48.10%` of 10,000 task-cluster bootstraps; leave-one-case-out and leave-one-task-out retention were both `100%`, and no single task determined the observed selection. Against Base-10 on the 40 development cases, 20k Snap-1 differed by `-2.5` percentage points with paired case-bootstrap 95% interval `[-12.5%, +5.0%]`, task-cluster interval `[-10.0%, +5.0%]`, and exact McNemar `p=1.0`.
- Predictive boundary: only the matched initial replan was queryable. Leave-one-task-out combined disagreement did not provide useful held-out-task failure prediction (AUROC `0.270`, AUPRC `0.150`, sensitivity `0` at the frozen specificity target); later failure-relative timing and multi-noise variance remain unavailable. No causal mechanism is claimed.
- Prospective design: the frozen 600-case primary manifest has zero state-identity overlap with formal100 and dev40 and includes a deterministic 200-case diagnostic subset. Under the reused paired-percentile-bootstrap procedure and pooled observed discordance, the 600-case design passed only `58.8%` of simulations at true delta zero; Task 9 therefore recommends 1,200 primary cases for planning but does not expand the manifest or execute them.
- Consequence: Task 9's operational preflight is complete, but neither the 600-case rollout nor a 1,200-case expansion is authorized by this decision. The original Formal ordinary-LIBERO gate remains FAIL, 20k remains selected for development and not confirmed, and CRP/CAG/KD/new-method training remains HOLD.

## D-016 — FREEZE the Task 10 1,200-case protocol and trajectory instrumentation; no rollout authorization

- Date: 2026-08-16
- Status: **APPROVED for freeze + instrumentation + preflight only**.
- Decision: extend the frozen Task 9 design from 40 tasks × 15 states to exactly 40 tasks × 30 states. Preserve all 600 Task 9 cases byte-semantically as an explicit subset and add only fixed state indices 20–34 for each task. Freeze a new deterministic, outcome-blind five-cases-per-task diagnostic subset with seed `20260816` before any confirmation outcome exists.
- Primary boundary: the only primary comparison is Base-10 versus the D-014 20k Snap-1 checkpoint; the non-inferiority margin remains `-0.03`. Success definition, evaluator, simulator termination, execution horizon, normalization, model weights, and checkpoint selection are unchanged. The primary aggregator must refuse output unless all 1,200 paired cases are complete and must reject diagnostic or non-primary contamination.
- Instrumentation boundary: every replan must preserve full raw and normalized observations, serialized simulator/qpos/qvel/object/robot/gripper state, policy/noise/action tensors, processor provenance, and objective simulator events. Counterfactual branches restore the identical frozen state independently and log unweighted raw transition-divergence components at horizons 1/3/5/10; no outcome-tuned combined scalar is allowed.
- Consequence: `READY_FOR_1200_CONFIRMATION`, if reached, means infrastructure readiness only. It does not confirm 20k, pass the Formal ordinary-LIBERO Gate, authorize a confirmation rollout, or authorize CRP/CAG/KD/new-method training. All such training remains HOLD.

## D-017 — Task 10 infrastructure is READY_FOR_1200_CONFIRMATION; execution and training remain unauthorized

- Date: 2026-08-16
- Status: **READY_FOR_1200_CONFIRMATION / no rollout authorization**.
- Decision: accept the clean-attempt002 1,200-case manifest, frozen 200-case diagnostic subset, per-replan trace schema, exact state-restoration contract, isolated counterfactual branch interface, raw transition metrics, strict primary aggregator, and attempt006 preflight as the authoritative Task 10 infrastructure package.
- Verified evidence: 40 tasks × 30 cases; an exact 600-case Task 9 subset plus 600 fixed new cases; 1,200 unique initial-state, simulator-state, and qpos/qvel hashes; zero formal100/dev40 overlap; 31/31 preflight checks; exact physical transition repeatability after full wrapper reset; horizons 1/3/5/10; 31 relevant remote tests passed; 74/74 artifact checksums verified. All admitted build/preflight failures remain immutable and disclosed, including R-014 render variance.
- Evidence boundary: no confirmation primary case was executed, no confirmation outcome was aggregated, and no model was trained or reselected. The original Formal ordinary-LIBERO Gate remains FAIL; 20k remains SELECTED FOR DEVELOPMENT / NOT CONFIRMED.
- Consequence: stop Task 10. A separate explicit authorization is required to execute the frozen 1,200-case confirmation. CRP/CAG/factual-KD/response-preserving/new-method training remains HOLD regardless of this infrastructure readiness decision.

## D-018 — Task 11 confirms 20k Snap-1 non-inferiority; authorize Task 12 mechanism audit

- Date: 2026-08-16
- Status: **FORMAL_CONFIRMATION1200_PASS / GO for Task 12 mechanism analysis**.
- Decision: accept the frozen 1,200-case Base-10 versus 20k Snap-1 result as the authoritative ordinary-LIBERO confirmation. Base-10 succeeded on `969/1200` cases and 20k Snap-1 on `959/1200`; the paired difference was `-0.008333` with frozen paired-percentile-bootstrap 95% interval `[-0.029167, +0.0125]`. The lower bound is above the unchanged `-0.03` non-inferiority margin.
- Integrity evidence: `2400/2400` arm results, `1200/1200` exact pairs, `40/40` shard statuses, zero arm-level failures/retries, 47,276 replan archives and 94,552 capsule members verified, and 4,895/4,895 finalizer checksum entries passed. One pre-arm state-identity event was recovered by exactly reproducing the frozen manifest-builder reset sequence; a duplicate-writer guard later stopped a second writer without overwrite. Both events are disclosed in the execution addendum and do not change the frozen primary statistic.
- Interpretation: the project now has a verified mature one-step baseline that preserves aggregate ordinary-LIBERO success within the registered margin. It does not establish that local trajectories are preserved, and it does not identify a causal mechanism.
- Authorization: run Task 12 Phase A on the complete archived trajectories, then the separately frozen 200-case student-visited same-state/counterfactual audit. The user has authorized autonomous continuation and training adjustment, but Phase A alone cannot authorize training. A single scoped exploratory preservation probe may be approved only after Phase B localizes a repairable mechanism and a separate frozen training decision is logged.
- Exclusions: no broad hyperparameter sweep, no confirmation-set checkpoint selection, no LIBERO-CF training, and no claim that a Task 12 probe is a confirmed method result.

## D-019 — APPROVE one exploratory student-state action-preservation probe

- Date: 2026-08-16
- Status: **APPROVED / single fixed exploratory probe only**.
- Verified trigger: Task 12 analyzed all 1,200 paired trajectories, then exactly replayed Base-10 and 20k Snap-1 on 7,772 archived states with 7,772/7,772 exact self-replays. On Snap-visited states, harmful cases had higher same-state action disagreement than preserved cases, while the analogous Base-visited contrasts crossed zero. The isolated-branch audit restored 3,842 Snap-visited states and completed 7,684/7,684 branches. Harmful-minus-preserved 95% intervals were strictly positive for joint and end-effector divergence at horizons 1/3/5/10; object-position intervals crossed zero.
- Decision: authorize one 500-step continuation of the frozen 20k checkpoint. Each update contains one Snap-visited observation/noise pair trained toward the cached Base-10 normalized first-10 action prefix and one Base-visited pair anchored to the original Snap-1 prefix. Use the outcome-blind one-case-per-shard held-out split, seed `20260816`, AdamW learning rate `1e-6`, bf16, equal loss weights, gradient clip `1.0`, and save only the fixed final checkpoint.
- Required preflight: verify same-state action round-trip on 64 deterministic samples, a finite nonzero gradient, and exact tensor identity after a zero-learning-rate optimizer step. Any failure stops the probe before an admitted update.
- Evaluation boundary: use the 40 held-out diagnostic cases for offline exploratory loss only and the pre-existing replay-v2 dev40 capsules for closed-loop exploratory evaluation. Do not rerun or reinterpret Task 11 as a post-training confirmation set. No intermediate checkpoint selection, broad sweep, LIBERO-CF training, or confirmed-method claim is authorized.

## D-020 — COMPLETE Task 12 without promoting the exploratory probe

- Date: 2026-08-16
- Status: **TASK12_COMPLETE / exploratory probe NOT PROMOTED**.
- Verified result: the single D-019 run reduced held-out Snap-visited Base-target normalized-prefix MSE from `0.01165795` to `0.01040619` (10.7% relative), while Base-visited original-Snap anchor drift was `0.00035369`. On the pre-existing frozen replay-v2 dev40 cases, Snap-10 stayed `30/40`, Snap-2 changed `32/40` to `34/40`, and the primary Snap-1 endpoint changed `32/40` to `30/40`. For Snap-1, the paired difference was `-0.05`, 95% case/task-cluster bootstrap intervals were both `[-0.15, +0.05]`, and exact McNemar `p=0.625` (`n11/n10/n01/n00 = 29/1/3/7`, where `n10` is probe-only success).
- Integrity: all 120 probe rollout records completed under matched replay-v2 invariants; all 120 trace paths and recomputed content hashes were unique. Exactly one attempt admitted optimizer updates, no checkpoint sweep occurred, and Task 11 confirmation cases were not rerun for post-training selection.
- Decision: do not promote or further tune this checkpoint. Retain the D-014/D-018 20k Snap-1 checkpoint as the confirmed project baseline. The offline objective is informative for mechanism diagnosis but is not an adequate closed-loop selection objective.
- Next-study boundary: any additional training requires a new preregistered protocol with a disjoint closed-loop development set and must preserve Task 11 as untouched confirmation evidence. Task 12 itself is complete.

## D-021 — FREEZE Task 13 prospective closed-loop impact validation

- Date: 2026-08-16
- Status: **APPROVED / GO for Task 13 validation only; NO TRAINING**.
- Decision: build outcome-blind Dev-B from fixed ordinary-LIBERO state indices 35–44 for all 40 tasks, targeting exactly 400 cases. Reject any overlap or duplicate without replacement. Use only Base-10 and the confirmed original 20k Snap-1 checkpoint; the Task 12 probe is excluded.
- Frozen hypotheses: prospectively test student-visited distribution specificity, joint/EEF trajectory amplification at horizons 1/3/5/10, and whether the single frozen EEF-position h=5 predictor improves leave-one-task-out discrimination over the Task 12 raw first-10-action discrepancy.
- Frozen analysis: harmful is Base-success/Snap-failure and preserved is both-success. Aggregate state metrics to case means; retain separate case and task-cluster bootstraps, component-wise transition metrics, Base-visited/random-state/action-norm/horizon-zero controls, and the five permitted Task 13 terminal states.
- Consequence: Task 11 remains locked, no checkpoint/loss/horizon/metric selection is allowed after Dev-B outcomes, and Task 13 cannot train or automatically authorize Task 14.

## D-022 — QUARANTINE Task 13 attempt001 and authorize fixed recovery attempt002

- Date: 2026-08-16
- Status: **TASK13_ATTEMPT001_PROCEDURALLY_INVALID / GO for recovery attempt002 only**.
- Decision: permanently classify attempt001's 800/800 technically complete rollouts and Base-10/Snap-1 `336/400` versus `316/400` result as `DESCRIPTIVE_ONLY`. The pre-outcome builder incorrectly consumed a case-aggregated Task 12 file and checked zero archived states, so attempt001 cannot support primary H1/H2/H3 conclusions even if a post-hoc audit finds zero actual overlap.
- Recovery authorization: construct a typed used-state registry from the 40 authoritative Task 12 raw query shards and replan metadata; retain distinct full-state, state-blob, qpos, qvel, branch, initial-state, and case identities. Then freeze exactly states 45–49 for all 40 tasks as untouched Dev-B2, fail closed without replacement, and commit the registry snapshot, 200-case manifest, byte-identical metrics, and byte-identical decision rule before any attempt002 rollout.
- Scientific boundary: attempt002 is the only prospective primary replication; N=200 cannot be adaptively expanded. Attempt001 may be secondary supporting evidence only after attempt002 primary analysis and only if its post-hoc audit is disjoint. No hypothesis, metric, horizon, bootstrap, predictor, checkpoint, or training objective may change.

## D-023 — COMPLETE Task 13 recovery as partial mechanism replication; do not authorize Task 14

- Date: 2026-08-17
- Status: **PARTIAL_MECHANISM_REPLICATION / TASK 14 NOT AUTHORIZED**.
- Verified closed-loop result: Attempt002 completed all 400 arm rollouts for 200 frozen Dev-B2 cases. Base-10 succeeded on `168/200` (`84.0%`) and the original confirmed 20k Snap-1 on `165/200` (`82.5%`); the paired difference was `-0.015` with case-bootstrap 95% interval `[-0.065, +0.035]`. The frozen strata were preserved `153`, harmful `15`, student-only `12`, and both-fail `20`.
- Mechanism decision: H1 passed. On Snap-visited states, harmful-minus-preserved raw-action discrepancy was `0.1561079`, with case-bootstrap 95% interval `[0.0693539, 0.2502889]` and task-cluster interval `[0.0603199, 0.2803984]`; the Base-visited control contrast was `0.0265475` and crossed zero under both resampling schemes. H2 failed its preregistered two-family gate: EEF-position passed, but joint and EEF-orientation did not pass all case/task/leave-one-task-out criteria. H3 failed: the frozen EEF-position h5 transition feature did not improve leave-one-task-out prediction over raw-action discrepancy (AUROC difference `-0.0980392`, task-bootstrap 95% interval `[-0.3449463, +0.0706665]`).
- Audit correction: the original Attempt001 post-hoc overlap report counted 1,200 self-matches because the attempt001 dataset was included while auditing itself. The corrected read-only audit excludes the self dataset and finds zero overlap with prior Task 9–12 identities and zero internal duplicates. Attempt001 nevertheless remains permanently `DESCRIPTIVE_ONLY` because its disjointness was not established before outcome exposure; this correction does not change the prospective Attempt002 result or terminal decision.
- Decision: complete Task 13 without training, method promotion, or Task 14 authorization. Retain the original 20k Snap-1 checkpoint as the confirmed baseline. Any further training or causal mechanism intervention requires a new preregistered task and new authorization; the present evidence supports a replicated student-state action-discrepancy association, not a complete transition-amplification or prediction mechanism.

## D-024 — AUTHORIZE Task 14 audit and STOP for insufficient untouched formal states

- Date: 2026-08-18
- Status: **INSUFFICIENT_UNTOUCHED_FORMAL_STATES / NO FORMAL ROLLOUT / NO TRAINING**.
- Authorization: the user supplied and explicitly authorized the fixed `TASK14_PROSPECTIVE_POSITION_CAUSAL_AUDIT` protocol. This supersedes D-023 only for the bounded, no-training component intervention audit; it does not authorize model updates or method development.
- Verified gate result: every one of the 40 ordinary-LIBERO tasks has exactly 50 pinned initial states. Historical manifests cover old dev40, formal100, Confirmation1200, Task 13 Attempt001, and Task 13 Attempt002. Their union leaves only state 3 for every task and additionally state 2 for task IDs 5–9: 60 untouched task/state pairs total, 1–2 per task, versus the frozen requirement of 10 per task and 400 total.
- Decision: stop before formal preregistration, manifest creation, outcome reveal, or rollout. Do not reuse old states, adaptively expand, generate replacement simulator states, or report an A–F causal conclusion. H14.1–H14.4 remain untested, and Task15 limited method development is not authorized.
- Engineering evidence: the seven-dimensional relative OSC action is semantically identifiable as position dimensions 0–2, relative axis-angle orientation dimensions 3–5, and gripper direction dimension 6. Pure component-composition, clipping-audit, seed, and capacity-gate tests passed, but no formal intervention arm was executed after the capacity stop.

## D-025 — AUTHORIZE TASK14R fresh-state recovery without formal outcome reveal

- Date: 2026-08-18
- Status: **APPROVED for engineering dry run + policy-independent state-bank recovery / NO FORMAL OUTCOME / NO TRAINING**.
- Decision: name the successor `TASK14R_FRESH_STATE_BANK_RECOVERY`, not Task15. Phase A may use exactly the 15 already revealed Attempt002 harmful cases to validate live-observation Base/Snap queries and the six BaseRepeat/SnapRepeat/NoOp/PosSwap/RotSwap/FullSwap arms. Every result is permanently `ENGINEERING_ONLY / OUTCOME_CONDITIONED / NON_PRIMARY`; exact NoOp/SnapRepeat and FullSwap/BaseRepeat stepwise closed-loop identity are mandatory engineering gates.
- State-bank boundary: freeze exactly 20 outcome-blind generated candidates per standard task, accept only simulator/task validity and reproducibility checks, hash-sort the first 10 accepted states per task into a 400-case formal bank, and keep the rest as nonadaptive reserve. A separate 64-seed representative probe per suite must yield at least 51 valid unique reproducible states. No policy, checkpoint, Base/Snap outcome, discrepancy, or harmful label may enter generation or selection.
- Distribution boundary: compare the new bank against the released 50 states per task under the frozen component-wise support audit. Generated states are a simulator-generated evaluation distribution, not official LIBERO test states. Stop with `GENERATED_STATE_DISTRIBUTION_MISMATCH` if the registered global mismatch rule fires; otherwise emit `READY_FOR_TASK14_CAUSAL_RELAUNCH` only after all engineering, capacity, 400-case, disjointness, reproducibility, and distribution gates pass.
- Consequence: this decision does not authorize new formal Base/Snap/PosSwap/RotSwap/FullSwap outcomes. Those require a separate `TASK14C_PROSPECTIVE_POSITION_CAUSAL_AUDIT`. Do not reduce 400 to 60, reinterpret Attempt002 interventions as causal evidence, train a method, or begin Task15.

## D-026 — QUARANTINE Task14R Phase A attempt001 before the first rollout

- Date: 2026-08-18
- Status: **PHASE_A_ATTEMPT001_STARTUP_FAILED / ZERO ROLLOUT / NO FORMAL OUTCOME / NO TRAINING**.
- Verified event: Phase A attempt001 started from clean commit `3425d7b1bee81722bac77e1e829c768fdd786b44` with the frozen 15-case, six-arm command. Before the first case or arm, the Base checkpoint preprocessor attempted to resolve `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` over the Hub because its saved tokenizer name was not covered by the runner's local VLM override. The process was interrupted while blocked in `AutoTokenizer.from_pretrained`; no rollout, simulator step, policy outcome, partial case result, or parameter update was admitted.
- Repair boundary: preserve attempt001 and its console log unchanged. Bind both policy preprocessors to the frozen local tokenizer directory, require its tokenizer files, and force policy weight loading to local files. This is a startup-path correction only; it does not alter the frozen cases, states, noises, arms, checkpoints, action composition, evaluator, or identity gates.
- Consequence: any replacement run requires a new clean commit, a new attempt namespace, and explicit confirmation of its exact command. Phase B, Phase C, Task14C, formal outcomes, PR merge, and training remain unauthorized.

## D-027 — QUARANTINE Task14R Phase A attempt002 before the first case

- Date: 2026-08-18
- Status: **PHASE_A_ATTEMPT002_STARTUP_FAILED / ZERO ROLLOUT / NO FORMAL OUTCOME / NO TRAINING**.
- Verified event: attempt002 ran from clean repair commit `ef91fa404bdb06a923b4d3168c0524b58f67bf4d`. Both Base and Snap policies and preprocessors loaded from local files with no network connection, confirming the D-026 tokenizer repair. Before entering the case loop, the runner stopped with `FileExistsError`: standard-LIBERO configuration had created the attempt root before a later fail-closed `mkdir(exist_ok=False)` tried to reserve the same root.
- Repair boundary: preserve attempt002 unchanged. Reserve the unused attempt root exactly once before standard-LIBERO configuration writes its child directory, retain `exist_ok=False`, and remove only the later duplicate reservation. Frozen cases, policies, states, noises, arms, execution, evaluator, and identity gates remain unchanged.
- Consequence: attempt002 admits zero case/arm results and no intervention evidence. Any replacement requires another clean commit, a new attempt namespace, and explicit exact-command confirmation; all later phases and training remain unauthorized.

## D-028 — STOP Task14R Phase A after historical same-state contract failure

- Date: 2026-08-18
- Status: **TASK14R_PHASE_A_ENGINEERING_GATES_FAILED / NO REPLACEMENT AUTHORIZED**.
- Verified event: attempt003 ran from clean commit `34473e4b7aa44bd7cdb79c2acb9c5edb49aaac4f` and completed only BaseRepeat, SnapRepeat, and NoOp for the first frozen harmful case. BaseRepeat reproduced historical success, but SnapRepeat succeeded instead of reproducing historical failure, and NoOp then failed despite using the same Snap action at step zero. The process was stopped before PosSwap, RotSwap, or FullSwap and before a complete six-arm case existed.
- Root cause: archived Base/Snap branches restored identical flattened MuJoCo state, qpos, and qvel but not identical fixed-fixture model state. LIBERO hard reset repeatedly appends `object_property_initializers`; the same seed therefore consumes 3, then 4, then 5 property draws, changing `sim.model.body_pos`, which is absent from the saved flattened state. In the first case, cabinet and wine-rack placement drift changed both camera streams immediately; the first processed-policy and action mismatch followed at the next replan.
- Scope audit: 5/15 frozen harmful pairs and 15/200 complete Task13 Attempt002 pairs have unequal archived physical-state hashes despite 200/200 equal state-blob/qpos/qvel hashes. Mismatches are outcome-imbalanced: harmful `5/15`, preserved `7/153`, student-only `3/12`, both-fail `0/20`. Task13 Attempt002 aggregate outcomes remain historical observations, but its strict matched-state mechanism interpretation is quarantined because the prospective pairing contract was violated for a nonrandom subset.
- Decision: Phase A terminates as an engineering failure. Do not launch attempt004, interpret PosSwap/RotSwap, or proceed to Phase B, Phase C, Task14C, formal outcomes, training, or PR merge. A successor would require a new protocol that serializes/restores complete model state and uses uncontaminated engineering cases; `hard_reset=False` alone is not approved because the diagnostic observed stale controller-cache differences on its first repeat.

## D-029 — FREEZE Task14R reset-transaction R0; execution requires the new clean commit

- Date: 2026-08-18
- Status: **TASK14R_R0_PROTOCOL_FROZEN / POLICY-FREE EXECUTION PENDING / NO FORMAL OUTCOME / NO TRAINING**.
- Authorization: after reviewing the successor design, the user authorized continued implementation. The successor is named `TASK14R_RESET_TRANSACTION_RECOVERY`; it is not attempt004 and does not rehabilitate the contaminated Attempt002 pairs.
- R0 contract: cover exactly one fixed official state for each of all 40 ordinary-LIBERO tasks, with three complete-state restores and three identical ten-step policy-free action probes per task. Each task uses one compiled MuJoCo model and renderer context; no wrapper reset or hard model rebuild is permitted after capsule capture. Capture MuJoCo `mjSTATE_INTEGRATION`, compiled-model MJB plus every exposed model array hash, explicit controller/interpolator state, Panda gripper action state, robot buffers, wrapper counters, observation cache, and observable timing/value state. Any unsupported state or exact mismatch fails closed.
- Renderer contract: use EGL with MuJoCo offscreen MSAA frozen to `offsamples=0` and require exact pixels. A diagnostic with the default EGL MSAA observed three wrist-camera pixels changing by one gray level despite exact physics/controller state; OSMesa was unavailable on this host. Disabling MSAA then produced one exact three-repeat, ten-step real-LIBERO smoke. This smoke is engineering compatibility evidence only and does not pass the 40-task R0.
- Consequence: freeze the compact protocol before execution and require a clean artifact-bearing commit plus explicit command review. R0 may emit only `TASK14R_R0_RESTORE_TRANSACTION_PASSED` or `TASK14R_R0_RESTORE_TRANSACTION_FAILED`. It loads no policy and cannot automatically enter the six-arm engineering run, Phase B, Phase C, Task14C, any formal outcome, or training.

## D-030 — HARDEN Task14R R0 qualification before execution

- Date: 2026-08-18
- Status: **TASK14R_R0_AUDIT_HARDENED / POLICY-FREE EXECUTION STILL PENDING / NO FORMAL OUTCOME / NO TRAINING**.
- Decision: replace the v1 ten-step probe with a v2 fixed 15-step probe that independently excites all six continuous OSC dimensions in both directions at magnitude `0.05` and includes gripper open and close. Preserve the original policy-independent task/state seed selection and the 40-task, three-restore design.
- Audit gate: an R0 pass now requires exactly 40/40 passed tasks, zero failed tasks, 120 restore transactions, 120 complete probe trajectories, 1,800 probe steps, zero policy/formal activity, zero training or parameter updates, and no automatic next phase. Counts are terminal predicates rather than descriptive fields. Every task failure is materialized and counted; every repeat has a hash-manifested step trace with field-specific mismatch localization.
- Provenance and renderer boundary: freeze v2 schema, source hashes for runner/transaction/launcher, fixed implementation parent `20757544607413c6ab1a3458092ceec6bc1fc85d`, permitted terminal statuses, and EGL/no-MSAA renderer contract. R0 qualifies only the no-MSAA complete-state transaction and cannot establish standard-renderer equivalence; a separate policy-free renderer-shift audit remains mandatory before R1.
- Consequence: this hardening authorizes code, protocol, and test changes only. It does not launch R0, create an R0 runtime output root, load a policy, inspect Base/Snap outcomes, run a renderer-shift audit, enter Phase A/R1, Phase B, Phase C, or Task14C, or train/update parameters.

## D-031 — ACCEPT R0 no-MSAA qualification and FREEZE policy-free R0S renderer audit

- Date: 2026-08-18
- Status: **R0 PASSED / R0S PROTOCOL FROZEN / R0S EXECUTION PENDING / NO POLICY OR FORMAL OUTCOME / NO TRAINING**.
- Verified R0 result: the exact clean commit `60fbf7370d8668316b34a9d0d1a334247f792cca` completed all 40 tasks with 40 passed and zero failed, 120 complete-state restores, 120 full probe trajectories, and 1,800 fixed probe steps. Policy queries, formal cases, formal outcome rollouts, training/parameter updates, and automatic next-phase activity were all zero/false; every terminal gate was true. `TASK14R_R0_FINAL.json` has SHA-256 `ae443481ec991f6c46cb74a8bf4adf90817cd528d5f1ff4e44d04f6727df1a3e`.
- Evidence boundary: R0 establishes exact complete-state transaction repeatability only for EGL `offsamples=0`. It does not establish equivalence to the standard EGL renderer and does not rehabilitate Task13 Attempt002 or Task14R Phase A attempt003.
- R0S decision: freeze a separate policy-free 40-task paired audit with three `standard_msaa` (`offsamples=4`) and three `no_msaa` (`offsamples=0`) trajectories per task, using the same released state 0, deterministic seed, complete-state capsule, and 15 fixed actions. Require exact physical/controller/non-image identity within and across renderers plus exact no-MSAA repeat pixels. Persist every camera frame and report pixel differences without an outcome-tuned tolerance.
- Terminal interpretation: byte-exact images may return `TASK14R_R0S_RENDERERS_EXACTLY_EQUIVALENT`; nonzero images with exact physics return `TASK14R_R0S_RENDERER_SHIFT_CHARACTERIZED`; any technical/count/provenance failure returns `TASK14R_R0S_RENDERER_AUDIT_FAILED`. No R0S status automatically authorizes R1, a policy query, Phase A/B/C, Task14C, a formal outcome, training, PR creation, or merge. R0S execution requires a new clean artifact-bearing commit and separate exact-command approval.
