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
