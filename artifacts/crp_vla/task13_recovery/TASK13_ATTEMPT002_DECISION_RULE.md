# CRP-VLA Task 13 Decision Rule

**Status: TASK13_DECISION_RULE_FROZEN**

Infrastructure or provenance failure yields `TASK13_INVALID` and no scientific conclusion.

H1 is supported when the Snap-visited harmful-minus-preserved raw-action case-bootstrap interval is positive and its point effect exceeds the identically computed Base-visited point effect. H2 is supported when at least two of joint, EEF-position, and EEF-orientation families have positive case-bootstrap lower bounds at three or more of four frozen horizons, positive task-cluster lower bounds at two or more horizons, and leave-one-task-out signs are not controlled by one task.

H3 uses only EEF-position divergence at h=5 versus raw action distance. “More informative” requires positive task-bootstrap lower bounds for both leave-one-task-out AUROC and AUPRC differences; Brier is reported without changing this gate.

- H1 + H2, with stable Snap-over-Base specificity: `CLOSED_LOOP_IMPACT_REPLICATED`.
- H2 but H3 not supported: `CLOSED_LOOP_AMPLIFICATION_REPLICATED_BUT_NO_PREDICTIVE_ADVANTAGE`.
- Some preregistered family/horizon effects replicate but H2 is not met: `PARTIAL_MECHANISM_REPLICATION`.
- No preregistered mechanism family replicates: `MECHANISM_NOT_REPLICATED`.

The optional tag `IMPACT_MORE_INFORMATIVE_THAN_RAW_ACTION_ERROR` is emitted only if H3 passes. No result authorizes training in Task 13.
