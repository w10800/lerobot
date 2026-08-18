# Confirmation Blinding and Aggregation

- Before all primary case/arm records are `COMPLETED` and pass integrity checks, operators may view only `COMPLETED / FAILED / INVALID` infrastructure status.
- No live arm success rate, paired difference, aggregate table, early stopping, or threshold revision is permitted.
- The final aggregator rejects missing, duplicate, failed, invalid, diagnostic-only, or overlapping primary records.
- Diagnostic arms are keyed to the frozen diagnostic subset and excluded from the primary aggregator.
- Aggregation becomes available only after the exact primary manifest cardinality is complete and all provenance/overlap/hash gates pass.
