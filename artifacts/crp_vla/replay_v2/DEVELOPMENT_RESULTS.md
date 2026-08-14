# Replay-v2 Development Results

**Status: DIAGNOSTIC/DEVELOPMENT. The original formal gate remains FAIL; CRP remains HOLD.**

## Six-arm success and latency

| Arm | Success | Rate | Median policy latency |
|---|---:|---:|---:|
| base10 | 33/40 | 82.5% | 262.30 ms |
| base2 | 30/40 | 75.0% | 113.93 ms |
| base1 | 32/40 | 80.0% | 99.48 ms |
| snap10 | 31/40 | 77.5% | 267.86 ms |
| snap2 | 29/40 | 72.5% | 115.40 ms |
| snap1 | 31/40 | 77.5% | 100.70 ms |

## Required answers

1. 2 NFE recovery over 1 NFE: Base -5.0%; Snap -5.0%.
2. 10→2 versus 2→1 success loss: Base +7.5% versus -5.0%; Snap +5.0% versus -5.0%.
3. Executed-prefix error association with failure is reported descriptively in `summary.json`; it is not interpreted causally.
4. Gripper class-disagreement association is reported on the same cases; unavailable transition timing support is not imputed.
5. Base and Snap NFE patterns are shown separately above; no equality is assumed.
6. Observed 2-NFE latency increment: Base +14.45 ms; Snap +14.70 ms, for success changes -5.0%/-5.0% respectively.

## Evidence boundary

All statistics use the frozen 40-case development set. Failure labels are emitted only when registered trajectory/contact evidence supports them; ambiguous cases remain UNKNOWN. This set may be used for the pre-registered mature-baseline checkpoint selection, but it is not a confirmation set.
