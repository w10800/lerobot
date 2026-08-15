# Checkpoint Selection Robustness

**The D-014 selected checkpoint remains frozen at 20k regardless of this audit.**

- Eligible checkpoints: `[1000, 3000, 5000, 20000, 30000]` (10k is descriptive-only and excluded).
- 20k case-bootstrap selection frequency: `0.492`.
- Gap to second-highest case-bootstrap frequency: `0.288`.
- Leave-one-case-out 20k retention: `1.000`.
- Leave-one-task-out 20k retention: `1.000`.
- A single task can change the bootstrap selector: `False`.
