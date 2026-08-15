#!/usr/bin/env python
"""Standalone prospective paired-Bernoulli simulation for CRP-VLA Task 9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_task9_analysis import _simulate_design_fast, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--discordance", type=float, action="append", required=True)
    args = parser.parse_args()
    rows = []
    for n in (200, 400, 600, 800):
        for delta in (0.0, -0.01, -0.025, -0.05):
            for discordance in args.discordance:
                rows.append(
                    _simulate_design_fast(n, delta, discordance, args.trials, 20260815 + len(rows) * 1000)
                )
    write_json(args.output, {"schema_version": 1, "results": rows})
    print(json.dumps({"status": "SIMULATION_COMPLETED", "rows": len(rows)}))


if __name__ == "__main__":
    main()
