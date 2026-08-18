#!/usr/bin/env python
"""Freeze the policy-free Task14R R0 reset-transaction protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from task14r_reset_transaction import build_task14r_r0_protocol, validate_task14r_r0_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = build_task14r_r0_protocol(implementation_parent_commit=git_value("rev-parse", "HEAD"))
    validate_task14r_r0_protocol(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": protocol["status"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
