#!/usr/bin/env python
"""Freeze the policy-free Task14R R0 reset-transaction protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from task14r_reset_transaction import (
    TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0_SOURCE_FILES,
    build_task14r_r0_protocol,
    validate_task14r_r0_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_files_sha256() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        for relative in TASK14R_R0_SOURCE_FILES
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hashes = source_files_sha256()
    protocol = build_task14r_r0_protocol(
        implementation_parent_commit=TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=source_hashes,
    )
    validate_task14r_r0_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=source_hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": protocol["status"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
