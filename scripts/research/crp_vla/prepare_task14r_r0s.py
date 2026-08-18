#!/usr/bin/env python
"""Freeze the policy-free Task14R R0S renderer-shift protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from task14r_renderer_shift import (
    TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
    TASK14R_R0S_SOURCE_FILES,
    build_task14r_r0s_protocol,
    validate_task14r_r0s_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_files_sha256() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        for relative in TASK14R_R0S_SOURCE_FILES
    }


def main() -> None:
    args = parse_args()
    digest_output = args.output.with_suffix(".sha256")
    if args.output.exists() or digest_output.exists():
        raise FileExistsError(args.output if args.output.exists() else digest_output)
    hashes = source_files_sha256()
    protocol = build_task14r_r0s_protocol(
        implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        source_files_sha256=hashes,
    )
    validate_task14r_r0s_protocol(
        protocol,
        expected_implementation_parent_commit=TASK14R_R0S_IMPLEMENTATION_PARENT_COMMIT,
        expected_source_files_sha256=hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    protocol_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    digest_output.write_text(f"{protocol_sha256}  {args.output.name}\n")
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "output": str(args.output.resolve()),
                "sha256": protocol_sha256,
                "sha256_file": str(digest_output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
