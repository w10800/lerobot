#!/usr/bin/env python
"""Correct the Attempt001 post-hoc self-comparison and final report provenance."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_task13_recovery import overlap_rows
from task9_common import file_sha256, load_json
from task13_recovery_common import validate_task12_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--used-state-registry", type=Path, required=True)
    parser.add_argument("--attempt001-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def duplicate_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for field in ("initial_state_hash", "simulator_state_hash", "qpos_qvel_hash"):
        counts = Counter(str(case[field]) for case in cases)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        output[field] = {"duplicate_count": len(cases) - len(counts), "duplicate_values": duplicates}
    return output


def task12_identity_count(registry: dict[str, Any], identity_type: str) -> int:
    return len(
        {
            str(row["identity_value"])
            for row in registry["entries"]
            if row["dataset"] in {"task12_mechanism_query", "task12_preservation_probe"}
            and row["identity_type"] == identity_type
        }
    )


def main() -> None:
    args = parse_args()
    if git_value("status", "--porcelain"):
        raise RuntimeError("Task 13 audit correction requires a clean worktree")
    registry = load_json(args.used_state_registry)
    validate_task12_registry(registry)
    attempt1 = load_json(args.attempt001_manifest)
    cases = attempt1.get("cases", [])
    if len(cases) != 400:
        raise RuntimeError("Attempt001 manifest cardinality drift")
    overlaps = overlap_rows(cases, registry, include_attempt1_dataset=False)
    duplicates = duplicate_summary(cases)
    by_dataset = Counter(row["dataset"] for row in overlaps)
    root = args.output_root.resolve()
    posthoc_path = root / "ATTEMPT001_POSTHOC_OVERLAP_AUDIT.md"
    final_path = root / "TASK13_RECOVERY_FINAL_REPORT.md"
    if not posthoc_path.exists() or not final_path.exists():
        raise FileNotFoundError("Task 13 recovery reports are incomplete")
    old_posthoc_sha = file_sha256(posthoc_path)
    posthoc_path.write_text(
        "# Attempt001 Post-hoc Overlap Audit\n\n"
        "**Status: ATTEMPT001_POSTHOC_DISJOINT_BUT_PROCEDURALLY_INVALID**\n\n"
        f"- Attempt001 cases checked: `{len(cases)}`.\n"
        f"- Task 12 raw query records checked: `{registry['task12_authoritative_query_record_count']}`.\n"
        f"- Unique Task 12 archived state payload hashes: `{registry['task12_unique_state_payload_hash_count']}`.\n"
        f"- Unique Task 12 state-blob identities checked: `{task12_identity_count(registry, 'state_blob_sha256')}`.\n"
        f"- Attempt001 internal duplicate initial/simulator/qpos-qvel identities: `"
        f"{duplicates['initial_state_hash']['duplicate_count']} / "
        f"{duplicates['simulator_state_hash']['duplicate_count']} / "
        f"{duplicates['qpos_qvel_hash']['duplicate_count']}`.\n"
        f"- old dev40 overlap records: `{by_dataset['old_dev40']}`.\n"
        f"- formal100 overlap records: `{by_dataset['formal100']}`.\n"
        f"- Confirmation1200 overlap records: `{by_dataset['task11_confirmation1200']}`.\n"
        f"- Task12 mechanism/probe overlap records: `{by_dataset['task12_mechanism_or_probe']}`.\n"
        f"- Attempt001 total prior/Task12 overlap records: `{len(overlaps)}`.\n"
        "- Attempt001 self-dataset entries were deliberately excluded from the comparison support.\n"
        "- Prospective primary validity remains: `NO`, regardless of the zero post-hoc overlap.\n\n"
        + (json.dumps(overlaps, indent=2) + "\n" if overlaps else "No overlap identities found.\n")
    )
    final_text = final_path.read_text()
    final_text, replacements = re.subn(
        r"4\. Attempt001 versus .*?overlap.*?\n",
        f"4. Attempt001 versus prior/Task12 datasets overlap records: `{len(overlaps)}`.\n",
        final_text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("Task 13 final report overlap answer could not be corrected")
    final_path.write_text(final_text)
    correction_path = root / "TASK13_RECOVERY_AUDIT_CORRECTION.md"
    correction_path.write_text(
        "# Task 13 Recovery Audit Correction\n\n"
        "**Status: VERIFIED REPORT-LOGIC CORRECTION**\n\n"
        "The initial Attempt001 post-hoc report incorrectly included `task13_attempt001_initial` "
        "as a comparison dataset while auditing Attempt001 itself. This produced exactly 1,200 "
        "self-matches (400 cases x three initial-state identity fields), not overlap with Task12 or "
        "another prior support. The corrected audit explicitly excludes the self-dataset and finds "
        f"`{len(overlaps)}` prior/Task12 overlap records.\n\n"
        "This correction does not restore prospective validity to Attempt001, does not change any "
        "Attempt002 case, outcome, query, branch, H1/H2/H3 statistic, or the final "
        "`PARTIAL_MECHANISM_REPLICATION` decision.\n\n"
        f"- Superseded post-hoc report SHA-256: `{old_posthoc_sha}`.\n"
        f"- Corrected post-hoc report SHA-256: `{file_sha256(posthoc_path)}`.\n"
        f"- Correction code commit: `{git_value('rev-parse', 'HEAD')}`.\n"
        f"- Corrected unix ns: `{time.time_ns()}`.\n"
    )
    provenance_path = root / "TASK13_RECOVERY_PROVENANCE.json"
    provenance = load_json(provenance_path)
    provenance.update(
        {
            "audit_correction_commit": git_value("rev-parse", "HEAD"),
            "attempt001_corrected_posthoc_overlap_records": len(overlaps),
            "attempt001_self_dataset_excluded": True,
            "audit_correction_sha256": file_sha256(correction_path),
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    provenance_md = root / "TASK13_RECOVERY_PROVENANCE.md"
    provenance_md.write_text(
        provenance_md.read_text()
        + "\n## Audit correction\n\n"
        + f"- Attempt001 corrected prior/Task12 overlap records: `{len(overlaps)}`.\n"
        + "- Attempt001 self-dataset excluded: `YES`.\n"
        + f"- Correction artifact SHA-256: `{file_sha256(correction_path)}`.\n"
    )
    hash_candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name.startswith(("ATTEMPT002_", "TASK13_RECOVERY_"))
        and path.name != "TASK13_RECOVERY_SHA256SUMS.txt"
    ] + sorted((root / "figures").glob("*"))
    (root / "TASK13_RECOVERY_SHA256SUMS.txt").write_text(
        "\n".join(f"{file_sha256(path)}  {path.relative_to(root)}" for path in sorted(hash_candidates))
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "TASK13_RECOVERY_AUDIT_CORRECTED",
                "attempt001_overlap_records": len(overlaps),
                "scientific_decision_changed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
