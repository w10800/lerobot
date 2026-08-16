#!/usr/bin/env python
"""Build the Task 13 recovery used-state registry from authoritative raw artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from task9_common import file_sha256, load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse registry construction inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal100", type=Path, required=True)
    parser.add_argument("--old-dev40", type=Path, required=True)
    parser.add_argument("--confirmation1200", type=Path, required=True)
    parser.add_argument("--task11-trace-manifest", type=Path, required=True)
    parser.add_argument("--task12-query-root", type=Path, required=True)
    parser.add_argument("--task12-branch-root", type=Path, required=True)
    parser.add_argument("--task12-probe-protocol", type=Path, required=True)
    parser.add_argument("--attempt001-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add(
    entries: list[dict[str, Any]],
    *,
    dataset: str,
    identity_type: str,
    identity_value: Any,
    source: Path,
    case_id: str | None = None,
    arm: str | None = None,
    replan_index: int | None = None,
) -> None:
    """Append one typed state identity without conflating hash semantics."""
    if identity_value in (None, ""):
        return
    entries.append(
        {
            "dataset": dataset,
            "identity_type": identity_type,
            "identity_value": str(identity_value),
            "case_id": case_id,
            "arm": arm,
            "replan_index": replan_index,
            "source_path": str(source.resolve()),
        }
    )


def add_manifest(entries: list[dict[str, Any]], path: Path, dataset: str) -> None:
    """Add available initial-state identities from one manifest."""
    payload = load_json(path)
    rows = payload.get("cases", payload.get("results", []))
    for row in rows:
        case_id = row.get("case_id")
        for field in (
            "initial_state_hash",
            "simulator_state_hash",
            "initial_sim_state_sha256",
            "qpos_qvel_hash",
        ):
            add(
                entries,
                dataset=dataset,
                identity_type=field,
                identity_value=row.get(field),
                source=path,
                case_id=case_id,
            )


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Build a provenance-bearing registry from raw Task 12 state records."""
    entries: list[dict[str, Any]] = []
    for path, dataset in (
        (args.old_dev40, "old_dev40"),
        (args.formal100, "formal100"),
        (args.confirmation1200, "task11_confirmation1200"),
        (args.attempt001_manifest, "task13_attempt001_initial"),
    ):
        add_manifest(entries, path, dataset)

    trace_records = load_json(args.task11_trace_manifest)["records"]
    trace_by_key = {(row["case_id"], row["arm"]): Path(row["trace_path"]) for row in trace_records}
    top_cache: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    probe = load_json(args.task12_probe_protocol)
    probe_cases = set(probe["split"]["train_case_ids"]) | set(probe["split"]["heldout_case_ids"])
    query_files = sorted(args.task12_query_root.glob("*/same_state_metrics.jsonl"))
    if len(query_files) != 40:
        raise RuntimeError(f"Expected 40 authoritative query shards, found {len(query_files)}")
    query_records = 0
    query_state_hashes: set[str] = set()
    for query_path in query_files:
        for line in query_path.open():
            row = json.loads(line)
            query_records += 1
            case_id = str(row["case_id"])
            arm = str(row["origin_arm"])
            replan_index = int(row["replan_index"])
            state_hash = str(row["state_payload_sha256"])
            query_state_hashes.add(state_hash)
            key = (case_id, arm)
            if key not in top_cache:
                top = load_json(trace_by_key[key])
                top_cache[key] = {int(record["replan_index"]): record for record in top["replan_records"]}
            replan = top_cache[key][replan_index]
            metadata_path = Path(replan["path"]) / "metadata.json"
            metadata = load_json(metadata_path)
            if metadata["integrity"]["simulator_state_sha256"] != state_hash:
                raise RuntimeError(
                    f"Task 12 query/replan state hash mismatch: {case_id}/{arm}/{replan_index}"
                )
            simulator = metadata["payload_skeleton"]["simulator_state"]
            values = {
                "task12_state_payload_sha256": state_hash,
                "state_blob_sha256": simulator["state_blob_sha256"],
                "qpos_sha256": simulator["qpos_sha256"],
                "qvel_sha256": simulator["qvel_sha256"],
            }
            for identity_type, identity_value in values.items():
                add(
                    entries,
                    dataset="task12_mechanism_query",
                    identity_type=identity_type,
                    identity_value=identity_value,
                    source=metadata_path,
                    case_id=case_id,
                    arm=arm,
                    replan_index=replan_index,
                )
                if case_id in probe_cases:
                    add(
                        entries,
                        dataset="task12_preservation_probe",
                        identity_type=identity_type,
                        identity_value=identity_value,
                        source=metadata_path,
                        case_id=case_id,
                        arm=arm,
                        replan_index=replan_index,
                    )

    if query_records == 0 or not query_state_hashes:
        raise RuntimeError("Task 12 authoritative raw state registry is empty")
    branch_files = sorted(args.task12_branch_root.glob("*/branch_transition_metrics.jsonl"))
    if len(branch_files) != 40:
        raise RuntimeError(f"Expected 40 authoritative branch shards, found {len(branch_files)}")
    branch_records = 0
    for branch_path in branch_files:
        for line in branch_path.open():
            row = json.loads(line)
            branch_records += 1
            add(
                entries,
                dataset="task12_counterfactual_branch",
                identity_type="branch_initial_state_sha256",
                identity_value=row["initial_state_sha256"],
                source=branch_path,
                case_id=row["case_id"],
                arm=row["origin_arm"],
                replan_index=int(row["replan_index"]),
            )
    sources = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (
            ("formal100", args.formal100),
            ("old_dev40", args.old_dev40),
            ("confirmation1200", args.confirmation1200),
            ("task11_trace_manifest", args.task11_trace_manifest),
            ("task12_probe_protocol", args.task12_probe_protocol),
            ("attempt001_manifest", args.attempt001_manifest),
        )
    }
    sources["task12_query_shards"] = {
        "paths": [str(path.resolve()) for path in query_files],
        "sha256": [file_sha256(path) for path in query_files],
    }
    sources["task12_branch_shards"] = {
        "paths": [str(path.resolve()) for path in branch_files],
        "sha256": [file_sha256(path) for path in branch_files],
    }
    return {
        "schema_version": 1,
        "status": "USED_STATE_REGISTRY_COMPLETE",
        "authoritative_sources": sources,
        "task12_authoritative_query_record_count": query_records,
        "task12_unique_state_payload_hash_count": len(query_state_hashes),
        "task12_duplicate_state_payload_record_count": query_records - len(query_state_hashes),
        "task12_counterfactual_record_count": branch_records,
        "identity_semantics_kept_separate": True,
        "entry_count": len(entries),
        "entries_by_dataset": dict(sorted(Counter(row["dataset"] for row in entries).items())),
        "entries": entries,
    }


def main() -> None:
    """Write the immutable used-state registry."""
    args = parse_args()
    registry = build_registry(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": registry["status"],
                "task12_records": registry["task12_authoritative_query_record_count"],
                "task12_unique_states": registry["task12_unique_state_payload_hash_count"],
                "entries": registry["entry_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
