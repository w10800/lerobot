#!/usr/bin/env python
"""Apply the frozen D-013 eligibility and checkpoint-selection rule."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

REGISTERED_STEPS = (1_000, 3_000, 5_000, 10_000, 20_000, 30_000)
ARMS = ("snap10", "snap2", "snap1")
EXPECTED_DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--pilot-development-manifest", type=Path, required=True)
    parser.add_argument("--factual-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def step_label(step: int) -> str:
    return f"{step // 1000}K"


def parse_training_metrics(log_text: str) -> tuple[dict[int, dict[str, float]], list[str]]:
    metrics: dict[int, dict[str, float]] = {}
    for step in REGISTERED_STEPS:
        matches = [line for line in log_text.splitlines() if f"step:{step_label(step)} " in line]
        if not matches:
            continue
        values = {}
        for name, raw in re.findall(r"([^\s:]+):([-+0-9.eE]+)", matches[-1]):
            with contextlib.suppress(ValueError):
                values[name] = float(raw)
        metrics[step] = values
    failure_patterns = {
        "traceback": r"\bTraceback \(most recent call last\)",
        "exception": r"\b(?:Exception|Error):",
        "oom": r"out of memory|CUDA OOM",
        "non_finite": r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])",
    }
    failures = [name for name, pattern in failure_patterns.items() if re.search(pattern, log_text, re.I)]
    return metrics, failures


def factual_gate(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record.get("status") != "FACTUAL_CHECKPOINT_EVALUATED":
        reasons.append("factual status incomplete")
    if record.get("task_count") != 40 or record.get("record_count") != 360:
        reasons.append("factual coverage incomplete")
    if not record.get("repeatability", {}).get("exact"):
        reasons.append("fixed-noise repeatability failed")
    if not all_finite(record):
        reasons.append("non-finite factual value")
    if set(record.get("checkpoint_files", {})) != {
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "training_step",
        "config",
    }:
        reasons.append("checkpoint provenance incomplete")
    return reasons


def replay_gate(record: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    reasons = []
    results = record.get("results", [])
    counts = Counter(result.get("arm") for result in results)
    successes = {
        arm: sum(bool(result.get("success")) for result in results if result.get("arm") == arm)
        for arm in ARMS
    }
    if record.get("status") != "MATURATION_REPLAY_COMPLETED":
        reasons.append("replay status incomplete")
    if record.get("case_count") != 40 or record.get("rollout_count") != 120:
        reasons.append("replay coverage incomplete")
    if any(counts[arm] != 40 for arm in ARMS):
        reasons.append("arm coverage incomplete")
    if any(result.get("status") != "COMPLETED" for result in results):
        reasons.append("non-completed rollout")
    if not all_finite(record):
        reasons.append("non-finite replay value")
    return reasons, successes


def pilot_rates(manifest: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    results = manifest.get("results", [])
    output = {}
    for arm in ("base10", "snap10", "snap2", "snap1"):
        subset = [result for result in results if result.get("arm") == arm]
        if len(subset) != 40:
            raise ValueError(f"Pilot development manifest lacks 40 {arm} records")
        count = sum(bool(result.get("success")) for result in subset)
        output[arm] = {"successes": count, "cases": 40, "success_rate": count / 40}
    return output


def write_reports(summary: dict[str, Any], output_root: Path) -> None:
    rows = summary["checkpoints"]
    table = [
        "| Step | Eligible | Factual loss | Snap-10 | Snap-2 | Snap-1 |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['step']} | {'yes' if row['eligible'] else 'no'} | "
            f"{row['factual_total_validation_loss']:.6f} | {row['successes']['snap10']}/40 | "
            f"{row['successes']['snap2']}/40 | {row['successes']['snap1']}/40 |"
        )
    comparison = [
        "# Maturation Checkpoint Comparison",
        "",
        "**Development-only baseline maturation; Formal Gate remains FAIL and CRP/CAG/KD remain HOLD.**",
        "",
        *table,
        "",
        f"Frozen rule selected step **{summary['selected_step']}**: maximum eligible Snap-1 success, "
        "with the earlier step winning an exact tie.",
    ]
    (output_root / "CHECKPOINT_COMPARISON.md").write_text("\n".join(comparison) + "\n")

    selected = summary["selected"]
    pilot = summary["pilot"]
    selected_text = [
        "# Selected Mature SnapFlow Candidate",
        "",
        f"**Selected checkpoint: {summary['selected_step']} steps.**",
        "",
        f"It was eligible and achieved {selected['successes']['snap1']}/40 Snap-1 successes. "
        "This is a development-selected candidate, not a confirmed checkpoint.",
        "",
        f"The old 1k pilot achieved {pilot['snap1']['successes']}/40 Snap-1 successes; the new "
        f"continuous run's 1k checkpoint achieved {rows[0]['successes']['snap1']}/40.",
        "",
        "The original Formal LIBERO Gate remains FAIL. No LIBERO-CF, condition-response, CRP, "
        "confirmation-set, or subgroup result was used for selection.",
    ]
    (output_root / "SELECTED_CHECKPOINT.md").write_text("\n".join(selected_text) + "\n")

    answers = summary["required_answers"]
    training = [
        "# SnapFlow Maturation Training Summary",
        "",
        "**VERIFIED development result; not a formal-gate or confirmation result.**",
        "",
        f"1. Old versus new 1k Snap-1: {answers['old_1k_snap1_successes']}/40 versus "
        f"{answers['new_1k_snap1_successes']}/40.",
        f"2. Snap-1 success trajectory: {answers['snap1_trajectory']}.",
        f"3. Selected Snap-10 versus frozen Base-10: {answers['selected_snap10_successes']}/40 versus "
        f"{answers['pilot_base10_successes']}/40.",
        f"4. Selected Snap-2 versus Snap-1: {answers['selected_snap2_successes']}/40 versus "
        f"{answers['selected_snap1_successes']}/40.",
        f"5. Development-selected mature candidate: step {summary['selected_step']}.",
        "",
        "CRP/CAG/KD training remains HOLD.",
    ]
    (output_root / "TRAINING_SUMMARY.md").write_text("\n".join(training) + "\n")


def main() -> None:
    args = parse_args()
    fixture = load(args.fixture_manifest)
    if fixture.get("status") != "MATURATION_FACTUAL_FIXTURE_FROZEN":
        raise ValueError("Fixture is not frozen")
    if fixture.get("registered_steps") != list(REGISTERED_STEPS):
        raise ValueError("Fixture registered steps mismatch")
    if set(fixture.get("data_order_prefix_sha256", {})) != {str(step) for step in REGISTERED_STEPS}:
        raise ValueError("Data-order prefix hashes incomplete")
    if fixture.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise ValueError("Dataset revision mismatch")

    log_text = args.training_log.read_text(errors="replace")
    training_metrics, training_failures = parse_training_metrics(log_text)
    pilot = pilot_rates(load(args.pilot_development_manifest))
    rows = []
    factual_records = {}
    replay_records = {}
    for step in REGISTERED_STEPS:
        factual_path = args.factual_dir / f"step_{step:06d}.json"
        replay_path = args.replay_dir / f"step_{step:06d}.json"
        factual = load(factual_path)
        replay = load(replay_path)
        if factual.get("checkpoint_step") != step or replay.get("checkpoint_step") != step:
            raise ValueError(f"Checkpoint-step mismatch at {step}")
        reasons = factual_gate(factual)
        replay_reasons, successes = replay_gate(replay)
        reasons.extend(replay_reasons)
        if step not in training_metrics:
            reasons.append("registered-step training metrics missing")
        reasons.extend(f"training log contains {failure}" for failure in training_failures)
        loss = float(factual["summary"]["1"]["total_validation_loss"])
        factual_records[step] = (factual_path, factual)
        replay_records[step] = (replay_path, replay)
        rows.append(
            {
                "step": step,
                "eligible": False,
                "ineligibility_reasons": reasons,
                "factual_total_validation_loss": loss,
                "successes": successes,
                "training_metrics": training_metrics.get(step),
                "data_order_prefix_sha256": fixture["data_order_prefix_sha256"][str(step)],
                "factual_manifest_sha256": file_sha256(factual_path),
                "replay_manifest_sha256": file_sha256(replay_path),
                "checkpoint_sha256": replay["checkpoint_sha256"],
            }
        )
    threshold = rows[0]["factual_total_validation_loss"]
    for row in rows:
        if row["factual_total_validation_loss"] > threshold:
            row["ineligibility_reasons"].append("factual loss exceeds same-run 1k")
        row["eligible"] = not row["ineligibility_reasons"]
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No checkpoint, including 1k, passed the frozen eligibility gates")
    selected = min(eligible, key=lambda row: (-row["successes"]["snap1"], row["step"]))
    summary = {
        "schema_version": 1,
        "status": "MATURATION_CHECKPOINT_SELECTED",
        "formal_gate_unchanged": True,
        "crp_cag_kd_training_hold": True,
        "selection_rule": "maximize eligible 40-case Snap-1 success; exact tie selects earlier step",
        "fixture_manifest_sha256": file_sha256(args.fixture_manifest),
        "training_log_sha256": file_sha256(args.training_log),
        "pilot": pilot,
        "checkpoints": rows,
        "selected_step": selected["step"],
        "selected": selected,
        "required_answers": {
            "old_1k_snap1_successes": pilot["snap1"]["successes"],
            "new_1k_snap1_successes": rows[0]["successes"]["snap1"],
            "snap1_trajectory": ", ".join(
                f"{row['step']}={row['successes']['snap1']}/40" for row in rows
            ),
            "selected_snap10_successes": selected["successes"]["snap10"],
            "pilot_base10_successes": pilot["base10"]["successes"],
            "selected_snap2_successes": selected["successes"]["snap2"],
            "selected_snap1_successes": selected["successes"]["snap1"],
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "checkpoint_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_reports(summary, args.output_root)
    products = [
        args.output_root / "TRAINING_SUMMARY.md",
        args.output_root / "CHECKPOINT_COMPARISON.md",
        args.output_root / "SELECTED_CHECKPOINT.md",
        args.output_root / "checkpoint_manifest.json",
    ]
    (args.output_root / "sha256_manifest.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in products)
    )
    print(json.dumps({"status": summary["status"], "selected_step": selected["step"]}))


if __name__ == "__main__":
    main()
