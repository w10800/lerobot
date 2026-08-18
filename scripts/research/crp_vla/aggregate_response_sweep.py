#!/usr/bin/env python
"""Aggregate registered counterfactual diagnostics and plot the compression gap."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--teacher-response-threshold", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(path.read_text()) for path in args.inputs]
    if not records:
        raise ValueError("At least one diagnostic record is required")
    for record in records:
        if record.get("repository_dirty"):
            raise ValueError(f"Dirty repository record is not admissible: {record['action_archive']}")

    retained = [
        record
        for record in records
        if record["metrics"]["global"]["teacher_response_norm"] >= args.teacher_response_threshold
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in retained:
        grouped[f"{record['task_group']} / {record['intervention_type']}"].append(record)

    group_summary = {}
    for label, group in sorted(grouped.items()):
        group_summary[label] = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for metric in (
                "response_norm_ratio",
                "response_cosine",
                "normalized_response_error",
                "factual_action_mse",
                "teacher_response_norm",
            )
            if (values := [item["metrics"]["global"][metric] for item in group])
        }

    summary = {
        "schema_version": 1,
        "status": "ANALYZED",
        "teacher_response_threshold": args.teacher_response_threshold,
        "input_count": len(records),
        "retained_count": len(retained),
        "excluded_below_teacher_threshold": len(records) - len(retained),
        "task_groups": sorted({record["task_group"] for record in retained}),
        "intervention_types": sorted({record["intervention_type"] for record in retained}),
        "noise_seeds": sorted({record["noise_seed"] for record in retained}),
        "groups": group_summary,
        "inputs": [str(path.resolve()) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    labels = list(group_summary)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 3.5 * len(labels)), 4.5))
    for axis, metric, title in (
        (axes[0], "response_norm_ratio", "Conditional-response norm ratio"),
        (axes[1], "normalized_response_error", "Normalized conditional-response error"),
    ):
        means = [group_summary[label][metric]["mean"] for label in labels]
        stds = [group_summary[label][metric]["std"] for label in labels]
        axis.bar(x, means, yerr=stds, capsize=4)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
