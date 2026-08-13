#!/usr/bin/env python
"""Compute CRP metrics from a deterministic action archive."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.smolvla_crp.response_metrics import compute_response_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True, help="NPZ with four action arrays")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    required = {
        "teacher_factual",
        "teacher_counterfactual",
        "student_factual",
        "student_counterfactual",
    }
    with np.load(args.actions, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Action archive is missing arrays: {sorted(missing)}")
        metrics = compute_response_metrics(
            *(
                torch.from_numpy(archive[key])
                for key in sorted(
                    required,
                    key=(
                        "teacher_factual",
                        "teacher_counterfactual",
                        "student_factual",
                        "student_counterfactual",
                    ).index,
                )
            )
        )
    output = {
        "schema_version": 1,
        "status": "ANALYZED",
        "source": str(args.actions.resolve()),
        "metrics": jsonify(metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
