#!/usr/bin/env python
"""Pure helpers for the CRP-VLA Task 12 mechanism audit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

TASK12_BOOTSTRAP_SEED = 20260817
TASK12_BOOTSTRAP_REPEATS = 10_000
COUNTERFACTUAL_HORIZONS = (1, 3, 5, 10)
OUTCOME_STRATA = (
    "concordant_success",
    "snap_only_success",
    "base_only_success",
    "concordant_failure",
)


def outcome_stratum(base_success: bool, snap_success: bool) -> str:
    mapping = {
        (True, True): "concordant_success",
        (False, True): "snap_only_success",
        (True, False): "base_only_success",
        (False, False): "concordant_failure",
    }
    return mapping[(bool(base_success), bool(snap_success))]


def validate_task11_pass(stats: dict[str, Any]) -> None:
    if stats.get("status") != "FORMAL_CONFIRMATION1200_PASS":
        raise RuntimeError("Task 12 PASS branch requires the frozen Task 11 PASS result")
    if stats.get("case_count") != 1200 or not stats.get("noninferiority_pass"):
        raise RuntimeError("Task 11 PASS cardinality/decision mismatch")
    if stats.get("frozen_protocol_modified") is not False:
        raise RuntimeError("Task 11 reports a modified frozen protocol")


def contact_set(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    for item in rows:
        left, right = sorted((str(item.get("left")), str(item.get("right"))))
        output.add((left, right))
    return output


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def bootstrap_median_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    repeats: int = TASK12_BOOTSTRAP_REPEATS,
    seed: int = TASK12_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap median(left)-median(right) as descriptive mechanism evidence."""
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    left_array = left_array[np.isfinite(left_array)]
    right_array = right_array[np.isfinite(right_array)]
    if not len(left_array) or not len(right_array):
        return {"available": False, "reason": "one or both groups empty"}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        left_sample = left_array[rng.integers(0, len(left_array), size=len(left_array))]
        right_sample = right_array[rng.integers(0, len(right_array), size=len(right_array))]
        estimates[index] = np.median(left_sample) - np.median(right_sample)
    return {
        "available": True,
        "left_n": int(len(left_array)),
        "right_n": int(len(right_array)),
        "point": float(np.median(left_array) - np.median(right_array)),
        "ci95": [float(value) for value in np.quantile(estimates, [0.025, 0.975])],
        "repeats": repeats,
        "seed": seed,
        "confirmatory": False,
    }


def progress_bin(index: int, length: int, bins: int = 10) -> int:
    if length <= 1:
        return 0
    return min(bins - 1, int(index * bins / length))
