"""Pure clustered statistics for CRP-VLA Task 13 recovery."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def outcome_label(base_success: bool, snap_success: bool) -> str:
    return {
        (True, True): "preserved",
        (True, False): "harmful",
        (False, True): "student_only_success",
        (False, False): "both_fail",
    }[(bool(base_success), bool(snap_success))]


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    variance = ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / (
        len(left) + len(right) - 2
    )
    if variance <= 0:
        return float("nan")
    return float((left.mean() - right.mean()) / np.sqrt(variance))


def _quantiles(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(np.asarray(values), [0.025, 0.975])]


def contrast(
    rows: list[dict[str, Any]],
    key: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    harmful = [row for row in rows if row["outcome_label"] == "harmful"]
    preserved = [row for row in rows if row["outcome_label"] == "preserved"]
    left = np.asarray([float(row[key]) for row in harmful], dtype=np.float64)
    right = np.asarray([float(row[key]) for row in preserved], dtype=np.float64)
    if not len(left) or not len(right) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"Contrast {key} requires finite harmful and preserved case values")
    rng = np.random.default_rng(seed)
    case_estimates = []
    for _ in range(repeats):
        case_estimates.append(
            float(
                left[rng.integers(0, len(left), len(left))].mean()
                - right[rng.integers(0, len(right), len(right))].mean()
            )
        )

    tasks = sorted({str(row["task_id"]) for row in rows})
    by_task = {task: [row for row in rows if str(row["task_id"]) == task] for task in tasks}
    task_estimates: list[float] = []
    attempts = 0
    while len(task_estimates) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(tasks), len(tasks))
        sample = [row for index in sampled for row in by_task[tasks[int(index)]]]
        sample_left = [float(row[key]) for row in sample if row["outcome_label"] == "harmful"]
        sample_right = [float(row[key]) for row in sample if row["outcome_label"] == "preserved"]
        if sample_left and sample_right:
            task_estimates.append(float(np.mean(sample_left) - np.mean(sample_right)))
    if len(task_estimates) != repeats:
        raise RuntimeError(f"Unable to form {repeats} valid task-cluster bootstrap replicates for {key}")

    loto = []
    for task in tasks:
        selected = [row for row in rows if str(row["task_id"]) != task]
        sample_left = [float(row[key]) for row in selected if row["outcome_label"] == "harmful"]
        sample_right = [float(row[key]) for row in selected if row["outcome_label"] == "preserved"]
        loto.append(
            {
                "held_out_task": task,
                "difference": float(np.mean(sample_left) - np.mean(sample_right)),
                "harmful_n": len(sample_left),
                "preserved_n": len(sample_right),
            }
        )
    return {
        "metric": key,
        "harmful_n": int(len(left)),
        "preserved_n": int(len(right)),
        "harmful_mean": float(left.mean()),
        "preserved_mean": float(right.mean()),
        "mean_difference": float(left.mean() - right.mean()),
        "median_difference": float(np.median(left) - np.median(right)),
        "standardized_mean_difference": standardized_mean_difference(left, right),
        "case_bootstrap_ci95": _quantiles(case_estimates),
        "task_cluster_bootstrap_ci95": _quantiles(task_estimates),
        "leave_one_task_out": {
            "positive_count": sum(item["difference"] > 0 for item in loto),
            "total": len(loto),
            "minimum": min(item["difference"] for item in loto),
            "maximum": max(item["difference"] for item in loto),
            "records": loto,
        },
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "cluster_units": {"case": True, "task": True, "state_as_independent_sample": False},
    }


def auroc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.int8)
    values = np.asarray(list(scores), dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.int8)
    values = np.asarray(list(scores), dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("Average precision requires positives")
    order = np.argsort(-values, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(precision[sorted_y == 1].sum() / positives)


def _logistic_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = float(train_x.mean())
    scale = float(train_x.std())
    if scale == 0:
        scale = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        weight = np.clip(probability * (1 - probability), 1e-8, None)
        information = design.T @ (weight[:, None] * design) + np.eye(2) * 1e-8
        update = np.linalg.solve(information, design.T @ (train_y - probability))
        beta += update
        if float(np.linalg.norm(update)) < 1e-10:
            break
    eta = beta[0] + beta[1] * x_test
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def leave_one_task_out_predictions(
    rows: list[dict[str, Any]],
    predictors: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["outcome_label"] in {"harmful", "preserved"}]
    tasks = sorted({str(row["task_id"]) for row in selected})
    output = []
    for held_out in tasks:
        train = [row for row in selected if str(row["task_id"]) != held_out]
        test = [row for row in selected if str(row["task_id"]) == held_out]
        train_y = np.asarray([row["outcome_label"] == "harmful" for row in train], dtype=np.float64)
        if not len(test) or len(set(train_y.tolist())) != 2:
            raise RuntimeError(f"Invalid leave-one-task-out fold: {held_out}")
        predictions = {}
        for predictor in predictors:
            predictions[predictor] = _logistic_fit_predict(
                np.asarray([float(row[predictor]) for row in train]),
                train_y,
                np.asarray([float(row[predictor]) for row in test]),
            )
        for index, row in enumerate(test):
            output.append(
                {
                    "case_id": row["case_id"],
                    "task_id": row["task_id"],
                    "label": int(row["outcome_label"] == "harmful"),
                    **{predictor: float(predictions[predictor][index]) for predictor in predictors},
                }
            )
    return sorted(output, key=lambda row: str(row["case_id"]))


def prediction_summary(
    predictions: list[dict[str, Any]],
    raw_key: str,
    transition_key: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in predictions]

    def metrics(key: str, sample: list[dict[str, Any]]) -> dict[str, float]:
        y = [int(row["label"]) for row in sample]
        scores = [float(row[key]) for row in sample]
        return {
            "AUROC": auroc(y, scores),
            "AUPRC": average_precision(y, scores),
            "Brier": float(np.mean((np.asarray(scores) - np.asarray(y)) ** 2)),
        }

    raw = metrics(raw_key, predictions)
    transition = metrics(transition_key, predictions)
    tasks = sorted({str(row["task_id"]) for row in predictions})
    by_task = {task: [row for row in predictions if str(row["task_id"]) == task] for task in tasks}
    rng = np.random.default_rng(seed)
    differences = {"AUROC": [], "AUPRC": [], "Brier": []}
    attempts = 0
    while len(differences["AUROC"]) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(tasks), len(tasks))
        sample = [row for index in sampled for row in by_task[tasks[int(index)]]]
        if len({int(row["label"]) for row in sample}) != 2:
            continue
        left = metrics(raw_key, sample)
        right = metrics(transition_key, sample)
        differences["AUROC"].append(right["AUROC"] - left["AUROC"])
        differences["AUPRC"].append(right["AUPRC"] - left["AUPRC"])
        differences["Brier"].append(right["Brier"] - left["Brier"])
    if len(differences["AUROC"]) != repeats:
        raise RuntimeError("Unable to form requested prediction task bootstraps")
    return {
        "case_count": len(predictions),
        "harmful_count": int(sum(labels)),
        "preserved_count": int(len(labels) - sum(labels)),
        "raw_action": raw,
        "transition_eef_position_h5": transition,
        "transition_minus_raw": {
            metric: {
                "point": float(transition[metric] - raw[metric]),
                "task_bootstrap_ci95": _quantiles(values),
            }
            for metric, values in differences.items()
        },
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "cross_validation": "leave-one-task-out univariate logistic calibration",
    }
